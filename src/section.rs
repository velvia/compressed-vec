/// A BinaryVector MAY consist of multiple sections.  Each section can represent
/// potentially different encoding parameters (bit widths, sparsity, etc.) and
/// has its own header to allow for quickly skipping ahead even when different
/// sections are encoded differently.   Or, one section may represent null data.
///
/// There are two varieties of sections represented.  See `SectionWriter` for variable-sized
/// sections, and see `FixedSection` for constant-length (number of elements) sections.
///
/// The code uses Scroll to ensure efficient encoding but one that works across platforms and endianness.

use core::marker::PhantomData;

use crate::error::CodingError;
use crate::nibblepacking;
use crate::nibblepack_simd;
use crate::sink::*;

use std::ops::Add;
use std::convert::TryFrom;

use enum_dispatch::enum_dispatch;
use num::{PrimInt, Unsigned, Num, Bounded};
use packed_simd::{u32x8, u64x8};
use scroll::{ctx, Endian, Pread, Pwrite, LE};


/// For FixedSections this represents the first (and maybe only) byte of the section.
/// For SectionHeader based sections this is the byte at offset 4 into the header.
/// FixedSections are generic, they do not contain type information which is in the vector type.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SectionType {
    Null = 0,                 // FIXED_LEN unavailable or null elements in a row
    NibblePackedMedium = 1,   // Nibble-packed u64/u32's, total size < 64KB
    DeltaNPMedium      = 3,   // Nibble-packed u64/u32's, delta encoded, total size < 64KB
    Constant           = 5,   // Constant value section
}

impl TryFrom<u8> for SectionType {
    type Error = CodingError;
    fn try_from(n: u8) -> Result<SectionType, Self::Error> {
        match n {
            0 => Ok(SectionType::Null),
            1 => Ok(SectionType::NibblePackedMedium),
            3 => Ok(SectionType::DeltaNPMedium),
            5 => Ok(SectionType::Constant),
            _ => Err(CodingError::InvalidSectionType(n)),
        }
    }
}

impl SectionType {
    pub fn as_num(&self) -> u8 { *self as u8 }
}

// This is a royal pain that Scroll cannot derive codecs for simple enums.  :/
impl<'a> ctx::TryFromCtx<'a, Endian> for SectionType {
  type Error = scroll::Error;
  fn try_from_ctx (src: &'a [u8], ctx: Endian) -> Result<(SectionType, usize), Self::Error> {
      u8::try_from_ctx(src, ctx).and_then(|(n, bytes)| {
          SectionType::try_from(n).map(|s| (s, bytes))
              .map_err(|e| match e {
                  CodingError::InvalidSectionType(n) =>
                      scroll::Error::Custom(format!("InvalidSectionType {:?}", n)),
                  _ => scroll::Error::Custom("Unknown error".to_string())
              })
      })
  }
}

impl ctx::TryIntoCtx<Endian> for &SectionType {
    type Error = scroll::Error;
    fn try_into_ctx(self, buf: &mut [u8], ctx: Endian) -> Result<usize, Self::Error> {
        u8::try_into_ctx(self.as_num(), buf, ctx)
    }
}


/// `SectionHeader` represents FiloDB-style HistogramColumn sections.  Each section has a 5-byte header that
/// encapsulates number of bytes, number of elements, and the section type.  The idea is that sections can
/// denote different encodings, and be large enough to allow quick skipping over elements for faster access.
/// In FiloDB sections also denote things like counter drops/resets, which could also be implemented.
#[derive(Copy, Clone, Debug, PartialEq, Pread, Pwrite)]
pub struct SectionHeader {
    num_bytes: u16,         // Number of bytes in section after this header
    num_elements: u16,      // Number of elements.
    typ: SectionType,
}

/// Result: (bytes_written, elements_written)
type CodingResult = Result<(u16, u16), CodingError>;

/// SectionWriter stores state for active writing of multiple SectionHeader-based sections in a vector.
/// It manages rollover from one section to another when there's not enough space.
/// The main API is `add_64kb` which uses a closure to fill in section contents without copying.
///
/// Example which adds 8 0xff elements and returns an error if there isn't enough space:
/// ```
/// # use compressed_vec::section::*;
/// # use compressed_vec::error::CodingError;
/// let mut buf = [0u8; 1024];
/// let mut writer = SectionWriter::new(&mut buf, 256);
/// let res = writer.add_64kb(SectionType::Null, |writebuf: &mut [u8], _| {
///     if writebuf.len() < 8 { Err(CodingError::NotEnoughSpace) }
///     else {
///         for n in 0..8 { writebuf[n] = 0xff; }
///         Ok((8, 8))
///     }
/// });
/// ```
#[derive(Debug)]
pub struct SectionWriter<'a> {
    write_buf: &'a mut [u8],     // Be sure length is total capacity to write
    cur_pos: usize,              // Current write position within buffer
    cur_header_pos: usize,       // Buffer position of current section header
    max_elements_per_sect: u16,  // Max # elements within a single section
    cur_header: SectionHeader
}

impl<'a> SectionWriter<'a> {
    /// Default constructor given mutable buffer and initial position of 0
    pub fn new(buf: &'a mut [u8], max_elements_per_sect: u16) -> Self {
        Self { write_buf: buf,
               cur_pos: 0,     // 0 means no section initialized
               cur_header_pos: 0,
               max_elements_per_sect,
               cur_header: SectionHeader { num_bytes: 0, num_elements: 0, typ: SectionType::Null }
        }
    }

    pub fn cur_pos(&self) -> usize { self.cur_pos }

    fn init_new_section(&mut self, sect_type: SectionType) -> CodingResult {
        self.cur_header.num_bytes = 0;
        self.cur_header.num_elements = 0;
        self.cur_header.typ = sect_type;
        self.cur_header_pos = self.cur_pos;
        let (bytes_written, _) = self.update_sect_header()?;
        self.cur_pos += bytes_written as usize;
        Ok((bytes_written, 0))
    }

    fn update_sect_header(&mut self) -> CodingResult {
        let bytes_written = self.write_buf.pwrite_with(self.cur_header, self.cur_header_pos, LE)?;
        Ok((bytes_written as u16, 0))
    }

    /// Adds an "element" by filling in mutable buffer up to 64KB in length.
    /// Method advances to a new section if necessary.
    /// Closure must be passed which is given &mut [u8] and returns WriteTaskResult.
    /// The filler returns how many bytes, elements were written - this accommodates variable-length encoding.
    /// If given slice is not large enough, then method may advance to next section
    /// which should give more room to grow.
    /// sect_type is used to fill in new section
    pub fn add_64kb<F>(&mut self, sect_type: SectionType, filler: F) -> CodingResult
        where F: Fn(&mut [u8], usize) -> CodingResult
    {
        // If buffer empty / no section initialized, go ahead initialize it
        if self.cur_pos == 0 { self.init_new_section(sect_type)?; }

        let elements_left = self.max_elements_per_sect - self.cur_header.num_elements;
        // Smaller of how much left in section vs how much left in input buffer
        let bytes_left = std::cmp::min(65535 - self.cur_header.num_bytes as usize,
                                       self.write_buf.len() - self.cur_pos);

        // Call filler func once.  If not enough space, try to allocate new section before giving up
        let writable_bytes = &mut self.write_buf[self.cur_pos..self.cur_pos + bytes_left];
        let filled_res = filler(writable_bytes, elements_left as usize);
        match filled_res {
            Ok((bytes_written, elements_written)) => {
                assert!(elements_written <= elements_left);
                // Update section header as well as other internal pointers
                self.cur_header.num_bytes += bytes_written;
                self.cur_header.num_elements += elements_written;
                self.cur_pos += bytes_written as usize;

                self.update_sect_header()?;
                Ok((bytes_written, elements_written))
            },
            Err(CodingError::NotEnoughSpace) => {
                // Try to write a new section
                self.init_new_section(sect_type)?;

                // Now try writing again
                self.add_64kb(sect_type, filler)
            }
            e @ Err(_) => return e,
        }
    }
}

// This should really be 256 for SIMD query filtering purposes.
// Don't adjust this unless you know what you're doing
pub const FIXED_LEN: usize = 256;

/// A FixedSection is a section with a fixed number of elements.
/// Thus a compressed vector could be made of a number of FixedSections.
/// Currently the implementation is tied to 256 elements.
///
/// Each section begins with a 1-byte SectionType enum, after which each one defines its own format.
///
/// NOTE: To avoid needing to box trait implementations for things like `FixedSectIterator`, we
/// use [enum_dispatch](https://docs.rs/enum_dispatch/0.2.2/enum_dispatch/); methods can be called on
/// `FixedSectEnum` and `try_into()` used to convert back to original values.
///
#[enum_dispatch]
pub trait FixedSection {
    /// The number of bytes total in this section including the section type header byte
    fn num_bytes(&self) -> usize;
    fn num_elements(&self) -> usize { FIXED_LEN }

    /// Return the byte slice corresponding to section bytes, if available
    fn sect_bytes(&self) -> Option<&[u8]>;
}

/// A FixedSectEnum is an enum over different FixedSection implementations, for the purpose of very fast,
/// inlineable iteration over different section types without resorting to dynamic method calls.
#[enum_dispatch(FixedSection)]
#[derive(Debug, PartialEq)]
pub enum FixedSectEnum<'buf, T: VectBase> {
    NullFixedSect,
    NibblePackMedFixedSect(NibblePackMedFixedSect<'buf, T>),
    DeltaNPMedFixedSect(DeltaNPMedFixedSect<'buf, T>),
    ConstFixedSect(ConstFixedSect<'buf, T>),
}

impl<'buf, T: VectBase> FixedSectEnum<'buf, T> {
    /// Decodes this section based on items of type T to a Sink.  This is the main decoding API.
    /// Note that you need to specify an explicit base type as FixedSectEnums are typeless.
    /// For example, to write to the generic section sink which materializes every value in a section:
    /// ```
    /// # use compressed_vec::section::{FixedSectEnum, SectionType};
    /// # use std::convert::TryFrom;
    /// # let mut sect_bytes = [0u8; 256];
    /// # sect_bytes[0] = SectionType::NibblePackedMedium as u8;
    /// # sect_bytes[1] = 253;
    ///     let sect = FixedSectEnum::<u32>::try_from(&sect_bytes[..]).unwrap();
    ///     let mut sink = compressed_vec::sink::U32_256Sink::new();
    ///     sect.decode(&mut sink).unwrap();
    ///     println!("{:?}", sink.values.iter().count());
    /// ```
    #[inline]
    pub fn decode<S>(self, sink: &mut S) -> Result<(), CodingError>
    where S: Sink<T::SI> {
        T::Utils::decode_to_sink(self, sink)
    }

    /// Is this a null section?
    #[inline]
    pub fn is_null(&self) -> bool {
        match self {
            FixedSectEnum::NullFixedSect(_) => true,
            _ => false,
        }
    }
}

impl<'buf, T: VectBase> TryFrom<&'buf [u8]> for FixedSectEnum<'buf, T> {
    type Error = CodingError;
    /// Tries to extract a FixedSection from a slice, whose first byte contains the section type byte.
    /// The length of the slice should contain at least all the data in the section.
    fn try_from(s: &'buf [u8]) -> Result<FixedSectEnum<'buf, T>, CodingError> {
        if s.len() <= 0 { return Err(CodingError::InputTooShort) }
        let sect_type = SectionType::try_from(s[0])?;
        match sect_type {
            SectionType::Null => Ok((NullFixedSect {}).into()),
            SectionType::NibblePackedMedium =>
                NibblePackMedFixedSect::try_from(s).map(|sect| sect.into()),
            SectionType::DeltaNPMedium =>
                DeltaNPMedFixedSect::try_from(s).map(|sect| sect.into()),
            SectionType::Constant =>
                ConstFixedSect::try_from(s).map(|sect| sect.into()),
        }
    }
}

/// Reader trait for FixedSections, has some common methods for iteration and extraction of values
pub trait FixedSectReader<T: VectBase>: FixedSection {
    /// Decodes values from this section to a sink.
    /// This is the most generic method of processing data from a section.
    /// For example, to get an iterator out:
    /// ```
    /// # use compressed_vec::section::{FixedSectReader, NibblePackMedFixedSect};
    /// # use compressed_vec::nibblepack_simd;
    /// # let mut sect_bytes = [0u8; 256];
    /// # sect_bytes[1] = 253;
    ///     let sect = NibblePackMedFixedSect::<u32>::try_from(&sect_bytes[..]).unwrap();
    ///     let mut sink = compressed_vec::sink::U32_256Sink::new();
    ///     sect.decode_to_sink(&mut sink).unwrap();
    ///     println!("{:?}", sink.values.iter().count());
    /// ```
    fn decode_to_sink<Output>(&self, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<T::SI>;
}

/// Utility trait for FixedSectReader/Writers, to help:
/// - map FixedSectReader from a FixedSectEnum
/// - for a const byte width
/// - connect generic section implementations to type-specific methods such as Scroll pread/write
pub trait FSUtils<T: VectBase> {
    const BYTE_WIDTH: usize;
    fn decode_to_sink<Output>(e: FixedSectEnum<T>, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<T::SI>;

    /// Read a primitive T from a buffer at an offset, little-endian
    fn read_le_offset<'a>(buf: &'a [u8], offset: usize) -> Result<T, scroll::Error>;

    /// Write a primitive T to a buffer at an offset, little-endian
    fn write_le_offset<'a>(buf: &'a mut [u8], offset: usize, value: T) -> Result<usize, scroll::Error>;

    /// Generic: decoding to sink method for a single encoded NibblePacked 8 octets of data
    fn nibblepack_decode<'a, S: Sink<T::SI>>(buf: &'a [u8], sink: &mut S) -> Result<&'a [u8], CodingError>;
}

pub struct FSUtilsMarker {}

impl<'buf> FSUtils<u32> for FSUtilsMarker {
    const BYTE_WIDTH: usize = 4;

    #[inline]
    fn decode_to_sink<Output>(e: FixedSectEnum<u32>, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<u32x8> {
        match e {
            FixedSectEnum::NullFixedSect(nfs) => FixedSectReader::<u32>::decode_to_sink(&nfs, output),
            FixedSectEnum::NibblePackMedFixedSect(fs) => fs.decode_to_sink(output),
            FixedSectEnum::DeltaNPMedFixedSect(fs)    => fs.decode_to_sink(output),
            FixedSectEnum::ConstFixedSect(cs)         => cs.decode_to_sink(output),
            // _ => Err(CodingError::InvalidFormat(format!("Section {:?} invalid for u32", e))),
        }
    }

    #[inline]
    fn read_le_offset<'a>(buf: &'a [u8], offset: usize) -> Result<u32, scroll::Error> {
        buf.pread_with(offset, LE)
    }

    #[inline]
    fn write_le_offset<'a>(buf: &'a mut [u8], offset: usize, value: u32) -> Result<usize, scroll::Error> {
        buf.pwrite_with(value, offset, LE)
    }

    #[inline]
    fn nibblepack_decode<'a, S: Sink<u32x8>>(buf: &'a [u8], sink: &mut S) -> Result<&'a [u8], CodingError> {
        nibblepack_simd::unpack8_u32_simd(buf, sink)
    }
}

impl<'buf> FSUtils<u64> for FSUtilsMarker {
    const BYTE_WIDTH: usize = 8;

    #[inline]
    fn decode_to_sink<Output>(e: FixedSectEnum<u64>, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<u64x8> {
        match e {
            FixedSectEnum::NullFixedSect(nfs) => FixedSectReader::<u64>::decode_to_sink(&nfs, output),
            FixedSectEnum::NibblePackMedFixedSect(fs) => fs.decode_to_sink(output),
            FixedSectEnum::DeltaNPMedFixedSect(fs)    => fs.decode_to_sink(output),
            FixedSectEnum::ConstFixedSect(cs)         => cs.decode_to_sink(output),
            // _ => Err(CodingError::InvalidFormat(format!("Section {:?} invalid for u64", e))),
        }
    }

    #[inline]
    fn read_le_offset<'a>(buf: &'a [u8], offset: usize) -> Result<u64, scroll::Error> {
        buf.pread_with(offset, LE)
    }

    #[inline]
    fn write_le_offset<'a>(buf: &'a mut [u8], offset: usize, value: u64) -> Result<usize, scroll::Error> {
        buf.pwrite_with(value, offset, LE)
    }

    #[inline]
    fn nibblepack_decode<'a, S: Sink<u64x8>>(buf: &'a [u8], sink: &mut S) -> Result<&'a [u8], CodingError> {
        nibblepacking::nibble_unpack8(buf, sink)
    }
}


/// This is a base trait to tie together many disparate types: SinkInput, the base number type of the vector,
/// the FixedSectReader and Enum types, FSUtils, etc. etc.
/// Many other structs such as VectorReader and Filter structs will take VectBase as a base type.
/// Choose the base type for your vector - u32, u64 etc.  This should be same type used in Appender as well as
/// readers, filters, etc.
pub trait VectBase: Num + Bounded + Ord + Copy + std::fmt::Debug {
    type SI: SinkInput<Item = Self> + Add<Self::SI, Output = Self::SI>;
    type Utils: FSUtils<Self>;
}

impl VectBase for u32 {
    type SI = u32x8;
    type Utils = FSUtilsMarker;
}

impl VectBase for u64 {
    type SI = u64x8;
    type Utils = FSUtilsMarker;
}

pub const NULL_SECT_U32: [u32; 256] = [0u32; 256];

/// A NullFixedSect are 256 "Null" or 0 elements.
/// For dictionary encoding they represent missing or Null values.
/// Its binary representation consists solely of a SectionType::Null byte.
#[derive(Debug, PartialEq)]
pub struct NullFixedSect {}

impl NullFixedSect {
    /// Writes out marker for null section, just one byte.  Returns offset+1 unless
    /// there isn't room or offset is invalid.
    pub fn write(out_buf: &mut [u8], offset: usize) -> Result<usize, CodingError> {
        out_buf.pwrite_with(SectionType::Null as u8, offset, LE)?;
        Ok(offset + 1)
    }
}

impl FixedSection for NullFixedSect {
    fn num_bytes(&self) -> usize { 1 }
    fn sect_bytes(&self) -> Option<&[u8]> { None }
}

impl<T: VectBase> FixedSectReader<T> for NullFixedSect {
    #[inline]
    fn decode_to_sink<Output>(&self, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<T::SI> {
        for _ in 0..FIXED_LEN/8 {
            output.process_zeroes();
        }
        Ok(())
    }
}

/// Statistics on data to be written by a FixedSectionWriter
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionWriterStats<T: VectBase> {
    min: T,
    max: T,
}

impl<T: VectBase> SectionWriterStats<T> {
    pub fn from_vect(vect: &[T]) -> Self {
        Self { min: *vect.iter().min().unwrap_or(&T::zero()),
               max: *vect.iter().max().unwrap_or(&T::zero()) }
    }

    #[inline]
    pub fn range(&self) -> T {
        self.max - self.min
    }
}

impl<T: VectBase + PrimInt> SectionWriterStats<T> {
    #[inline]
    pub fn num_bits_range(&self) -> u8 {
        (T::Utils::BYTE_WIDTH * 8) as u8 - self.range().leading_zeros() as u8
    }

    #[inline]
    pub fn num_bits_max(&self) -> u8 {
        (T::Utils::BYTE_WIDTH * 8) as u8 - self.max.leading_zeros() as u8
    }
}

/// A trait for FixedSection writers of a particular type
pub trait FixedSectionWriter<T: VectBase> {
    /// Writes out/encodes a fixed section given input values of a particular type, starting at a given offset
    /// into the destination buffer.  Stats on the values are needed.
    /// Returns the new offset after writing succeeds.
    fn write(out_buf: &mut [u8],
             offset: usize,
             values: &[T],
             stats: SectionWriterStats<T>) -> Result<usize, CodingError>;

    /// Convenience method to compute the stats automatically and call write
    #[inline]
    fn gen_stats_and_write(out_buf: &mut [u8], offset: usize, values: &[T]) -> Result<usize, CodingError> {
        let stats = SectionWriterStats::from_vect(values);
        Self::write(out_buf, offset, values, stats)
    }
}

/// A FixedSection which is: NP=NibblePack'ed, u64/u32 elements, Medium sized (<64KB)
/// Binary layout (all offsets are from start of section/type byte)
///  +0   SectionType::NibblePackedMedium
///  +1   2-byte LE size of NibblePack-encoded bytes to follow
///  +3   NibblePack-encoded 256 u64 elements
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct NibblePackMedFixedSect<'buf, T: VectBase> {
    sect_bytes: &'buf [u8],
    encoded_bytes: u16,   // This is a separate field as sect_bytes might extend beyond end of section
                          // for performance reasons.  It is faster to be able to read beyond end
    _type: PhantomData<T>,
}

impl<'buf, T: VectBase> NibblePackMedFixedSect<'buf, T> {
    /// Tries to create a new NibblePackU64MedFixedSect from a byte slice starting from the first
    /// section type byte of the section.  Byte slice should be as large as the length bytes indicate.
    pub fn try_from(sect_bytes: &'buf [u8]) -> Result<NibblePackMedFixedSect<T>, CodingError> {
        let encoded_bytes = sect_bytes.pread_with(1, LE)
                                .and_then(|n| {
                                    if (n + 3) <= sect_bytes.len() as u16 { Ok(n) }
                                    else { Err(scroll::Error::Custom("Slice not large enough".to_string())) }
                                })?;
        Ok(Self { sect_bytes, encoded_bytes, _type: PhantomData })
    }
}

impl<'buf, T: VectBase> FixedSectReader<T> for NibblePackMedFixedSect<'buf, T> {
    #[inline]
    fn decode_to_sink<Output>(&self, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<T::SI> {
        let mut values_left = FIXED_LEN;
        let mut inbuf = &self.sect_bytes[3..];
        while values_left > 0 {
            inbuf = T::Utils::nibblepack_decode(inbuf, output)?;
            values_left -= 8;
        }
        Ok(())
    }
}

impl<'buf, T: VectBase> FixedSection for NibblePackMedFixedSect<'buf, T> {
    fn num_bytes(&self) -> usize { self.encoded_bytes as usize + 3 }
    fn sect_bytes(&self) -> Option<&[u8]> { Some(self.sect_bytes) }
}

impl<'buf, T> FixedSectionWriter<T> for NibblePackMedFixedSect<'buf, T>
where T: PrimInt + Unsigned + VectBase + num::cast::AsPrimitive<u64> {
    /// Writes out a fixed NibblePacked medium section, including correct length bytes,
    /// performing NibblePacking in the meantime.  Note: length value will be written last.
    /// Only after the write succeeds should vector metadata such as length/num bytes be updated.
    /// Returns the final offset after last bytes written.
    fn write(out_buf: &mut [u8],
             offset: usize,
             values: &[T],
             _s: SectionWriterStats<T>) -> Result<usize, CodingError> {
        assert_eq!(values.len(), FIXED_LEN);
        out_buf.pwrite_with(SectionType::NibblePackedMedium as u8, offset, LE)?;
        let off = nibblepacking::pack_u64(values.iter().map(|&x| x.as_()),
                                          out_buf,
                                          offset + 3)?;
        let num_bytes = off - offset - 3;
        if num_bytes <= 65535 {
            out_buf.pwrite_with(num_bytes as u16, offset + 1, LE)?;
            Ok(off)
        } else {
            Err(CodingError::NotEnoughSpace)
        }
    }
}


/// A FixedSection which is: NP=NibblePack'ed, Medium sized (<64KB), Delta encoded
/// Binary layout (all offsets are from start of section/type byte)
///  +0   SectionType::DeltaNPMedium
///  +1   2-byte LE size of NibblePack-encoded bytes to follow after this header
///  +3   u8: number of bits needed by largest delta
///  +4   u64: base u64 value
///  +12   NibblePack-encoded 256 u64 deltas
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct DeltaNPMedFixedSect<'buf, T>
where T: VectBase {
    sect_bytes: &'buf [u8],
    encoded_bytes: u16,   // This is a separate field as sect_bytes might extend beyond end of section
                          // for performance reasons.  It is faster to be able to read beyond end
    base: T,              // base value from which deltas are added
    delta_numbits: u8,    // max number of bits needed for deltas.  Can be used to compute max
}

const DELTA_NP_SECT_HEADER_SIZE: usize = 12;

impl<'buf, T> DeltaNPMedFixedSect<'buf, T>
where T: VectBase {
    /// Tries to create a new DeltaNPMedFixedSect from a byte slice starting from the first
    /// section type byte of the section.  Byte slice should be as large as the length bytes indicate.
    pub fn try_from(sect_bytes: &'buf [u8]) -> Result<Self, CodingError> {
        let encoded_bytes = sect_bytes.pread_with(1, LE)
                                .and_then(|n| {
                                    if (n + DELTA_NP_SECT_HEADER_SIZE as u16) <= sect_bytes.len() as u16 { Ok(n) }
                                    else { Err(scroll::Error::Custom("Slice not large enough".to_string())) }
                                })?;
        let base: T = T::Utils::read_le_offset(sect_bytes, 4)?;
        let delta_numbits: u8 = sect_bytes[3];
        Ok(Self { sect_bytes, encoded_bytes, base, delta_numbits })
    }

    /// Returns the max range of deltas (rounded up to 2^n) using delta_numbits
    pub fn delta_range(&self) -> u64 {
        2u64.pow(self.delta_numbits as u32)
    }
}

impl<'buf, T> FixedSectReader<T> for DeltaNPMedFixedSect<'buf, T>
where T: PrimInt + Unsigned + VectBase {
    #[inline]
    fn decode_to_sink<Output>(&self, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<T::SI> {
        let mut values_left = FIXED_LEN;
        let mut inbuf = &self.sect_bytes[DELTA_NP_SECT_HEADER_SIZE..];
        let mut delta_sink = AddConstSink::new(self.base, output);
        while values_left > 0 {
            inbuf = T::Utils::nibblepack_decode(inbuf, &mut delta_sink)?;
            values_left -= 8;
        }
        Ok(())
    }
}

impl<'buf, T> FixedSectionWriter<T> for DeltaNPMedFixedSect<'buf, T>
where T: PrimInt + Unsigned + VectBase + num::cast::AsPrimitive<u64> {
    /// Writes out a delta-encoded NibblePacked section.
    /// Returns the final offset after last bytes written.
    fn write(out_buf: &mut [u8],
             offset: usize,
             values: &[T],
             stats: SectionWriterStats<T>) -> Result<usize, CodingError> {
        assert_eq!(values.len(), FIXED_LEN);
        out_buf.pwrite_with(SectionType::DeltaNPMedium as u8, offset, LE)?;
        let off = nibblepacking::pack_u64(values.iter().map(|&x| (x - stats.min).as_()),
                                          out_buf,
                                          offset + DELTA_NP_SECT_HEADER_SIZE)?;
        let num_bytes = off - offset - DELTA_NP_SECT_HEADER_SIZE;
        if num_bytes <= 65535 {
            out_buf.pwrite_with(num_bytes as u16, offset + 1, LE)?;
            T::Utils::write_le_offset(out_buf, offset + 4, stats.min)?;
            out_buf[offset + 3] = stats.num_bits_range();
            Ok(off)
        } else {
            Err(CodingError::NotEnoughSpace)
        }
    }
}

impl<'buf, T> FixedSection for DeltaNPMedFixedSect<'buf, T>
where T: VectBase {
    fn num_bytes(&self) -> usize { self.encoded_bytes as usize + DELTA_NP_SECT_HEADER_SIZE }
    fn sect_bytes(&self) -> Option<&[u8]> { Some(self.sect_bytes) }
}

/// A Constant section represents repeating values
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ConstFixedSect<'buf, T: VectBase> {
    sect_bytes: &'buf [u8],
    value: T
}

impl<'buf, T: VectBase> ConstFixedSect<'buf, T> {
    pub fn try_from(sect_bytes: &'buf [u8]) -> Result<Self, CodingError> {
        if sect_bytes.len() >= (1 + T::Utils::BYTE_WIDTH) {
            let value = T::Utils::read_le_offset(sect_bytes, 1)?;
            Ok(Self { sect_bytes, value })
        } else {
            Err(CodingError::InputTooShort)
        }
    }

    pub fn get_value(&self) -> T { self.value }
}

impl<'buf, T: VectBase> FixedSectReader<T> for ConstFixedSect<'buf, T> {
    #[inline]
    fn decode_to_sink<Output>(&self, output: &mut Output) -> Result<(), CodingError>
        where Output: Sink<T::SI> {
        let octet = T::SI::splat(self.value);
        for _ in 0..FIXED_LEN/8 {
            output.process(octet);
        }
        Ok(())
    }
}

impl<'buf, T: VectBase> FixedSectionWriter<T> for ConstFixedSect<'buf, T> {
    fn write(out_buf: &mut [u8],
             offset: usize,
             values: &[T],
             _stats: SectionWriterStats<T>) -> Result<usize, CodingError> {
        assert_eq!(values.len(), FIXED_LEN);
        out_buf.pwrite_with(SectionType::Constant as u8, offset, LE)?;
        T::Utils::write_le_offset(out_buf, offset + 1, values[0])?;
        Ok(offset + 1 + T::Utils::BYTE_WIDTH)
    }
}

impl<'buf, T: VectBase> FixedSection for ConstFixedSect<'buf, T> {
    fn num_bytes(&self) -> usize { 1 + T::Utils::BYTE_WIDTH }
    fn sect_bytes(&self) -> Option<&[u8]> { Some(self.sect_bytes) }
}


/// The AutoEncoder automatically picks the optimal type of section to use based on
/// the SectionWriterStats.
/// 1. If min==max, use a Constant or Null section
/// 2. If min-max range uses less nibbles than otherwise for max, then Delta is a win.
/// 3. Otherwise use standard NibblePackMedFixedSect
pub struct AutoEncoder {}

impl<'buf, T> FixedSectionWriter<T> for AutoEncoder
where T: VectBase + PrimInt + Unsigned + num::cast::AsPrimitive<u64> {
    fn write(out_buf: &mut [u8],
             offset: usize,
             values: &[T],
             stats: SectionWriterStats<T>) -> Result<usize, CodingError> {
        if stats.min == stats.max {
            if stats.min == T::zero() {
                // All 0's, write out a null section
                NullFixedSect::write(out_buf, offset)
            } else {
                // Constant section
                ConstFixedSect::write(out_buf, offset, values, stats)
            }
        } else {
            let regular_nibbles = (stats.num_bits_max() + 3) / 4;
            let range_nibbles = (stats.num_bits_range() + 3) / 4;
            // If doing delta results in less nibbles, it will probably save space
            if range_nibbles < regular_nibbles {
                DeltaNPMedFixedSect::write(out_buf, offset, values, stats)
            } else {
                NibblePackMedFixedSect::write(out_buf, offset, values, stats)
            }
        }
    }
}


/// Iterates over a series of encoded FixedSections, basically the data of any Vector encoded as Fixed256
pub struct FixedSectIterator<'buf, T: VectBase> {
    encoded_bytes: &'buf [u8],
    _typ: PhantomData<T>,
}

impl<'buf, T: VectBase> FixedSectIterator<'buf, T> {
    pub fn new(encoded_bytes: &'buf [u8]) -> Self {
        FixedSectIterator { encoded_bytes, _typ: PhantomData }
    }
}

/// FixedSectIterator iterates over Result of FixedSectEnum.  Any decoding errors, such as trying to decode
/// a u32 section with u64 or the wrong type, for example, would result in Err(CodingError).
/// Iterates until there are no more bytes left in self.encoded_bytes.
impl<'buf, T: VectBase> Iterator for FixedSectIterator<'buf, T> {
    type Item = Result<FixedSectEnum<'buf, T>, CodingError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.encoded_bytes.is_empty() {
            None
        } else {
            let res = FixedSectEnum::try_from(self.encoded_bytes);
            if let Ok(fsreader) = &res {
                self.encoded_bytes = &self.encoded_bytes[fsreader.num_bytes()..];
            }
            Some(res)
        }
    }
}

// This is partly for perf disassembly and partly for convenience
pub fn unpack_u32_section(buf: &[u8]) -> [u32; 256] {
    let mut sink = U32_256Sink::new();
    NibblePackMedFixedSect::<u32>::try_from(buf).unwrap().decode_to_sink(&mut sink).unwrap();
    sink.values
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_sectwriter_cannot_add_sect_header() {
        let mut buf = [0u8; 4];   // Too small to write a section header in!!
        let mut writer = SectionWriter::new(&mut buf, 256);

        let res = writer.add_64kb(SectionType::Null, |writebuf: &mut [u8], _| {
            if writebuf.len() < 8 { Err(CodingError::NotEnoughSpace) }
            else {
                for n in 0..8 { writebuf[n] = 0xff; }
                Ok((8, 8))
            }
        });

        assert!(res.is_err());
    }

    #[test]
    fn test_sectwriter_fill_section_normal() {
        let mut buf = [0u8; 20];
        let mut writer = SectionWriter::new(&mut buf, 256);

        let res = writer.add_64kb(SectionType::Null, |writebuf: &mut [u8], _| {
            if writebuf.len() < 8 { Err(CodingError::NotEnoughSpace) }
            else {
                for n in 0..8 { writebuf[n] = 0xff; }
                Ok((8, 8))
            }
        });

        assert_eq!(res, Ok((8, 8)));
        assert_eq!(writer.cur_pos(), 13);
    }

    #[test]
    fn test_npu64med_write_error_no_room() {
        // Allocate a buffer that's not large enough - first, no room for header
        let mut buf = [0u8; 2];  // header needs 3 bytes at least
        let data: Vec<u64> = (0..256).collect();

        let res = NibblePackMedFixedSect::gen_stats_and_write(&mut buf, 0, &data[..]);
        assert_eq!(res, Err(CodingError::NotEnoughSpace));

        // No room for all values
        let mut buf = [0u8; 100];  // Need ~312 bytes to NibblePack compress the inputs above

        let res = NibblePackMedFixedSect::gen_stats_and_write(&mut buf, 0, &data[..]);
        assert_eq!(res, Err(CodingError::NotEnoughSpace));
    }

    #[test]
    fn test_fixedsectiterator_write_and_read() {
        let mut buf = [0u8; 1024];
        let data: Vec<u64> = (0..256).collect();
        let mut off = 0;

        off = NullFixedSect::write(&mut buf, off).unwrap();
        assert_eq!(off, 1);

        off = NibblePackMedFixedSect::gen_stats_and_write(&mut buf, off, &data[..]).unwrap();

        // Now, create an iterator and collect enums.  Send only the slice of written data, no more.
        let sect_iter = FixedSectIterator::<u64>::new(&buf[0..off]);
        let sections = sect_iter.map(|x| x.unwrap()).collect::<Vec<FixedSectEnum<u64>>>();

        assert_eq!(sections.len(), 2);
        let sect = &sections[0];
        assert_eq!(sect.num_bytes(), 1);
        match sect {
            FixedSectEnum::NullFixedSect(..) => {},
            _ => panic!("Got the wrong sect: {:?}", sect),
        }

        let sect = &sections[1];
        assert!(sect.num_bytes() <= sect.sect_bytes().unwrap().len());
        if let FixedSectEnum::NibblePackMedFixedSect(inner_sect) = sect {
            let mut sink = U64_256Sink::new();
            inner_sect.decode_to_sink(&mut sink).unwrap();
            assert_eq!(sink.values[..data.len()], data[..]);
        } else {
            panic!("Wrong type obtained at sections[1]")
        }
    }

    #[test]
    fn test_fixedsect_u32_write_and_decode() {
        let mut buf = [0u8; 1024];
        let data: Vec<u32> = (0..256).collect();
        let mut off = 0;

        off = NibblePackMedFixedSect::gen_stats_and_write(&mut buf, off, &data[..]).unwrap();

        let values = unpack_u32_section(&buf[..off]);
        assert_eq!(values.iter().count(), 256);
        assert_eq!(values.iter().map(|&x| x).collect::<Vec<u32>>(), data);
    }

    #[test]
    fn test_delta_write_and_decode() {
        // u64
        let mut buf = [0u8; 1024];
        let now_inst = SystemTime::now();
        let base_millis = now_inst.duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        let data: Vec<u64> = (0..256).map(|x| x + base_millis).collect();
        let mut _off = 0;

        _off = DeltaNPMedFixedSect::gen_stats_and_write(&mut buf, _off, &data[..]).unwrap();

        let mut sink = U64_256Sink::new();
        let section = DeltaNPMedFixedSect::<u64>::try_from(&buf).unwrap();
        assert!(section.num_bytes() < 350);   // 12 + ~1 bytes per element + overhead of 25% =~ 320
        section.decode_to_sink(&mut sink).unwrap();
        assert_eq!(sink.values[..], data[..]);
        assert_eq!(section.delta_range(), 256);

        // u32
        let data: Vec<u32> = (0..256).map(|x| x + 100_000).collect();
        _off = 0;
        _off = DeltaNPMedFixedSect::<u32>::gen_stats_and_write(&mut buf, _off, &data[..]).unwrap();

        let mut sink = U32_256Sink::new();
        let section = DeltaNPMedFixedSect::<u32>::try_from(&buf).unwrap();
        assert!(section.num_bytes() < 350);   // 12 + ~1 bytes per element + overhead of 25% =~ 320
        section.decode_to_sink(&mut sink).unwrap();
        assert_eq!(sink.values[..], data[..]);
        assert_eq!(section.delta_range(), 256);
    }

    #[test]
    fn test_const_write_and_decode() {
        let mut buf = [0u8; 256];
        let data = [400u64; 256];
        let _off = ConstFixedSect::gen_stats_and_write(&mut buf, 0, &data[..]).unwrap();

        let mut sink = U64_256Sink::new();
        let section = ConstFixedSect::<u64>::try_from(&buf).unwrap();
        assert_eq!(section.num_bytes(), 9);
        section.decode_to_sink(&mut sink).unwrap();
        assert_eq!(sink.values[..], data[..]);
    }

    #[test]
    fn test_autoencoder() {
        let mut buf = [0u8; 1024];

        // Test 1: Constant, non-null
        let data = [23_000u64; 256];
        let _off = AutoEncoder::gen_stats_and_write(&mut buf, 0, &data[..]).unwrap();
        let sect = FixedSectEnum::<u64>::try_from(&buf[..]).unwrap();
        match sect {
            FixedSectEnum::ConstFixedSect(..) => {},
            _ => panic!("Got the wrong sect: {:?}", sect),
        }

        // Test 2: all 0's
        let data = [0u64; 256];
        let _off = AutoEncoder::gen_stats_and_write(&mut buf, 0, &data[..]).unwrap();
        let sect = FixedSectEnum::<u64>::try_from(&buf[..]).unwrap();
        match sect {
            FixedSectEnum::NullFixedSect(..) => {},
            _ => panic!("Got the wrong sect: {:?}", sect),
        }

        // Test 3: Normal items range between 1 and n
        let data: Vec<u32> = (0..256).collect();
        let _off = AutoEncoder::gen_stats_and_write(&mut buf, 0, &data[..]).unwrap();
        let sect = FixedSectEnum::<u32>::try_from(&buf[..]).unwrap();
        match sect {
            FixedSectEnum::NibblePackMedFixedSect(..) => {},
            _ => panic!("Got the wrong sect: {:?}", sect),
        }

        // Test 4: Elevated, should be delta (max-min << max)
        let data: Vec<u32> = (10_000..10_256).collect();
        let _off = AutoEncoder::gen_stats_and_write(&mut buf, 0, &data[..]).unwrap();
        let sect = FixedSectEnum::<u32>::try_from(&buf[..]).unwrap();
        match sect {
            FixedSectEnum::DeltaNPMedFixedSect(..) => {},
            _ => panic!("Got the wrong sect: {:?}", sect),
        }
    }
}

