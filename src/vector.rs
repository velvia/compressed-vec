use std::marker::PhantomData;
use std::mem;

use num::{Zero, Unsigned};
use scroll::{ctx, Endian, Pread, Pwrite, LE};

use crate::error::CodingError;
use crate::section::*;

/// BinaryVector: a compressed vector storing data of the same type
///   enabling high speed operations on compressed data without
///   the need for decompressing (in many cases, exceptions noted)
///
/// A BinaryVector MAY consist of multiple sections.  Each section can represent
/// potentially different encoding parameters (bit widths, sparsity, etc.) and
/// has its own header to allow for quickly skipping ahead even when different
/// sections are encoded differently.
///
/// This struct describes a common header for all BinaryVectors.  Note that the
/// first 16 bytes of a BinaryVector are reserved for the header, not just what is
/// defined here.
/// The major and minor types and the header bytes are compatible with FiloDB BinaryVectors.
#[repr(C)]
#[derive(Debug, PartialEq, Pwrite)]
pub struct BinaryVector {
    num_bytes: u32,         // Number of bytes in vector following this length
    major_type: VectorType, // These should probably be enums no?
    minor_type: VectorSubType,
    num_elements: u16,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VectorType {
    Empty = 0x01,
    BinSimple = 0x06,
    BinDict = 0x07,
    Delta2 = 0x08,      // Delta-delta encoded
    Histogram = 0x09,   // FiloDB sections with Histogram chunks per section
    FixedSection256 = 0x10,    // Fixed 256-element sections
}

impl VectorType {
    pub fn as_num(&self) -> u8 { *self as u8 }
}

impl ctx::TryIntoCtx<Endian> for &VectorType {
    type Error = scroll::Error;
    fn try_into_ctx(self, buf: &mut [u8], ctx: Endian) -> Result<usize, Self::Error> {
        u8::try_into_ctx(self.as_num(), buf, ctx)
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VectorSubType {
    Primitive = 0x00,
    STRING = 0x01,
    UTF8 = 0x02,
    FIXEDMAXUTF8 = 0x03, // fixed max size per blob, length byte
    DATETIME = 0x04,
    PrimitiveNoMask = 0x05,
    REPEATED = 0x06, // vectors.ConstVector
    INT = 0x07,      // Int gets special type because Longs and Doubles may be encoded as Int
    IntNoMask = 0x08,
}

impl VectorSubType {
    pub fn as_num(&self) -> u8 { *self as u8 }
}

impl ctx::TryIntoCtx<Endian> for &VectorSubType {
    type Error = scroll::Error;
    fn try_into_ctx(self, buf: &mut [u8], ctx: Endian) -> Result<usize, Self::Error> {
        u8::try_into_ctx(self.as_num(), buf, ctx)
    }
}

const NUM_HEADER_BYTES_TOTAL: usize = 16;

impl BinaryVector {
    pub fn new(major_type: VectorType, minor_type: VectorSubType) -> Self {
        Self { num_bytes: NUM_HEADER_BYTES_TOTAL as u32 - 4, major_type, minor_type, num_elements: 0 }
    }

    /// Returns the length of the BinaryVector including the length bytes
    pub fn whole_length(&self) -> u32 {
        self.num_bytes + (mem::size_of::<u32>() as u32)
    }

    pub fn reset(&mut self) {
        self.num_bytes = NUM_HEADER_BYTES_TOTAL as u32 - 4;
        self.num_elements = 0;
    }

    /// Writes the entire BinaryVector header into the beginning of the given buffer
    pub fn write_header(&self, buf: &mut [u8]) -> Result<(), CodingError> {
        buf.pwrite_with(self, 0, LE)?;
        Ok(())
    }

    pub fn update_num_elems(&mut self, buf: &mut [u8], num_elements: u16) -> Result<(), CodingError> {
        self.num_elements = num_elements;
        buf.pwrite_with(self.num_elements, 6, LE)?;
        Ok(())
    }

    /// Updates both the number of bytes in the vector and the number of elements at once.
    /// The num_body_bytes should be the number of bytes AFTER the 16-byte BinaryVector header.
    /// The buffer slice should point to the beginning of the header ie the length bytes
    pub fn update_length(&mut self,
                         buf: &mut [u8],
                         num_body_bytes: u32,
                         num_elements: u16) -> Result<(), CodingError> {
        self.num_bytes = num_body_bytes + (NUM_HEADER_BYTES_TOTAL - 4) as u32;
        buf.pwrite_with(self.num_bytes, 0, LE)?;
        self.update_num_elems(buf, num_elements)
    }
}

const GROW_BYTES: usize = 4096;

/// A builder for a BinaryVector holding encoded/compressed u64/u32 values
/// as 256-element FixedSections.   Buffers elements to be written and writes
/// them in 256-element sections at a time.  This builder owns its own write buffer memory, expanding it
/// as needed.  `finish()` wraps up the vector, cloning a copy, and `reset()` can be called to reuse
/// this appender.  The write buffer stays with this builder to minimize allocations.
/// NOTE: the vector state (elements, num bytes etc) are only updated when a section is updated.
/// So readers who read just the vector itself will not get the updates in write_buf.
/// This appender must be consulted for querying write_buf values.
pub struct FixedSectIntAppender<T, W>
where T: Zero + Unsigned + Clone,
      W: FixedSectionWriter<T> {
    vect_buf: Vec<u8>,
    offset: usize,
    header: BinaryVector,
    write_buf: Vec<T>,
    sect_writer: PhantomData<W>     // Uses no space, this tells rustc we need W
}

impl<T, W> FixedSectIntAppender<T, W>
where T: Zero + Unsigned + Clone,
      W: FixedSectionWriter<T> {
    /// Creates a new FixedSectIntAppender.  Usually you'll want to use one of the more concrete typed structs.
    /// Also initializes the vect_buf with a valid section header.  Initial capacity is the initial size of the
    /// write buffer, which can grow.
    pub fn try_new(major_type: VectorType,
                   minor_type: VectorSubType,
                   initial_capacity: usize) -> Result<Self, CodingError> {
        let mut new_self = Self {
            vect_buf: vec![0; initial_capacity],
            offset: NUM_HEADER_BYTES_TOTAL,
            header: BinaryVector::new(major_type, minor_type),
            write_buf: Vec::with_capacity(FIXED_LEN),
            sect_writer: PhantomData
        };
        new_self.write_header()?;
        Ok(new_self)
    }

    /// Total number of elements including encoded sections and write buffer
    pub fn num_elements(&self) -> usize {
        self.header.num_elements as usize + self.write_buf.len()
    }

    /// Resets the internal state for appending a new vector.
    pub fn reset(&mut self) -> Result<(), CodingError> {
        self.header.reset();
        self.offset = NUM_HEADER_BYTES_TOTAL;
        self.write_buf.clear();
        self.vect_buf.clear();
        self.write_header()
    }

    /// Writes out the header for the vector.  Done automatically during try_new() / reset().
    fn write_header(&mut self) -> Result<(), CodingError> {
        self.header.write_header(self.vect_buf.as_mut_slice())
    }

    /// Encodes all the values in write_buf.  Adjust the number of elements and other vector state.
    fn encode_section(&mut self) -> Result<(), CodingError> {
        assert!(self.write_buf.len() == FIXED_LEN);
        self.offset = self.retry_grow(|s| W::write(s.vect_buf.as_mut_slice(),
                                                   s.offset,
                                                   &s.write_buf[..]))?;
        self.write_buf.clear();
        self.header.update_length(self.vect_buf.as_mut_slice(),
                                  (self.offset - NUM_HEADER_BYTES_TOTAL) as u32,
                                  self.header.num_elements + FIXED_LEN as u16)
    }

    /// Retries a func which might return Result<..., CodingError> by growing the vect_buf.
    /// If it still fails then we return the Err.
    fn retry_grow<F, U>(&mut self, mut func: F) -> Result<U, CodingError>
        where F: FnMut(&mut Self) -> Result<U, CodingError> {
        func(self).or_else(|err| {
            match err {
                CodingError::NotEnoughSpace => {
                    // Expand vect_buf
                    self.vect_buf.reserve(GROW_BYTES);
                    self.vect_buf.resize(self.vect_buf.capacity(), 0);
                    func(self)
                }
                _ => Err(err),
            }
        })
    }

    /// Appends a single value to this vector.  When a section fills up, will encode all values in write buffer
    /// into the vector.
    pub fn append(&mut self, value: T) -> Result<(), CodingError> {
        self.write_buf.push(value);
        if self.write_buf.len() >= FIXED_LEN {
            self.encode_section()
        } else {
            Ok(())
        }
    }

    /// Appends a number of nulls at once to the vector.  Super useful and fast for sparse data.
    /// Nulls are equivalent to zero value for type T.
    pub fn append_nulls(&mut self, num_nulls: u16) -> Result<(), CodingError> {
        let mut left = num_nulls;
        while left > 0 {
            // If current write_buf is not empty, fill it up with zeroes and flush (maybe)
            if self.write_buf.len() > 0 {
                let num_to_fill = left.min((FIXED_LEN - self.write_buf.len()) as u16);
                self.write_buf.resize(self.write_buf.len() + num_to_fill as usize, T::zero());
                left -= num_to_fill;
                if self.write_buf.len() >= FIXED_LEN { self.encode_section()?; }
            // If empty, and we have at least FIXED_LEN nulls to go, insert a null section.
            } else if left >= (FIXED_LEN as u16) {
                self.offset = self.retry_grow(|s| NullFixedSect::write(s.vect_buf.as_mut_slice(), s.offset))?;
                self.header.update_length(self.vect_buf.as_mut_slice(),
                                          (self.offset - NUM_HEADER_BYTES_TOTAL) as u32,
                                          self.header.num_elements + FIXED_LEN as u16)?;
                left -= FIXED_LEN as u16;
            // If empty, and less than fixed_len nulls, insert nulls into write_buf
            } else {
                self.write_buf.resize(left as usize, T::zero());
                left = 0;
            }
        }
        Ok(())
    }

    /// Call this method to wrap up a vector and any unfinished sections, and clone out resulting vector.
    /// We have no more values, and need to fill up the appender with nulls/0's until it is the right length.
    /// This is because most query engines expect all vectors to be of the same number of elements.
    /// The number passed in will be stored as the actual number of elements for iteration purposes, however
    /// since this is a fixed size section vector, the number will be rounded up to the next FIXED_LEN so that
    /// an entire section is written.
    /// NOTE: TooFewRows is returned if total_num_rows is below the total number of elements written so far.
    pub fn finish(&mut self, total_num_rows: usize) -> Result<Vec<u8>, CodingError> {
        let total_so_far = self.header.num_elements as usize + self.write_buf.len();
        if total_so_far > total_num_rows { return Err(CodingError::InvalidNumRows(total_num_rows, total_so_far)); }
        if total_num_rows > u16::max_value() as usize {
            return Err(CodingError::InvalidNumRows(total_num_rows, u16::max_value() as usize));
        }

        // Round out the section if needed
        if self.write_buf.len() > 0 {
            let number_to_fill = FIXED_LEN - self.write_buf.len();
            self.append_nulls(number_to_fill as u16)?;
        }

        while self.header.num_elements < total_num_rows as u16 {
            self.append_nulls(256)?;
        }

        // Re-write the number of elements to reflect total_num_rows
        self.header.update_num_elems(self.vect_buf.as_mut_slice(), total_num_rows as u16)?;

        self.vect_buf.resize(self.offset, 0);
        let mut returned_vec = Vec::with_capacity(self.offset);
        returned_vec.append(&mut self.vect_buf);
        Ok(returned_vec)
    }

    /// Obtains a reader for reading from the bytes of this appender.
    /// NOTE: reader will only read what has been written so far, and due to Rust borrowing rules, one should
    /// not attempt to read and append at the same time; the returned reader is not safe across threads.
    pub fn reader(&self) -> FixedSectIntReader {
        // This should never fail, as we have already proven we can initialize the vector
        FixedSectIntReader::try_new(&self.vect_buf[..self.offset]).expect("Getting reader from appender failed")
    }
}

/// Regular U64 appender with just plain NibblePacked encoding
type FixedSectU64Appender = FixedSectIntAppender<u64, NibblePackU64MedFixedSect>;

impl FixedSectU64Appender {
    pub fn new(initial_capacity: usize) -> Result<FixedSectU64Appender, CodingError> {
        FixedSectU64Appender::try_new(VectorType::FixedSection256, VectorSubType::Primitive,
                                      initial_capacity)
    }
}

/// A reader for reading sections and elements from a `FixedSectIntAppender` written vector.
// TODO: have a reader trait of some kind?
pub struct FixedSectIntReader<'a> {
    vect_bytes: &'a [u8],
}

impl<'a> FixedSectIntReader<'a> {
    /// Creates a new reader out of the bytes for the vector.
    // TODO: verify that the vector is a fixed sect int.
    pub fn try_new(vect_bytes: &'a [u8]) -> Result<Self, CodingError> {
        let bytes_from_header: u32 = vect_bytes.pread_with(0, LE)?;
        if vect_bytes.len() < (bytes_from_header + 4) as usize {
            Err(CodingError::InputTooShort)
        } else {
            Ok(Self { vect_bytes })
        }
    }

    pub fn num_elements(&self) -> usize {
        // Should not fail since we have verified in try_new() that we have all header bytes
        self.vect_bytes.pread_with::<u16>(6, LE).unwrap() as usize
    }

    /// Returns an iterator over sections and each section's bytes
    pub fn sect_iter(&self) -> FixedSectIterator<'a> {
        FixedSectIterator::new(&self.vect_bytes[NUM_HEADER_BYTES_TOTAL..])
    }

    /// Returns the number of null sections
    pub fn num_null_sections(&self) -> usize {
        self.sect_iter().filter(|(sect, _)| sect.is_null()).count()
    }
}

/// Returns iterator over all items in a vector. NOTE: not designed to be performant.
/// Right now it boxes each section iterator since they are dynamic.
/// TODO: if we really need a performant version of this, think of some other solutions:
///  1. Create a struct which can switch on inner iterators
///  2. Create iterator of [u64; 8]; then flatmap it.  The iterator can be converted to u64.
///  3. Maybe use a regional or arena allocator.
pub fn fixed_iter_u64<'a>(reader: &FixedSectIntReader<'a>) -> impl Iterator<Item = u64> + 'a {
    reader.sect_iter().flat_map(|(sect, s_bytes)| {
        let iter: Box<dyn Iterator<Item = u64>> = match sect {
            FixedSectEnum::NibblePackU64MedFixedSect(mut inner_sect) =>
                Box::new(inner_sect.iter(s_bytes)),
            FixedSectEnum::NullFixedSect(_) =>
                Box::new((0..FIXED_LEN).map(|_s| 0u64)),
            _ => panic!("No other section types supported"),
        };
        iter
    }).take(reader.num_elements())
}

#[test]
fn test_append_u64_nonulls() {
    // Append more than 256 values, see if we get two sections and the right data back
    let num_values: usize = 500;
    let data: Vec<u64> = (0..num_values as u64).collect();

    let mut appender = FixedSectU64Appender::new(1024).unwrap();
    {
        let reader = appender.reader();

        assert_eq!(reader.num_elements(), 0);
        assert_eq!(reader.sect_iter().count(), 0);
    // Note: due to Rust borrowing rules we can only have reader as long as we are not appending.
    }

    // Now append the data
    data.iter().for_each(|&e| appender.append(e).unwrap());

    // At this point only 1 section has been written, the vector is not finished yet.
    let reader = appender.reader();
    assert_eq!(reader.num_elements(), 256);
    assert_eq!(reader.sect_iter().count(), 1);

    let finished_vec = appender.finish(num_values).unwrap();

    let reader = FixedSectIntReader::try_new(&finished_vec[..]).unwrap();
    assert_eq!(reader.num_elements(), num_values);
    assert_eq!(reader.sect_iter().count(), 2);
    assert_eq!(reader.num_null_sections(), 0);

    let elems: Vec<u64> = fixed_iter_u64(&reader).collect();
    assert_eq!(elems, data);
}

#[test]
fn test_append_u64_mixed_nulls() {
    // Have some values, then append a large number of nulls
    // (enough to pack rest of section, plus a null section, plus more in next section)
    // Thus sections should be: Sect1: 100 values + 156 nulls
    //    Sect2: null section
    //    Sect3: 50 nulls + 50 more values
    let data1: Vec<u64> = (0..100).collect();
    let num_nulls = (256 - data1.len()) + 256 + 50;
    let data2: Vec<u64> = (0..50).collect();

    let total_elems = data1.len() + data2.len() + num_nulls;

    let mut all_data = Vec::<u64>::with_capacity(total_elems);
    all_data.extend_from_slice(&data1[..]);
    (0..num_nulls).for_each(|_i| all_data.push(0));
    all_data.extend_from_slice(&data2[..]);

    let mut appender = FixedSectU64Appender::new(1024).unwrap();
    data1.iter().for_each(|&e| appender.append(e).unwrap());
    appender.append_nulls(num_nulls as u16).unwrap();
    data2.iter().for_each(|&e| appender.append(e).unwrap());

    let finished_vec = appender.finish(total_elems).unwrap();

    let reader = FixedSectIntReader::try_new(&finished_vec[..]).unwrap();
    assert_eq!(reader.num_elements(), total_elems);
    assert_eq!(reader.sect_iter().count(), 3);
    assert_eq!(reader.num_null_sections(), 1);

    let elems: Vec<u64> = fixed_iter_u64(&reader).collect();
    assert_eq!(elems, all_data);
}

#[test]
fn test_append_u64_mixed_nulls_grow() {
    // Same as last test but use smaller buffer to force growing of encoding buffer
    let data1: Vec<u64> = (0..300).collect();
    let num_nulls = 350;

    let total_elems = (data1.len() + num_nulls) * 2;

    let mut all_data = Vec::<u64>::with_capacity(total_elems);
    all_data.extend_from_slice(&data1[..]);
    (0..num_nulls).for_each(|_i| all_data.push(0));
    all_data.extend_from_slice(&data1[..]);
    (0..num_nulls).for_each(|_i| all_data.push(0));

    let mut appender = FixedSectU64Appender::new(300).unwrap();
    data1.iter().for_each(|&e| appender.append(e).unwrap());
    appender.append_nulls(num_nulls as u16).unwrap();
    data1.iter().for_each(|&e| appender.append(e).unwrap());
    appender.append_nulls(num_nulls as u16).unwrap();

    let finished_vec = appender.finish(total_elems).unwrap();

    let reader = FixedSectIntReader::try_new(&finished_vec[..]).unwrap();
    assert_eq!(reader.num_elements(), total_elems);
    assert_eq!(reader.sect_iter().count(), 6);
    assert_eq!(reader.num_null_sections(), 1);

    let elems: Vec<u64> = fixed_iter_u64(&reader).collect();
    assert_eq!(elems, all_data);
}
