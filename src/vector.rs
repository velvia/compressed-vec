/// vector module contains `BinaryVector`, which allows creation of compressed binary vectors which can be
/// appended to and read, queried, filtered, etc. quickly.
///
/// ## Appending values and reading them back
///
/// Appending values is easy.  Appenders dynamically size the input buffer.
/// ```
/// # use compressed_vec::vector::*;
///     let mut appender = VectorU32Appender::try_new(1024).unwrap();
///     appender.append(1).unwrap();
///     appender.append(2).unwrap();
///     appender.append_nulls(3).unwrap();
///     assert_eq!(appender.num_elements(), 5);
///
///     let reader = appender.reader();
///     println!("Elements so far: {:?}", reader.iterate().count());
///
///     // Continue appending!
///     appender.append(10).unwrap();
/// ```
///
/// ## Finishing vectors
///
/// Calling `finish()` clones the vector bytes to the smallest representation possible, after which the
/// appender is reset for creation of another new vector.  The finished vector is then immutable and the
/// caller can read it.
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;

use scroll::{ctx, Endian, Pread, Pwrite, LE};

use crate::error::CodingError;
use crate::filter::{SectFilterSink, VectorFilter};
use crate::section::*;
use crate::sink::*;

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
    _padding: u16,
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
    FixedU64  = 0x10,  // FixedSection256 with u64 elements
    FixedU32  = 0x11,  // FixedSection256 with u32 elements
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
const BINARYVECT_HEADER_SIZE: usize = std::mem::size_of::<BinaryVector>();

impl BinaryVector {
    pub fn new(major_type: VectorType, minor_type: VectorSubType) -> Self {
        Self { num_bytes: NUM_HEADER_BYTES_TOTAL as u32 - 4, major_type, minor_type, _padding: 0 }
    }

    /// Returns the length of the BinaryVector including the length bytes
    pub fn whole_length(&self) -> u32 {
        self.num_bytes + (mem::size_of::<u32>() as u32)
    }

    pub fn reset(&mut self) {
        self.num_bytes = NUM_HEADER_BYTES_TOTAL as u32 - 4;
    }

    /// Writes the entire BinaryVector header into the beginning of the given buffer
    pub fn write_header(&self, buf: &mut [u8]) -> Result<(), CodingError> {
        buf.pwrite_with(self, 0, LE)?;
        Ok(())
    }

    /// Updates the number of bytes in the vector.
    /// The num_body_bytes should be the number of bytes AFTER the 16-byte BinaryVector header.
    /// The buffer slice should point to the beginning of the header ie the length bytes
    pub fn update_num_bytes(&mut self,
                            buf: &mut [u8],
                            num_body_bytes: u32) -> Result<(), CodingError> {
        self.num_bytes = num_body_bytes + (NUM_HEADER_BYTES_TOTAL - 4) as u32;
        buf.pwrite_with(self.num_bytes, 0, LE)?;
        Ok(())
    }
}


/// Mapping of VectBase type to VectorSubType.  Allows checking of vector type by reader.
pub trait BaseSubtypeMapping {
    fn vect_subtype() -> VectorSubType;
}

impl BaseSubtypeMapping for u64 {
    fn vect_subtype() -> VectorSubType { VectorSubType::FixedU64 }
}

impl BaseSubtypeMapping for u32 {
    fn vect_subtype() -> VectorSubType { VectorSubType::FixedU32 }
}

#[derive(Debug, Copy, Clone, Pread, Pwrite)]
pub struct FixedSectStats {
    pub num_elements: u32,
    num_null_sections: u16,
}

impl FixedSectStats {
    pub fn new() -> Self {
        Self { num_elements: 0, num_null_sections: 0 }
    }

    pub fn reset(&mut self) {
        self.num_elements = 0;
        self.num_null_sections = 0;
    }

    /// Updates the number of elements only.  Writes entire stats at once.
    /// Assumes buf points to beginning of _vector_ not this struct.
    pub fn update_num_elems(&mut self, buf: &mut [u8], num_elements: u32) -> Result<(), CodingError> {
        self.num_elements = num_elements;
        buf.pwrite_with(*self, BINARYVECT_HEADER_SIZE, LE)?;
        Ok(())
    }
}

const GROW_BYTES: usize = 4096;

/// A builder for a BinaryVector holding encoded/compressed integral/floating values
/// as 256-element FixedSections.   Buffers elements to be written and writes
/// them in 256-element sections at a time.  This builder owns its own write buffer memory, expanding it
/// as needed.  `finish()` wraps up the vector, cloning a copy, and `reset()` can be called to reuse
/// this appender.  The write buffer stays with this builder to minimize allocations.
/// NOTE: the vector state (elements, num bytes etc) are only updated when a section is updated.
/// So readers who read just the vector itself will not get the updates in write_buf.
/// This appender must be consulted for querying write_buf values.
pub struct VectorAppender<T, W>
where T: VectBase + Clone,
      W: FixedSectionWriter<T> {
    vect_buf: Vec<u8>,
    offset: usize,
    header: BinaryVector,
    write_buf: Vec<T>,
    stats: FixedSectStats,
    sect_writer: PhantomData<W>     // Uses no space, this tells rustc we need W
}

impl<T, W> VectorAppender<T, W>
where T: VectBase + Clone + BaseSubtypeMapping,
      W: FixedSectionWriter<T> {
    /// Creates a new VectorAppender.  Initializes the vect_buf with a valid section header.
    /// Initial capacity is the initial size of the write buffer, which can grow.
    pub fn try_new(initial_capacity: usize) -> Result<Self, CodingError> {
        let mut new_self = Self {
            vect_buf: vec![0; initial_capacity],
            offset: NUM_HEADER_BYTES_TOTAL,
            header: BinaryVector::new(VectorType::FixedSection256, T::vect_subtype()),
            write_buf: Vec::with_capacity(FIXED_LEN),
            stats: FixedSectStats::new(),
            sect_writer: PhantomData
        };
        new_self.write_header()?;
        Ok(new_self)
    }

    /// Total number of elements including encoded sections and write buffer
    pub fn num_elements(&self) -> usize {
        self.stats.num_elements as usize + self.write_buf.len()
    }

    /// Resets the internal state for appending a new vector.
    pub fn reset(&mut self) -> Result<(), CodingError> {
        self.header.reset();
        self.offset = NUM_HEADER_BYTES_TOTAL;
        self.write_buf.clear();
        self.vect_buf.resize(self.vect_buf.capacity(), 0);  // Make sure entire vec is usable
        self.stats.reset();
        self.stats.update_num_elems(&mut self.vect_buf, 0)?;
        self.write_header()
    }

    /// Writes out the header for the vector.  Done automatically during try_new() / reset().
    fn write_header(&mut self) -> Result<(), CodingError> {
        self.header.write_header(self.vect_buf.as_mut_slice())
    }

    /// Encodes all the values in write_buf.  Adjust the number of elements and other vector state.
    fn encode_section(&mut self) -> Result<(), CodingError> {
        assert!(self.write_buf.len() == FIXED_LEN);
        self.offset = self.retry_grow(|s| W::gen_stats_and_write(s.vect_buf.as_mut_slice(),
                                                                 s.offset,
                                                                 &s.write_buf[..]))?;
        self.write_buf.clear();
        self.stats.update_num_elems(&mut self.vect_buf, self.stats.num_elements + FIXED_LEN as u32)?;
        self.header.update_num_bytes(self.vect_buf.as_mut_slice(),
                                     (self.offset - NUM_HEADER_BYTES_TOTAL) as u32)
    }

    /// Retries a func which might return Result<..., CodingError> by growing the vect_buf.
    /// If it still fails then we return the Err.
    fn retry_grow<F, U>(&mut self, mut func: F) -> Result<U, CodingError>
        where F: FnMut(&mut Self) -> Result<U, CodingError> {
        func(self).or_else(|err| {
            match err {
                CodingError::NotEnoughSpace | CodingError::BadOffset(_) => {
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
    pub fn append_nulls(&mut self, num_nulls: usize) -> Result<(), CodingError> {
        let mut left = num_nulls;
        while left > 0 {
            // If current write_buf is not empty, fill it up with zeroes and flush (maybe)
            if self.write_buf.len() > 0 {
                let num_to_fill = left.min(FIXED_LEN - self.write_buf.len());
                self.write_buf.resize(self.write_buf.len() + num_to_fill as usize, T::zero());
                left -= num_to_fill;
                if self.write_buf.len() >= FIXED_LEN { self.encode_section()?; }
            // If empty, and we have at least FIXED_LEN nulls to go, insert a null section.
            } else if left >= FIXED_LEN {
                self.offset = self.retry_grow(|s| NullFixedSect::write(s.vect_buf.as_mut_slice(), s.offset))?;
                self.stats.num_null_sections += 1;
                self.stats.update_num_elems(&mut self.vect_buf, self.stats.num_elements + FIXED_LEN as u32)?;
                self.header.update_num_bytes(self.vect_buf.as_mut_slice(),
                                             (self.offset - NUM_HEADER_BYTES_TOTAL) as u32)?;
                left -= FIXED_LEN;
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
        let total_so_far = self.stats.num_elements as usize + self.write_buf.len();
        if total_so_far > total_num_rows { return Err(CodingError::InvalidNumRows(total_num_rows, total_so_far)); }
        if total_num_rows > u32::max_value() as usize {
            return Err(CodingError::InvalidNumRows(total_num_rows, u32::max_value() as usize));
        }

        // Round out the section if needed
        if self.write_buf.len() > 0 {
            let number_to_fill = FIXED_LEN - self.write_buf.len();
            self.append_nulls(number_to_fill)?;
        }

        while self.stats.num_elements < total_num_rows as u32 {
            self.append_nulls(256)?;
        }

        // Re-write the number of elements to reflect total_num_rows
        self.stats.update_num_elems(self.vect_buf.as_mut_slice(), total_num_rows as u32)?;
        self.vect_buf.as_mut_slice().pwrite_with(&self.stats, BINARYVECT_HEADER_SIZE, LE)?;

        self.vect_buf.resize(self.offset, 0);
        let mut returned_vec = Vec::with_capacity(self.offset);
        returned_vec.append(&mut self.vect_buf);
        self.reset()?;
        Ok(returned_vec)
    }

    /// Obtains a reader for reading from the bytes of this appender.
    /// NOTE: reader will only read what has been written so far, and due to Rust borrowing rules, one should
    /// not attempt to read and append at the same time; the returned reader is not safe across threads.
    pub fn reader(&self) -> VectorReader<T> {
        // This should never fail, as we have already proven we can initialize the vector
        VectorReader::try_new(&self.vect_buf[..self.offset]).expect("Getting reader from appender failed")
    }
}

/// Regular U64 appender with AutoEncoder
pub type VectorU64Appender = VectorAppender<u64, AutoEncoder>;

/// Regular U32 appender with AutoEncoder
pub type VectorU32Appender = VectorAppender<u32, AutoEncoder>;


/// A reader for reading sections and elements from a `VectorAppender` written vector.
/// Use the same base type - eg VectorU32Appender -> VectorReader::<u32>
/// Can be reused many times; it has no mutable state and creates new iterators every time.
// TODO: have a reader trait of some kind?
pub struct VectorReader<'buf, T: VectBase> {
    vect_bytes: &'buf [u8],
    _reader: PhantomData<T>,
}

impl<'buf, T> VectorReader<'buf, T>
where T: VectBase + BaseSubtypeMapping {
    /// Creates a new reader out of the bytes for the vector.
    // TODO: verify that the vector is a fixed sect int.
    pub fn try_new(vect_bytes: &'buf [u8]) -> Result<Self, CodingError> {
        let bytes_from_header: u32 = vect_bytes.pread_with(0, LE)?;
        let subtype: u8 = vect_bytes.pread_with(offset_of!(BinaryVector, minor_type), LE)?;
        if vect_bytes.len() < (bytes_from_header + 4) as usize {
            Err(CodingError::InputTooShort)
        } else if subtype != T::vect_subtype() as u8 {
            Err(CodingError::WrongVectorType(subtype))
        } else {
            Ok(Self { vect_bytes, _reader: PhantomData })
        }
    }

    pub fn num_elements(&self) -> usize {
        // Should not fail since we have verified in try_new() that we have all header bytes
        self.get_stats().num_elements as usize
    }

    pub fn total_bytes(&self) -> usize {
        self.vect_bytes.len()
    }

    /// Iterates and discovers the number of null sections.  O(num_sections).  It will be faster to just use
    /// get_stats().
    pub fn num_null_sections(&self) -> Result<usize, CodingError> {
        let mut count = 0;
        for sect_res in self.sect_iter() {
            let sect = sect_res?;
            if sect.is_null() { count += 1 }
        }
        Ok(count)
    }

    /// Returns a FixedSectStats extracted from the vector header.
    pub fn get_stats(&self) -> FixedSectStats {
        self.vect_bytes.pread_with(BINARYVECT_HEADER_SIZE, LE).unwrap()
    }

    /// Returns an iterator over each section in this vector
    pub fn sect_iter(&self) -> FixedSectIterator<'buf, T> {
        FixedSectIterator::new(&self.vect_bytes[NUM_HEADER_BYTES_TOTAL..])
    }

    /// Returns a VectorFilter that iterates over 256-bit masks filtered from vector elements
    pub fn filter_iter<F: SectFilterSink<T>>(&self, f: F) -> VectorFilter<'buf, F, T> {
        VectorFilter::new(&self.vect_bytes[NUM_HEADER_BYTES_TOTAL..], f)
    }

    /// Returns an iterator over all items in this vector.
    pub fn iterate(&self) -> VectorItemIter<'buf, T> {
        VectorItemIter::new(self.sect_iter(), self.num_elements())
    }

    /// Decodes/processes this vector's elements through a Sink.  This is the most general purpose vector
    /// decoding/processing API.
    pub fn decode_to_sink<Output>(&self, output: &mut Output) -> Result<(), CodingError>
    where Output: Sink<T::SI> {
        for sect in self.sect_iter() {
            sect?.decode(output)?;
        }
        Ok(())
    }
}


/// Detailed stats, for debugging or perf analysis, on a Vector.  Includes the section types.
#[derive(Debug)]
pub struct VectorStats {
    num_bytes: usize,
    bytes_per_elem: f32,
    stats: FixedSectStats,
    sect_types: Vec<SectionType>,
}

impl VectorStats {
    pub fn new<'buf, T: VectBase + BaseSubtypeMapping>(reader: &VectorReader<'buf, T>) -> Self {
        let stats = reader.get_stats();
        Self {
            num_bytes: reader.total_bytes(),
            bytes_per_elem: reader.total_bytes() as f32 / stats.num_elements as f32,
            stats,
            sect_types: reader.sect_iter().map(|sect| sect.unwrap().sect_type()).collect(),
        }
    }

    /// Creates a histogram or count of each section type
    pub fn sect_types_histogram(&self) -> HashMap<SectionType, usize> {
        let mut map = HashMap::new();
        self.sect_types.iter().for_each(|&sect_type| {
            let count = map.entry(sect_type).or_insert(0);
            *count += 1;
        });
        map
    }

    /// Returns a short summary string of the stats, including a histogram summary
    pub fn summary_string(&self) -> String {
        let keyvalues: Vec<_> = self.sect_types_histogram().iter()
                                    .map(|(k, v)| format!("{:?}={:?}", k, v)).collect();
        format!("#bytes={:?}   #elems={:?}   bytes-per-elem={:?}\nsection type hist: {}",
                self.num_bytes, self.stats.num_elements, self.bytes_per_elem,
                keyvalues.join(", "))
    }
}


/// Iterator struct over all items in a vector, for convenience
/// Panics on decoding error - there's no really good way for an iterator to return an error
// NOTE: part of reason to do this is to better control lifetimes which is hard otherwise
pub struct VectorItemIter<'buf, T: VectBase> {
    sect_iter: FixedSectIterator<'buf, T>,
    sink: Section256Sink<T>,
    num_elems: usize,
    i: usize,
}

impl<'buf, T: VectBase> VectorItemIter<'buf, T> {
    pub fn new(sect_iter: FixedSectIterator<'buf, T>, num_elems: usize) -> Self {
        let mut s = Self {
            sect_iter,
            sink: Section256Sink::<T>::new(),
            num_elems,
            i: 0,
        };
        if num_elems > 0 {
            s.next_section();
        }
        s
    }

    fn next_section(&mut self) {
        self.sink.reset();
        if let Some(Ok(next_sect)) = self.sect_iter.next() {
            next_sect.decode(&mut self.sink).expect("Unexpected end of section");
        }
    }
}

impl<'buf, T: VectBase> Iterator for VectorItemIter<'buf, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.i < self.num_elems {
            let thing = self.sink.values[self.i % FIXED_LEN];
            self.i += 1;
            // If at boundary, get next_section
            if self.i % FIXED_LEN == 0 && self.i < self.num_elems {
                self.next_section();
            }
            Some(thing)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::filter::{EqualsSink, count_hits};

    #[test]
    fn test_append_u64_nonulls() {
        // Make sure the fixed sect stats above can still fit in total headers
        assert!(std::mem::size_of::<FixedSectStats>() + BINARYVECT_HEADER_SIZE <= NUM_HEADER_BYTES_TOTAL);

        // Append more than 256 values, see if we get two sections and the right data back
        let num_values: usize = 500;
        let data: Vec<u64> = (0..num_values as u64).collect();

        let mut appender = VectorU64Appender::try_new(1024).unwrap();
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

        let reader = VectorReader::try_new(&finished_vec[..]).unwrap();
        assert_eq!(reader.num_elements(), num_values);
        assert_eq!(reader.sect_iter().count(), 2);
        assert_eq!(reader.num_null_sections().unwrap(), 0);

        let elems: Vec<u64> = reader.iterate().collect();
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

        let mut appender = VectorU64Appender::try_new(1024).unwrap();
        data1.iter().for_each(|&e| appender.append(e).unwrap());
        appender.append_nulls(num_nulls).unwrap();
        data2.iter().for_each(|&e| appender.append(e).unwrap());

        let finished_vec = appender.finish(total_elems).unwrap();

        let reader = VectorReader::try_new(&finished_vec[..]).unwrap();
        assert_eq!(reader.num_elements(), total_elems);
        assert_eq!(reader.sect_iter().count(), 3);
        assert_eq!(reader.num_null_sections().unwrap(), 1);

        assert_eq!(reader.get_stats().num_null_sections, 1);

        let elems: Vec<u64> = reader.iterate().collect();
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

        let mut appender = VectorU64Appender::try_new(300).unwrap();
        data1.iter().for_each(|&e| appender.append(e).unwrap());
        appender.append_nulls(num_nulls).unwrap();
        data1.iter().for_each(|&e| appender.append(e).unwrap());
        appender.append_nulls(num_nulls).unwrap();

        let finished_vec = appender.finish(total_elems).unwrap();

        let reader = VectorReader::try_new(&finished_vec[..]).unwrap();
        println!("summary: {}", VectorStats::new(&reader).summary_string());
        assert_eq!(reader.num_elements(), total_elems);
        assert_eq!(reader.sect_iter().count(), 6);
        assert_eq!(reader.num_null_sections().unwrap(), 2);

        let elems: Vec<u64> = reader.iterate().collect();
        assert_eq!(elems, all_data);
    }

    #[test]
    fn test_append_u32_and_filter() {
        // First test appending with no nulls.  Just 1,2,3,4 and filter for 3, should get 1/4 of appended elements
        let vector_size = 400;
        let mut appender = VectorU32Appender::try_new(1024).unwrap();
        for i in 0..vector_size {
            appender.append((i % 4) + 1).unwrap();
        }
        let finished_vec = appender.finish(vector_size as usize).unwrap();

        let reader = VectorReader::<u32>::try_new(&finished_vec[..]).unwrap();
        assert_eq!(reader.num_elements(), vector_size as usize);
        assert_eq!(reader.sect_iter().count(), 2);

        let filter_iter = reader.filter_iter(EqualsSink::<u32>::new(&3));
        let count = count_hits(filter_iter) as u32;
        assert_eq!(count, vector_size / 4);

        // Test appending with stretches of nulls.  300, then 400 nulls, then 300 elements again
        let nonnulls = 300;
        let total_elems = nonnulls * 2 + 400;
        for i in 0..nonnulls {
            appender.append((i % 4) + 1).unwrap();
        }
        appender.append_nulls(400).unwrap();
        for i in 0..nonnulls {
            appender.append((i % 4) + 1).unwrap();
        }
        let finished_vec = appender.finish(total_elems as usize).unwrap();

        let reader = VectorReader::<u32>::try_new(&finished_vec[..]).unwrap();
        assert_eq!(reader.num_elements(), total_elems as usize);

        let filter_iter = reader.filter_iter(EqualsSink::<u32>::new(&3));
        let count = count_hits(filter_iter) as u32;
        assert_eq!(count, nonnulls * 2 / 4);

        // Iterate and decode_to_sink to VecSink should produce same values... except for trailing zeroes
        let mut sink = VecSink::<u32>::new();
        reader.decode_to_sink(&mut sink).unwrap();
        let it_data: Vec<u32> = reader.iterate().collect();
        assert_eq!(sink.vec[..total_elems as usize], it_data[..]);
    }

    #[test]
    fn test_append_u32_large_vector() {
        // 9999 nulls, then an item, 10 times = 100k items total
        let mut appender = VectorU32Appender::try_new(4096).unwrap();
        let vector_size = 100000;
        for _ in 0..10 {
            appender.append_nulls(9999).unwrap();
            appender.append(2).unwrap();
        }
        assert_eq!(appender.num_elements(), vector_size);

        let finished_vec = appender.finish(vector_size).unwrap();
        let reader = VectorReader::<u32>::try_new(&finished_vec[..]).unwrap();
        assert_eq!(reader.num_elements(), vector_size as usize);
    }

    #[test]
    fn test_read_wrong_type_error() {
        let vector_size = 400;
        let mut appender = VectorU32Appender::try_new(1024).unwrap();
        for i in 0..vector_size {
            appender.append((i % 4) + 1).unwrap();
        }
        let finished_vec = appender.finish(vector_size as usize).unwrap();

        let res = VectorReader::<u64>::try_new(&finished_vec[..]);
        assert_eq!(res.err().unwrap(), CodingError::WrongVectorType(VectorSubType::FixedU32 as u8))
    }
}

