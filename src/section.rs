/// A BinaryVector MAY consist of multiple sections.  Each section can represent
/// potentially different encoding parameters (bit widths, sparsity, etc.) and
/// has its own header to allow for quickly skipping ahead even when different
/// sections are encoded differently.   Or, one section may represent null data.
///
/// There are two varieties of sections represented.  See `SectionWriter` for variable-sized
/// sections, and see `FixedSection` for constant-length (number of elements) sections.
///
/// The code uses Scroll to ensure efficient encoding but one that works across platforms and endianness.

use std::convert::TryFrom;

use scroll::{ctx, Endian, Pread, Pwrite, LE};

/// For FixedSections this represents the first (and maybe only) byte of the section.
/// For SectionHeader based sections this is the byte at offset 4 into the header.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SectionType {
    Null = 0,                // n unavailable or null elements in a row
    NibblePackedU64Medium = 1,   // Nibble-packed u64's, total size < 64KB
}

impl TryFrom<u8> for SectionType {
    type Error = CodingError;
    fn try_from(n: u8) -> Result<SectionType, Self::Error> {
        match n {
            0 => Ok(SectionType::Null),
            1 => Ok(SectionType::NibblePackedU64Medium),
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

// TODO: move stupid thing to more general place
#[derive(Debug, PartialEq)]
pub enum CodingError {
    NotEnoughSpace,     //
    InvalidSectionType(u8),
    ScrollErr(String),
}

impl From<scroll::Error> for CodingError {
    fn from(err: scroll::Error) -> CodingError {
        match err {
            scroll::Error::TooBig { .. } => CodingError::NotEnoughSpace,
            _ => CodingError::ScrollErr(err.to_string()),
        }
    }
}

/// Result: (bytes_written, elements_written)
type CodingResult = Result<(u16, u16), CodingError>;

/// SectionWriter stores state for active writing of multiple SectionHeader-based sections in a vector.
/// It manages rollover from one section to another when there's not enough space.
/// The main API is `add_64kb` which uses a closure to fill in section contents without copying.
///
/// Example which adds 8 0xff elements and returns an error if there isn't enough space:
/// ```
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

