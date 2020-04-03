/// The `filter` module contains traits for fast filtering of vectors.
/// U32 vectors have SIMD-enabled filtering support for each section, which is
/// 256 elements long to enable SIMD bitmasking on AVX2 with a single instruction.
///
use std::collections::HashSet;

use packed_simd::u32x8;

use crate::error::CodingError;
use crate::section::*;
use crate::nibblepack_simd::SinkU32;

/// Filters on value exactly equal
pub struct EqualsU32 {
    pred: u32,
    sink: EqualsU32Sink,
}

impl EqualsU32 {
    pub fn new(pred: u32) -> Self {
        Self { pred, sink: EqualsU32Sink::new(pred) }
    }
}

/// Filters on value being one of the supplied values
pub struct OneOfU32(HashSet<u32>);

/// Sink for SIMD-enhanced U32 filtering
#[repr(align(16))]   // To ensure the mask is aligned and can transmute to u32
#[derive(Debug)]
pub struct EqualsU32Sink {
    mask: [u8; 32],
    predicate: u32x8,
    i: usize,
    pred_is_zero: bool,   // true if predicate is zero
}

impl EqualsU32Sink {
    pub fn new(pred: u32) -> Self {
        Self {
            mask: [0u8; 32],
            predicate: u32x8::splat(pred),
            i: 0,
            pred_is_zero: pred == 0
        }
    }

    pub fn get_mask(&self) -> u32x8 {
        // NOTE: we transmute the mask to u32; 8.  This is safe because we have aligned the struct for 16 bytes.
        let u32array = unsafe {
            std::mem::transmute::<[u8; 32], [u32; 8]>(self.mask)
        };
        u32x8::from(u32array)
    }

    pub fn reset(&mut self) {
        self.i = 0;
    }
}

impl SinkU32 for EqualsU32Sink {
    #[inline]
    fn process_zeroes(&mut self) {
        self.mask[self.i] = if self.pred_is_zero { 0xff } else { 0 };
        self.i += 1;
    }

    #[inline]
    fn process(&mut self, unpacked: u32x8) {
        self.mask[self.i] = unpacked.eq(self.predicate).bitmask();
        self.i += 1;
    }
}

/// For each type of filter, a SectionFilter implements how to filter that section
/// according to each filter type.
pub trait SectionFilter {
    // Filters each section, producing a mask of hits for each row in a 256-row section
    fn filter_sect(&mut self, sect: FixedSectEnum, sect_bytes: &[u8]) -> Result<u32x8, CodingError>;
}

const ALL_MATCHES: u32x8 = u32x8::splat(0xffff_ffff);  // All 1's
const NO_MATCHES: u32x8 = u32x8::splat(0);

impl SectionFilter for EqualsU32 {
    #[inline]
    fn filter_sect(&mut self, sect: FixedSectEnum, sect_bytes: &[u8]) -> Result<u32x8, CodingError> {
        match sect {
            FixedSectEnum::NullFixedSect(_) =>
                if self.pred == 0 { Ok(ALL_MATCHES) } else { Ok(NO_MATCHES) },
            FixedSectEnum::NibblePackU32MedFixedSect(_) => {
                self.sink.reset();
                NibblePackU32MedFixedSect::decode_to_sink(sect_bytes, &mut self.sink)?;
                Ok(self.sink.get_mask())
            },
            _ => panic!("Cannot use this filter on that section type, must be wrong vector"),
        }
        // If we are trying to match 0, then everything matches.  Otherwise, nothing matches!  Easy!
    }
}

/// Allows for filtering over each section of a vector.
/// Yields an Iterator of u32x8 mask for each section in the vector.
pub struct VectorFilter<'a, SF: SectionFilter> {
    sect_iter: FixedSectIterator<'a>,
    sf: SF,
}

impl<'a, SF: SectionFilter> VectorFilter<'a, SF> {
    pub fn new(vector_bytes: &'a [u8], sf: SF) -> VectorFilter<'a, SF> {
        Self { sect_iter: FixedSectIterator::new(vector_bytes), sf }
    }

    /// Advances the iterator without calling the filter.  This is used to skip processing the filter
    /// for short circuiting.
    pub fn advance(&mut self) {
        self.sect_iter.next();
    }
}

impl<'a, SF: SectionFilter> Iterator for VectorFilter<'a, SF> {
    type Item = u32x8;
    #[inline]
    fn next(&mut self) -> Option<u32x8> {
        self.sect_iter.next()
            .and_then(|(sect, s_bytes)| self.sf.filter_sect(sect, s_bytes).ok())
    }
}

/// Helper to facilitate filtering multiple vectors at the same time,
/// this one filters by the same type of filter (eg all Equals).
/// For each group of sections, the same section filter masks are then ANDed together.
/// It has one optimization: it short-circuits the ANDing as soon as the masking creates
/// an all-zero mask.  Thus it makes sense to put the most sparse and least likely to hit
/// vector first.
pub struct MultiVectorFilter<'a , SF: SectionFilter> {
    vect_filters: Vec<VectorFilter<'a, SF>>
}

impl<'a, SF: SectionFilter> MultiVectorFilter<'a, SF> {
    pub fn new(vect_filters: Vec<VectorFilter<'a, SF>>) -> Self {
        if vect_filters.is_empty() { panic!("Cannot pass in empty filters to MultiVectorFilter"); }
        Self { vect_filters }
    }
}

impl<'a, SF: SectionFilter> Iterator for MultiVectorFilter<'a, SF> {
    type Item = u32x8;
    fn next(&mut self) -> Option<u32x8> {
        // Get first filter
        let mut mask = match self.vect_filters[0].next() {
            Some(m) => m,
            None    => return None,   // Assume end of one vector is end of all
        };
        let mut n = 1;

        // Keep going if filter is not empty and there are still vectors to go
        while n < self.vect_filters.len() && mask != NO_MATCHES {
            mask &= match self.vect_filters[n].next() {
                Some(m) => m,
                None    => return None,
            };
            n += 1;
        }

        // short-circuit: just advance the iterator if we're already at zero mask.
        // No need to do expensive filtering.
        while n < self.vect_filters.len() {
            self.vect_filters[n].advance();
            n += 1;
        }

        Some(mask)
    }
}

/// Counts the output of VectorFilter iterator (or multiple VectorFilter results ANDed together)
/// for all the 1's in the output and returns the total
/// SIMD count_ones() is used for fast counting
pub fn count_hits<I>(filter_iter: I) -> u32
    where I: Iterator<Item = u32x8> {
    filter_iter.map(|mask| mask.count_ones().wrapping_sum()).sum()
}