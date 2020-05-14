/// The `filter` module contains traits for fast filtering of vectors.
/// U32 vectors have SIMD-enabled filtering support for each section, which is
/// 256 elements long to enable SIMD bitmasking on AVX2 with a single instruction.
///
use core::marker::PhantomData;

use packed_simd::u32x8;

use crate::section::*;
use crate::sink::Sink;


/// A Sink designed to filter 256-section vectors.  The workflow:
/// 1. Call sink.reset()
/// 2. Call decode on section with this sink
/// 3. get_mask()
/// - If the section is null, instead call null_mask()
pub trait SectFilterSink<T: VectBase>: Sink<T::SI> {
    /// Gets the mask, one bit is ON for each match in the section
    fn get_mask(&self) -> u32x8;

    /// Returns a mask when its a null section
    fn null_mask(&self) -> u32x8;
}


/// Sink for SIMD-enhanced U32 Equality filtering
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
}

impl Sink<u32x8> for EqualsU32Sink {
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

    #[inline]
    fn reset(&mut self) {
        self.i = 0;
    }
}

const ALL_MATCHES: u32x8 = u32x8::splat(0xffff_ffff);  // All 1's
const NO_MATCHES: u32x8 = u32x8::splat(0);

impl SectFilterSink<u32> for EqualsU32Sink {
    #[inline]
    fn get_mask(&self) -> u32x8 {
        // NOTE: we transmute the mask to u32; 8.  This is safe because we have aligned the struct for 16 bytes.
        let u32array = unsafe {
            std::mem::transmute::<[u8; 32], [u32; 8]>(self.mask)
        };
        u32x8::from(u32array)
    }

    #[inline]
    fn null_mask(&self) -> u32x8 {
        if self.pred_is_zero { ALL_MATCHES } else { NO_MATCHES }
    }
}

/// A Unary filter takes one mask input, does some kind of filtering and creates a new mask.
/// Filters that process and filter vectors are a subset of the above.
pub trait UnaryFilter {
    /// Filters input mask where each bit ON = match, and returns output mask
    fn filter(input: u32x8) -> u32x8;
}

/// Allows for filtering over each section of a vector.
/// Yields an Iterator of u32x8 mask for each section in the vector.
pub struct VectorFilter<'buf, SF, T>
where T: VectBase,
      SF: SectFilterSink<T> {
    sect_iter: FixedSectIterator<'buf>,
    sf: SF,
    _t: PhantomData<T>,
}

impl<'buf, SF, T> VectorFilter<'buf, SF, T>
where T: VectBase,
      SF: SectFilterSink<T> {
    pub fn new(vector_bytes: &'buf [u8], sf: SF) -> Self {
        Self { sect_iter: FixedSectIterator::new(vector_bytes), sf, _t: PhantomData }
    }

    /// Advances the iterator without calling the filter.  This is used to skip processing the filter
    /// for short circuiting.
    pub fn advance(&mut self) {
        self.sect_iter.next();
    }
}

impl<'buf, SF, T> Iterator for VectorFilter<'buf, SF, T>
where T: VectBase,
      SF: SectFilterSink<T>  {
    type Item = u32x8;

    #[inline]
    fn next(&mut self) -> Option<u32x8> {
        self.sect_iter.next()
            // .and_then(|sect| self.sf.filter_sect(sect).ok())
            .and_then(|sect| {
                if sect.is_null() {
                    Some(self.sf.null_mask())
                } else {
                    self.sf.reset();
                    sect.decode::<T, _>(&mut self.sf).ok()?;
                    Some(self.sf.get_mask())
                }
            })
    }
}

/// Helper to facilitate filtering multiple vectors at the same time,
/// this one filters by the same type of filter (eg all Equals).
/// For each group of sections, the same section filter masks are then ANDed together.
/// It has one optimization: it short-circuits the ANDing as soon as the masking creates
/// an all-zero mask.  Thus it makes sense to put the most sparse and least likely to hit
/// vector first.
pub struct MultiVectorFilter<'buf , SF, T>
where SF: SectFilterSink<T>,
       T: VectBase {
    vect_filters: Vec<VectorFilter<'buf, SF, T>>
}

impl<'buf, SF, T> MultiVectorFilter<'buf, SF, T>
where SF: SectFilterSink<T>,
       T: VectBase {
    pub fn new(vect_filters: Vec<VectorFilter<'buf, SF, T>>) -> Self {
        if vect_filters.is_empty() { panic!("Cannot pass in empty filters to MultiVectorFilter"); }
        Self { vect_filters }
    }
}

impl<'buf, SF, T> Iterator for MultiVectorFilter<'buf, SF, T>
where SF: SectFilterSink<T>,
       T: VectBase {
    type Item = u32x8;

    #[inline]
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

pub type EmptyFilter = std::iter::Empty<u32x8>;

pub const EMPTY_FILTER: EmptyFilter = std::iter::empty::<u32x8>();


/// Counts the output of VectorFilter iterator (or multiple VectorFilter results ANDed together)
/// for all the 1's in the output and returns the total
/// SIMD count_ones() is used for fast counting
pub fn count_hits<I>(filter_iter: I) -> u32
    where I: Iterator<Item = u32x8> {
    filter_iter.map(|mask| mask.count_ones().wrapping_sum()).sum()
}