/// The `filter` module contains traits for fast filtering of vectors.
/// U32 vectors have SIMD-enabled filtering support for each section, which is
/// 256 elements long to enable SIMD bitmasking on AVX2 with a single instruction.
///
/// TODO: add examples for EqualsSink, OneOfSink, etc.
///
use core::marker::PhantomData;

use packed_simd::u32x8;
use smallvec::SmallVec;

use crate::section::*;
use crate::sink::{Sink, SinkInput};


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


/// A Predicate is the value(s) for a filter to filter against
pub trait Predicate<T: VectBase> {
    type Input;

    /// Returns true if the predicate matches null or zero values
    fn pred_matches_zero(input: &Self::Input) -> bool;

    /// Creates this predicate from a predicate input type
    fn from_input(input: &Self::Input) -> Self;
}

pub trait InnerFilter<T: VectBase> {
    type P: Predicate<T>;

    /// This method is called with the SinkInput from the decoder, and has to do filtering using
    /// the predicate type and return a bitmask; LSB=first item processed
    fn filter_bitmask(pred: &Self::P, decoded: T::SI) -> u8;
}

/// Sink designed to filter 8 items at a time from the decoder, building up a bitmask for each section.
/// It is generic for different predicates and base types.  Has optimizations for null sections.
#[repr(align(16))]   // To ensure the mask is aligned and can transmute to u32
#[derive(Debug)]
pub struct GenericFilterSink<T: VectBase, IF: InnerFilter<T>> {
    mask: [u8; 32],
    predicate: IF::P,
    i: usize,
    match_zero: bool,   // true if zero value will be matched by the predicate
}

impl<T: VectBase, IF: InnerFilter<T>> GenericFilterSink<T, IF> {
    pub fn new(input: &<IF::P as Predicate<T>>::Input) -> Self {
        Self {
            mask: [0u8; 32],
            predicate: IF::P::from_input(input),
            i: 0,
            match_zero: IF::P::pred_matches_zero(input),
        }
    }
}

impl<T: VectBase, IF: InnerFilter<T>> Sink<T::SI> for GenericFilterSink<T, IF> {
    #[inline]
    fn process_zeroes(&mut self) {
        self.mask[self.i] = if self.match_zero { 0xff } else { 0 };
        self.i += 1;
    }

    #[inline]
    fn process(&mut self, unpacked: T::SI) {
        self.mask[self.i] = IF::filter_bitmask(&self.predicate, unpacked);
        self.i += 1;
    }

    #[inline]
    fn reset(&mut self) {
        self.i = 0;
    }
}

const ALL_MATCHES: u32x8 = u32x8::splat(0xffff_ffff);  // All 1's
const NO_MATCHES: u32x8 = u32x8::splat(0);

impl<T: VectBase, IF: InnerFilter<T>> SectFilterSink<T> for GenericFilterSink<T, IF> {
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
        if self.match_zero { ALL_MATCHES } else { NO_MATCHES }
    }
}


///  A predicate containing 8 values (probably SIMD) of each type for single comparisons
// type SingleValuePredicate<T> = <T as VectBase>::SI;
pub struct SingleValuePredicate<T: VectBase> {
    pred: T::SI,
}

impl<T: VectBase> Predicate<T> for SingleValuePredicate<T> {
    type Input = T;
    #[inline]
    fn pred_matches_zero(input: &T) -> bool {
        input.is_zero()
    }

    #[inline]
    fn from_input(input: &T) -> Self {
        Self { pred: T::SI::splat(*input) }
    }
}

pub struct EqualsIF {}

impl<T: VectBase> InnerFilter<T> for EqualsIF {
    type P = SingleValuePredicate<T>;
    #[inline]
    fn filter_bitmask(p: &Self::P, decoded: T::SI) -> u8 {
        T::SI::eq_mask(p.pred, decoded)
    }
}

pub type EqualsSink<T> = GenericFilterSink<T, EqualsIF>;


///  A predicate for low cardinality SET membership (one of/IN matches), consisting of a Vec of 8 values each
pub struct MembershipPredicate<T: VectBase> {
    set: Vec<T::SI>,
}

impl<T: VectBase> Predicate<T> for MembershipPredicate<T> {
    // Up to 4 items in the set, heap allocation not needed
    type Input = SmallVec<[T; 4]>;
    #[inline]
    fn pred_matches_zero(input: &Self::Input) -> bool {
        // If any member of set is 0, then pred can match 0
        input.iter().any(|x| x.is_zero())
    }

    #[inline]
    fn from_input(input: &Self::Input) -> Self {
        Self { set: input.iter().map(|&item| T::SI::splat(item)).collect() }
    }
}

pub struct OneOfIF {}

impl<T: VectBase> InnerFilter<T> for OneOfIF {
    type P = MembershipPredicate<T>;
    #[inline]
    fn filter_bitmask(p: &Self::P, decoded: T::SI) -> u8 {
        // SIMD compare of decoded value with each of the predicates, OR resulting masks
        let mut mask = 0u8;
        for pred in &p.set {
            mask |= T::SI::eq_mask(*pred, decoded);
        }
        mask
    }
}

pub type OneOfSink<T> = GenericFilterSink<T, OneOfIF>;


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
    #[inline]
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
            .and_then(|res| {
                let sect = res.expect("This should not fail!");
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
pub fn count_hits<I>(filter_iter: I) -> usize
    where I: Iterator<Item = u32x8> {
    (filter_iter.map(|mask| mask.count_ones().wrapping_sum()).sum::<u32>()) as usize
}

/// Creates a Vec of the element positions where matches occur
pub fn match_positions<I>(filter_iter: I) -> Vec<usize>
where I: Iterator<Item = u32x8> {
    let mut pos = 0;
    let mut matches = Vec::<usize>::new();
    filter_iter.for_each(|mask| {
        for word in 0..8 {
            let u32mask = mask.extract(word);
            if u32mask != 0 {
                // TODO: find highest bit (intrinsic) for O(on bits) speed
                for bit in 0..32 {
                    if (u32mask & (1 << bit)) != 0 {
                        matches.push(pos);
                    }
                    pos += 1;
                }
            }
        }
    });
    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    use smallvec::smallvec;
    use crate::filter::match_positions;
    use crate::vector::{VectorU32Appender, VectorU64Appender, VectorReader};

    #[test]
    fn test_filter_u64_equals() {
        let vector_size: usize = 400;
        let mut appender = VectorU64Appender::new(1024).unwrap();
        for i in 0..vector_size {
            appender.append((i as u64 % 4) + 1).unwrap();
        }
        let finished_vec = appender.finish(vector_size).unwrap();

        let reader = VectorReader::<u64>::try_new(&finished_vec[..]).unwrap();
        let filter_iter = reader.filter_iter(EqualsSink::<u64>::new(&3));
        let matches = match_positions(filter_iter);
        assert_eq!(matches.len(), vector_size / 4);

        // 1, 2, 3... so match for 3 starts at position 2
        let expected_pos: Vec<_> = (2..vector_size).step_by(4).collect();
        assert_eq!(matches, expected_pos);
    }

    #[test]
    fn test_filter_u32_oneof() {
        let vector_size: usize = 400;
        let mut appender = VectorU32Appender::new(1024).unwrap();
        for i in 0..vector_size {
            appender.append((i as u32 % 12) + 1).unwrap();
        }
        let finished_vec = appender.finish(vector_size).unwrap();

        let reader = VectorReader::<u32>::try_new(&finished_vec[..]).unwrap();
        let filter_iter = reader.filter_iter(OneOfSink::<u32>::new(&smallvec![3, 5]));
        let matches = match_positions(filter_iter);

        // 3 and 5 are 1/6th of 12 values.  400/6=66 but 400%12=4, so the 3 is last value matched again
        assert_eq!(matches.len(), 67);

        // 3, 5 are positions 2, 4..... etc.
        let mut expected_pos: Vec<_> = (2..vector_size).step_by(12).map(|i| vec![i, i+2]).flatten().collect();
        // have to trim last item
        expected_pos.resize(67, 0);
        assert_eq!(matches, expected_pos);
    }
}
