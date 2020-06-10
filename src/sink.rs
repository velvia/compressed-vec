/// A sink processes data during unpacking.  The type, Input, is supposed to represent 8 integers of fixed width,
/// since NibblePack works on 8 ints at a time.
/// This module contains common sinks for all types, such as ones used for vector/section unpacking/iteration,
/// and sinks for writing to Vecs.
/// Sinks can be stacked for processing.  For example, unpack and multiply f32's, then store to Vec:
///     regular unpack8_u32_simd -> u32 to f32 XOR sink -> MultiplySink -> VecSink
/// TODO: examples
use crate::section::VectBase;

use num::Zero;
use packed_simd::{u32x8, u64x8};

/// An input to a sink.  Sinks take a type which represents 8 values of an int, such as [u64; 8].
/// Item type represents the underlying type of each individual item in the 8 item SinkInput.
pub trait SinkInput: Copy {
    type Item: Zero + Copy;

    /// Writes the sink input to a mutable slice of type Item
    fn write_to_slice(&self, slice: &mut [Self::Item]);

    /// Creates one of these types from a base Item type by splatting (replicating it 8x)
    fn splat(item: Self::Item) -> Self;

    /// Methods for implementing filtering/masking.
    /// Compares my 8 values to other 8 values, returning a bitmask for equality
    fn eq_mask(self, other: Self) -> u8;
}

// TODO: remove
impl SinkInput for [u64; 8] {
    type Item = u64;

    #[inline]
    fn write_to_slice(&self, slice: &mut [Self::Item]) { slice.copy_from_slice(self) }

    #[inline]
    fn splat(item: u64) -> Self { [item; 8] }

    #[inline]
    fn eq_mask(self, other: Self) -> u8 {
        let mut mask = 0u8;
        for i in 0..8 {
            if self[i] == other[i] {
                mask |= 1 << i;
            }
        }
        mask
    }
}

impl SinkInput for u64x8 {
    type Item = u64;

    #[inline]
    fn write_to_slice(&self, slice: &mut [Self::Item]) {
        self.write_to_slice_unaligned(slice);
    }

    #[inline]
    fn splat(item: u64) -> Self { u64x8::splat(item) }

    #[inline]
    fn eq_mask(self, other: Self) -> u8 {
        self.eq(other).bitmask()
    }
}

impl SinkInput for u32x8 {
    type Item = u32;

    #[inline]
    fn write_to_slice(&self, slice: &mut [Self::Item]) {
        // NOTE: use unaligned writes for now.  See simd_aligned for a possible solution.
        // Pointer check align_offset is not enabled for now.
        self.write_to_slice_unaligned(slice);
    }

    #[inline]
    fn splat(item: u32) -> Self { u32x8::splat(item) }

    #[inline]
    fn eq_mask(self, other: Self) -> u8 {
        self.eq(other).bitmask()
    }
}


/// A sink processes data during unpacking.  The type, Input, is supposed to represent 8 integers of fixed width,
/// since NibblePack works on 8 ints at a time.
pub trait Sink<Input: SinkInput> {
    /// Processes 8 items. Sink responsible for space allocation and safety.
    fn process(&mut self, data: Input);

    /// Called when all zeroes or 8 null outputs
    fn process_zeroes(&mut self);

    /// Resets state in the sink; exact meaning depends on the sink itself.  Many sinks operate on more than
    /// 8 items; for example 256 items or entire sections.
    fn reset(&mut self);
}


/// A Sink which writes all values to a Vec.  A good choice as the final Sink in a chain of Sink processors!
/// Important!  This Sink will decode entire sections at a time, so the result will have up to 255 extra values.
#[derive(Debug)]
pub struct VecSink<T: VectBase> {
    pub vec: Vec<T>,
}

const DEFAULT_CAPACITY: usize = 64;

impl<T: VectBase> VecSink<T> {
    pub fn new() -> Self {
        VecSink { vec: Vec::with_capacity(DEFAULT_CAPACITY) }
    }
}

impl<T: VectBase> Sink<T::SI> for VecSink<T> {
    #[inline]
    fn process(&mut self, data: T::SI) {
        // So first we need to resize the Vec, then we write in values using write_to_slice
        let new_len = self.vec.len() + 8;
        self.vec.resize(new_len, T::zero());
        data.write_to_slice(&mut self.vec[new_len-8..new_len]);
    }

    #[inline]
    fn process_zeroes(&mut self) {
        for _ in 0..8 {
            self.vec.push(T::zero());
        }
    }

    fn reset(&mut self) {
        self.vec.clear()
    }
}

// #[repr(simd)]  // SIMD 32x8 alignment
// struct U32Values([u32; 256]);

/// A simple sink storing up to 256 values in an array, ie all the values in a section.
/// Useful for iterating over or processing all the raw values of a section.
// NOTE (u32x8): we want to do fast aligned SIMD writes, but looks like that might not happen.
// See simd_aligned for a possible solution.  It is possible the alignment check might fail
// due to values being a [u32];.
// TODO for SIMD: Try using aligned crate (https://docs.rs/aligned/0.3.2/aligned/) and see if
// it allows for aligned writes
#[repr(align(32))]  // SIMD alignment?
pub struct Section256Sink<T>
where T: VectBase {
    pub values: [T; 256],
    i: usize,
    // _stc: PhantomData<STC>,  // Takes no space -- needed as implementation needs an STC
}

impl<T> Section256Sink<T>
where T: VectBase {
    pub fn new() -> Self {
        Self { values: [T::zero(); 256], i: 0 }
    }
}

impl<T> Sink<T::SI> for Section256Sink<T>
where T: VectBase {
    #[inline]
    fn process(&mut self, unpacked: T::SI) {
        if self.i < self.values.len() {
            unpacked.write_to_slice(&mut self.values[self.i..self.i+8]);
            self.i += 8;
        }
    }

    #[inline]
    fn process_zeroes(&mut self) {
        if self.i < self.values.len() {
            // We need to write zeroes in case the sink is reused; previous values won't be zero.
            // This is fairly fast in any case.  NOTE: fill() is a new API in nightly
            // Alternative, not quite as fast, is use copy_from_slice() and memcpy from zero slice
            self.values[self.i..self.i+8].fill(T::zero());
            self.i += 8;
        }
    }

    fn reset(&mut self) {
        self.i = 0;  // No need to zero things out, process() methods will fill properly
    }
}

pub type U32_256Sink = Section256Sink<u32>;
pub type U64_256Sink = Section256Sink<u64>;
