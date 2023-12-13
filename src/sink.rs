/// A sink processes data during unpacking.  The type, Input, is supposed to represent 8 integers of fixed width,
/// since NibblePack works on 8 ints at a time.
/// This module contains common sinks for all types, such as ones used for vector/section unpacking/iteration,
/// and sinks for writing to Vecs.
/// Sinks can be stacked for processing.  For example, unpack and multiply f32's, then store to Vec:
///     regular unpack8_u32_simd -> u32 to f32 XOR sink -> MultiplySink -> VecSink
/// TODO: examples
use core::marker::PhantomData;
use std::ops::{Add, BitXor};

use crate::section::VectBase;

use num::{Zero, Unsigned, Float};
use packed_simd::{u32x8, u64x8, f32x8, FromCast, FromBits, IntoBits};

/// An input to a sink.  Sinks take a type which represents 8 values of an int, such as [u64; 8].
/// Item type represents the underlying type of each individual item in the 8 item SinkInput.
pub trait SinkInput: Copy + core::fmt::Debug {
    type Item: Zero + Copy;

    const ZERO: Self;    // The zero item for myself

    /// Writes the sink input to a mutable slice of type Item
    fn write_to_slice(&self, slice: &mut [Self::Item]);

    /// Creates one of these types from a base Item type by splatting (replicating it 8x)
    fn splat(item: Self::Item) -> Self;

    /// Methods for implementing filtering/masking.
    /// Compares my 8 values to other 8 values, returning a bitmask for equality
    fn eq_mask(self, other: Self) -> u8;

    /// Loads the bits from a slice into a u64x8. Mostly used for converting FP bits to int bits for XORing.
    fn to_u64x8_bits(slice: &[Self::Item]) -> u64x8;
}

// TODO: remove
impl SinkInput for [u64; 8] {
    type Item = u64;
    const ZERO: [u64; 8] = [0u64; 8];

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

    #[inline]
    fn to_u64x8_bits(_slice: &[u64]) -> u64x8 { todo!("blah") }
}

impl SinkInput for u64x8 {
    type Item = u64;
    const ZERO: u64x8 = u64x8::splat(0);

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

    #[inline]
    fn to_u64x8_bits(slice: &[u64]) -> u64x8 { u64x8::from_slice_unaligned(slice) }
}

impl SinkInput for u32x8 {
    type Item = u32;
    const ZERO: u32x8 = u32x8::splat(0);

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

    #[inline]
    fn to_u64x8_bits(slice: &[u32]) -> u64x8 {
        u64x8::from_cast(u32x8::from_slice_unaligned(slice))
    }
}

impl SinkInput for f32x8 {
    type Item = f32;
    const ZERO: f32x8 = f32x8::splat(0.0);

    #[inline]
    fn write_to_slice(&self, slice: &mut [f32]) {
        self.write_to_slice_unaligned(slice);
    }

    #[inline]
    fn splat(item: f32) -> Self { f32x8::splat(item) }

    #[inline]
    fn eq_mask(self, other: Self) -> u8 {
        self.eq(other).bitmask()
    }

    #[inline]
    fn to_u64x8_bits(slice: &[f32]) -> u64x8 {
        let f_bits: u32x8 = f32x8::from_slice_unaligned(slice).into_bits();
        u64x8::from_cast(f_bits)
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


/// A sink for FP/XOR decoding.  Keeps a running "last bits" octet and XORs each new octet with the last one.
/// Forwards resulting XORed/restored output to another sink.
#[derive(Debug)]
pub struct XorSink<'a, F, I, S>
where F: VectBase + Float,      // Output floating point type
      I: VectBase + Unsigned,   // Input: unsigned (u32/u64) int type
      S: Sink<F::SI> {
    last_bits: I::SI,
    inner_sink: &'a mut S,
    _f: PhantomData<F>,
}

impl<'a, F, I, S> XorSink<'a, F, I, S>
where F: VectBase + Float,      // Output floating point type
      I: VectBase + Unsigned,   // Input: unsigned (u32/u64) type
      S: Sink<F::SI> {
    pub fn new(inner_sink: &'a mut S) -> Self {
        Self {
            last_bits: I::SI::ZERO,
            inner_sink,
            _f: PhantomData,
        }
    }
}

impl<'a, F, I, S> Sink<I::SI> for XorSink<'a, F, I, S>
where F: VectBase + Float,      // Output floating point type
      I: VectBase + Unsigned,   // Input: unsigned (u32/u64) type
      S: Sink<F::SI>,
      // bitxor is supported for underlying int types, and into_bits supported to/from FP types
      I::SI: BitXor<I::SI, Output = I::SI>,
      F::SI: FromBits<I::SI> {
    #[inline]
    fn process(&mut self, unpacked: I::SI) where   {
        let new_bits = self.last_bits.bitxor(unpacked);
        self.last_bits = new_bits;
        self.inner_sink.process(new_bits.into_bits());
    }

    #[inline]
    fn process_zeroes(&mut self) {
        // last XOR 0 == last
        self.inner_sink.process(self.last_bits.into_bits());
    }

    fn reset(&mut self) {}
}

/// A Sink for adding a constant value to all output elements.  Note that all SIMD types we use also support Add :)
/// This sink is also used for decoding Delta-encoded u64/u32 values, then passing the output to another sink.
#[derive(Debug)]
pub struct AddConstSink<'a, T, S>
where T: VectBase,
      S: Sink<T::SI> {
    base: T::SI,
    inner_sink: &'a mut S,
}

impl<'a, T, S> AddConstSink<'a, T, S>
where T: VectBase,
      S: Sink<T::SI> {
    pub fn new(base: T, inner_sink: &'a mut S) -> Self {
        Self { base: T::SI::splat(base), inner_sink }
    }
}

impl<'a, T, S> Sink<T::SI> for AddConstSink<'a, T, S>
where T: VectBase,
      S: Sink<T::SI>,
      T::SI: Add<T::SI, Output = T::SI> {
    #[inline]
    fn process(&mut self, unpacked: T::SI) {
        self.inner_sink.process(unpacked + self.base);
    }

    #[inline]
    fn process_zeroes(&mut self) {
        // base + 0 == base
        self.inner_sink.process(self.base);
    }

    fn reset(&mut self) {}
}