#![allow(unused)] // needed for dbg!() macro, but folks say this should not be needed

use core::ops::BitAnd;
use std::ops::{Shl, Shr};

use crate::byteutils::*;
use crate::error::CodingError;
use crate::nibblepacking::*;

use lazy_static::*;
use packed_simd::{shuffle, u64x4, u32x8};


const ZEROES_U64X4: u64x4 = u64x4::splat(0);

#[inline(always)]
pub fn nibble_pack8_simd(inputs: &[u64; 8], out_buffer: &mut Vec<u8>) {
    let input_l = u64x4::from_slice_unaligned(inputs);
    let input_h = u64x4::from_slice_unaligned(&inputs[4..]);

    // Compute nonzero bitmask, comparing each input word to zeroes
    let zeroes = u64x4::splat(0);
    let nonzero_mask = input_h.ne(zeroes).bitmask() << 4 |
                       input_l.ne(zeroes).bitmask();
    out_buffer.push(nonzero_mask);

    if nonzero_mask != 0 {
        // Compute min of leading and trailing zeroes, using SIMD for speed.
        // Fastest way is to OR all the bits together, then can use the ORed bits to find leading/trailing zeroes
        let ored_bits = input_l.or() | input_h.or();
        let min_leading_zeros = ored_bits.leading_zeros();
        let min_trailing_zeros = ored_bits.trailing_zeros();

        // Convert min leading/trailing to # nibbles.  Start packing!
        // NOTE: num_nibbles cannot be 0; that would imply every input was zero
        let trailing_nibbles = min_trailing_zeros / 4;
        let num_nibbles = 16 - (min_leading_zeros / 4) - trailing_nibbles;
        let nibble_word = (((num_nibbles - 1) << 4) | trailing_nibbles) as u8;
        out_buffer.push(nibble_word);

        // TODO: finish this, convert to use &mut [u8] buf, and
        // use SIMD pack - basically the steps in unpack reversed.
        // Start with u32 - much easier and faster.

        // if (num_nibbles % 2) == 0 {
        //     pack_to_even_nibbles(inputs, out_buffer, num_nibbles, trailing_nibbles);
        // } else {
        //     pack_universal(inputs, out_buffer, num_nibbles, trailing_nibbles);
        // }
    }
}

// Variable shifts for each SIMD lane to decode NibblePacked data
const U32_SIMD_SHIFTS: [u32x8; 4] = [
    // 1 nibble / 4 bits for same u32 word
    u32x8::new(0, 4, 8, 12, 16, 20, 24, 28),
    // 2 nibbles: lower u32 (8 bits x 4), upper u32 (8 bits x 4)
    u32x8::new(0, 8, 16, 24, 0, 8, 16, 24),
    // 3 nibbles: 4 groups of u32 words (12 bits x 2)
    u32x8::new(0, 12, 0, 12, 0, 12, 0, 12),
    // 4 nibbles: 4 groups of u32 words (16 bits x 2)
    u32x8::new(0, 16, 0, 16, 0, 16, 0, 16),
];

// Bitmask for ANDing during SIMD unpacking
const U32_SIMD_ANDMASK: [u32x8; 4] = [
    // 1 nibble
    u32x8::splat(0x0f),
    // 2 nibbles, etc.
    u32x8::splat(0x0ff),
    u32x8::splat(0x0fff),
    u32x8::splat(0x0ffff),
];

const U32_SIMD_ZEROES: u32x8 = u32x8::splat(0);

// Shuffles used in unpacking.  Given input bitmask, it calculates the shuffle
// matrix needed to "expand" or move the elements to the right place given null elements.
// from is the source element number.
lazy_static! {
    static ref SHUFFLE_UNPACK_IDX_U32: [u32x8; 256] = {
        let mut shuffle_indices = [u32x8::splat(0); 256];
        for bitmask in 0usize..256 {
            let mut from_pos = 0;
            let mut indices = [0u32; 8];
            for to_pos in 0..8 {
                // If bit in bitmask is on, then map from_pos to current pos
                if bitmask & (1 << to_pos) != 0 {
                    indices[to_pos] = from_pos;
                    from_pos += 1;
                // If bit is off, then use the last index into which 0 is stuffed.
                } else {
                    indices[to_pos] = 7;
                }
            }
            shuffle_indices[bitmask as usize] = u32x8::from(indices);
        }
        shuffle_indices
    };
}

/// u32 SIMD sink
pub trait SinkU32 {
    /// Called when all zeroes or 8 null outputs
    fn process_zeroes(&mut self);
    /// Called for normal output
    fn process(&mut self, unpacked: u32x8);
}

// #[repr(simd)]  // SIMD 32x8 alignment
// struct U32Values([u32; 256]);

/// A simple sink storing up to 256 u32 values in an array
// NOTE: we want to do fast aligned SIMD writes, but looks like that might not happen.
// See simd_aligned for a possible solution.  It is possible the alignment check might fail
// due to values being a [u32];.
#[repr(align(32))]  // SIMD 32x8 alignment?
pub struct U32_256Sink {
    pub values: [u32; 256],
    i: usize,
}

impl U32_256Sink {
    pub fn new() -> Self {
        Self { values: [0u32; 256], i: 0 }
    }
}

impl SinkU32 for U32_256Sink {
    #[inline]
    fn process(&mut self, unpacked: u32x8) {
        if self.i < self.values.len() {
            // NOTE: use unaligned writes for now.  See simd_aligned for a possible solution.
            // Pointer check align_offset is not enabled for now.
            unpacked.write_to_slice_unaligned(&mut self.values[self.i..self.i+8]);
            self.i += 8;
        }
    }

    #[inline]
    fn process_zeroes(&mut self) {
        // NOP. just advance the pointer.  The values were already initialized to 0.
        self.i += 8;
    }
}

#[inline]
fn preload_u32x8_3_4_nibble(buf: &[u8],
                            stride: usize,
                            nonzeroes: u32) -> Result<(u32x8, u32), CodingError> {
    let total_bytes = (stride * nonzeroes as usize + 1) / 2;
    let inword1 = direct_read_uint_le(buf, 2)?;
    let words0 = inword1 as u32;
    let words1 = (inword1 >> (stride * 8)) as u32;
    let (words2, words3) = if (stride * 2) < total_bytes {
        // We have processed stride*2 bytes.  If total bytes is more than that, keep reading.
        let inword2 = direct_read_uint_le(buf, 2 + stride*2)?;
        (inword2 as u32, (inword2 >> (stride * 8)) as u32)
    } else { (0, 0) };
    let simd_word = u32x8::new(words0, words0, words1, words1, words2, words2, words3, words3);
    Ok((simd_word, total_bytes as u32))
}

/// SIMD-based decoding of NibblePacked data to u32x8.  Errors out if number of nibbles exceeds 8.
#[inline]
pub fn unpack8_u32_simd<'a, Output: SinkU32>(
    inbuf: &'a [u8],
    output: &mut Output,
) -> Result<&'a [u8], CodingError> {
    if inbuf.is_empty() { return Err(CodingError::NotEnoughSpace) }
    let nonzero_mask = inbuf[0];
    let nonzero_count = nonzero_mask.count_ones();
    if nonzero_mask == 0 {
        output.process_zeroes();
        Ok(&inbuf[1..])
    } else {
        // NOTE: if nonzero values, must be at least two more bytes: the nibble count and packed nibbles
        if inbuf.len() < 3 { return Err(CodingError::NotEnoughSpace) }
        let num_nibbles = (inbuf[1] >> 4) + 1;
        let num_nibs_1 = num_nibbles as usize - 1;
        let trailing_zeros = (inbuf[1] & 0x0f) * 4;

        // First step: load encoded bytes in parallel to SIMD registers
        // Also figure out how many bytes are taken up by packed nibbles
        let (simd_inputs, num_bytes) = match num_nibbles {
            1 => {   // one nibble, easy peasy.
                // Step 1. single nibble x 8 is u32, so we can just splat it  :)
                let encoded0 = direct_read_uint_le(inbuf, 2)? as u32;
                (u32x8::splat(encoded0), (nonzero_count + 1) / 2)
            },
            2 => {
                // 2 nibbles/byte: first 4 values gets lower 32 bits, second gets upper 32 bits
                let in_word = direct_read_uint_le(inbuf, 2)?;
                let lower_u32 = in_word as u32;
                let upper_u32 = (in_word >> 32) as u32;
                (u32x8::new(lower_u32, lower_u32, lower_u32, lower_u32,
                            upper_u32, upper_u32, upper_u32, upper_u32),
                 nonzero_count)
            },
            3 => preload_u32x8_3_4_nibble(inbuf, 3, nonzero_count)?,
            4 => preload_u32x8_3_4_nibble(inbuf, 4, nonzero_count)?,
            // TODO: support 5-8 nibbles.  Won't be anywhere near as efficient though  :/
            _ => return Err(CodingError::InvalidFormat(
                            format!("{:?} nibbles is too many for u32 decoder", num_nibbles))),
        };

        // Step 2. Variable right shift to shift each set of nibbles in right place
        let shifted = simd_inputs.shr(U32_SIMD_SHIFTS[num_nibs_1]);

        // Step 3. AND mask to strip upper bits, so each lane left with its own value
        let anded = shifted.bitand(U32_SIMD_ANDMASK[num_nibs_1]);

        // Step 4. Left shift for trailing zeroes, if needed
        let leftshifted = if (trailing_zeros == 0) { anded } else { anded.shl(trailing_zeros as u32) };

        // Step 5. Shuffle inputs based on nonzero mask to proper places
        let shuffled = if (nonzero_count == 8) { leftshifted } else {
            let shifted1 = leftshifted.replace(7, 0);  // Stuff 0 into unused final slot
            shifted1.shuffle1_dyn(SHUFFLE_UNPACK_IDX_U32[nonzero_mask as usize])
        };

        // Step 6. Send to sink, and advance input slice
        output.process(shuffled);
        Ok(&inbuf[(2 + num_bytes as usize)..])
    }
}

#[test]
fn test_unpack_u32simd_1_2nibbles() {
    let mut buf = [55u8; 512];
    dbg!(&SHUFFLE_UNPACK_IDX_U32[..32]);

    // 1 nibble, no nulls
    let mut sink = U32_256Sink::new();
    let data = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let written = pack_u64(data.iter().map(|&x| x as u64), &mut buf, 0).unwrap();
    let rest = unpack8_u32_simd(&buf[..written], &mut sink).unwrap();

    // should use up all but last 4 bytes, and first 8 bytes should be identical
    assert_eq!(rest.len(), 4);
    assert_eq!(sink.values[..8], data[..8]);

    unpack8_u32_simd(rest, &mut sink).unwrap();
    assert_eq!(sink.values[8..12], data[8..12]);

    // 2 nibbles, no nulls. NOTE; final values all are multiples of 16; this tests leading_zeroes == 4
    let mut sink = U32_256Sink::new();
    let data2 = [32u32, 34, 40, 48, 56, 72, 80, 88, 96, 112, 128, 144];
    let written = pack_u64(data2.iter().map(|&x| x as u64), &mut buf, 0).unwrap();
    let rest2 = unpack8_u32_simd(&buf[..written], &mut sink).unwrap();

    // assert_eq!(rest2.len(), 6);
    assert_eq!(sink.values[..8], data2[..8]);

    unpack8_u32_simd(rest2, &mut sink).unwrap();
    assert_eq!(sink.values[8..12], data2[8..12]);

    // 1 nibble, nulls
    let mut sink = U32_256Sink::new();
    let data = [1u32, 2, 0, 3, 4, 0, 5, 6, 0, 8, 9, 10, 0, 12];
    let written = pack_u64(data.iter().map(|&x| x as u64), &mut buf, 0).unwrap();
    let rest = unpack8_u32_simd(&buf[..written], &mut sink).unwrap();

    // should use up all but last 4 bytes, and first 8 bytes should be identical
    assert_eq!(rest.len(), 4);
    assert_eq!(sink.values[..8], data[..8]);

    unpack8_u32_simd(rest, &mut sink).unwrap();
    assert_eq!(sink.values[8..data.len()], data[8..]);

    // 2 nibbles, nulls
    let mut sink = U32_256Sink::new();
    let data2 = [32u32, 34, 40, 0, 0, 48, 56, 72, 80, 0, 88, 0, 96];
    let written = pack_u64(data2.iter().map(|&x| x as u64), &mut buf, 0).unwrap();
    let rest2 = unpack8_u32_simd(&buf[..written], &mut sink).unwrap();

    // assert_eq!(rest2.len(), 6);
    assert_eq!(sink.values[..8], data2[..8]);

    unpack8_u32_simd(rest2, &mut sink).unwrap();
    assert_eq!(sink.values[8..12], data2[8..12]);
}

fn make_nonzeroes_u64x64(num_nonzeroes: usize) -> [u64; 64] {
    let mut inputs = [0u64; 64];
    for i in 1..=num_nonzeroes {
        inputs[i] = (((i as f32) * std::f32::consts::PI / (num_nonzeroes as f32)).sin() * 1000.0) as u64
    }
    inputs
}

#[test]
fn test_unpack_u32simd_3_4nibbles() {
    dbg!(&make_nonzeroes_u64x64(63)[32..]);
    // Tests edge case where 4 nibbles (16 bits) pack edge
    // 4 nibbles = 2^16, so values < 65536
    let inputs = [65535u64; 8];
    let mut buf = [0u8; 512];
    let written = nibble_pack8(&inputs, &mut buf, 0).unwrap();

    let mut sink = U32_256Sink::new();
    let _rest = unpack8_u32_simd(&buf[..written], &mut sink).unwrap();

    assert_eq!(sink.values[..8], [65535u32; 8]);

    // case 2
    let mut sink = U32_256Sink::new();
    let inputs = [0u32, 1000, 1001, 1002, 1003, 2005, 2010, 3034, 4045, 5056, 6067, 7078];

    let written = pack_u64(inputs.iter().map(|&x| x as u64), &mut buf, 0).unwrap();
    let rest = unpack8_u32_simd(&buf[..written], &mut sink).unwrap();

    unpack8_u32_simd(rest, &mut sink).unwrap();
    assert_eq!(sink.values[..inputs.len()], inputs);
}

// NOTE: cfg(test) is needed so that proptest can just be a "dev-dependency" and not linked for final library
// NOTE2: somehow cargo is happier when we put props tests in its own module
#[cfg(test)]
mod props {
    extern crate proptest;

    use self::proptest::prelude::*;
    use super::*;

    // Generators (Arb's) for numbers of given # bits with fractional chance of being zero.
    // Also input arrays of 8 with the given properties above.
    prop_compose! {
        /// zero_chance: 0..1.0 chance of obtaining a zero
        fn arb_maybezero_nbits_u32
            (nbits: usize, zero_chance: f32)
            (is_zero in prop::bool::weighted(zero_chance as f64),
             n in 0u32..(1 << nbits))
            -> u32
        {
            if is_zero { 0 } else { n }
        }
    }

    // Generate random u32 source arrays
    prop_compose! {
        fn arb_u32_vectors()
                          (nbits in 4usize..16, chance in 0.1f32..0.6)
                          (mut v in proptest::collection::vec(arb_maybezero_nbits_u32(nbits, chance), 2..40))
         -> Vec<u32> { v }
    }

    proptest! {
        #[test]
        fn prop_u32simd_pack_unpack(input in arb_u32_vectors()) {
            let mut buf = [0u8; 2048];
            pack_u64(input.iter().map(|&x| x as u64), &mut buf, 0).unwrap();
            let mut sink = U32_256Sink::new();
            let res = unpack8_u32_simd(&buf, &mut sink).unwrap();
            let maxlen = 8.min(input.len());
            assert_eq!(sink.values[..maxlen], input[..maxlen]);
        }
    }
}
