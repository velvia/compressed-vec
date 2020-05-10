#![allow(unused)]    // needed for dbg!() macro, but folks say this should not be needed
#![feature(slice_fill)]

use core::ops::BitAnd;
use std::ops::{Shl, Shr};

use crate::byteutils::*;
use crate::error::CodingError;
use crate::nibblepacking::*;
use crate::sink::*;

use packed_simd::{shuffle, u64x4, u32x8, m32x8, isizex8, cptrx8};


const ZEROES_U64X4: u64x4 = u64x4::splat(0);
const ZEROES_U32X8: u32x8 = u32x8::splat(0);

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
const U32_SIMD_SHIFTS: [u32x8; 9] = [
    // 0 nibbles: this should never be used
    u32x8::splat(0),
    // 1 nibble / 4 bits for same u32 word
    u32x8::new(0, 4, 8, 12, 16, 20, 24, 28),
    // 2 nibbles: lower u32 (8 bits x 4), upper u32 (8 bits x 4)
    u32x8::new(0, 8, 16, 24, 0, 8, 16, 24),
    // 3 nibbles: 4 groups of u32 words (12 bits x 2)
    u32x8::new(0, 12, 0, 12, 0, 12, 0, 12),
    // 4 nibbles: 4 groups of u32 words (16 bits x 2)
    u32x8::new(0, 16, 0, 16, 0, 16, 0, 16),
    // 5-8 nibbles: 8 u32 words shifted
    u32x8::new(0, 4, 0, 4, 0, 4, 0, 4),
    u32x8::splat(0),
    u32x8::new(0, 4, 0, 4, 0, 4, 0, 4),
    u32x8::splat(0),
];

// Byte offsets for reading U32 values from memory vs number of nibbles.
// Combined with U32_SIMD_SHIFTS, allows us to place shifted U32 values into each lane.
const U32_SIMD_PTR_OFFSETS: [isizex8; 9] = [
    // 0 nibbles: should never be used
    isizex8::splat(0),
    // 1 nibble, 4x8 bits fits into one u32, so no offset
    isizex8::splat(0),
    // 2 nibbles, 8x8 bits, two u32s offset by 4 bytes
    isizex8::new(0, 0, 0, 0, 4, 4, 4, 4),
    // 3 nibbles. 4 groups of u32s 3 bytes apart
    isizex8::new(0, 0, 3, 3, 6, 6, 9, 9),
    // 4 nibbles. 4 groups of u32s 4 bytes apart
    isizex8::new(0, 0, 4, 4, 8, 8, 12, 12),
    // 5-8 nibbles: individual u32 words spaced apart
    isizex8::new(0, 2, 5, 7, 10, 12, 15, 17),
    isizex8::new(0, 3, 6, 9, 12, 15, 18, 21),
    isizex8::new(0, 3, 7, 10, 14, 17, 21, 24),
    isizex8::new(0, 4, 8, 12, 16, 20, 24, 28),
];

// Bitmask for ANDing during SIMD unpacking
const U32_SIMD_ANDMASK: [u32x8; 9] = [
    u32x8::splat(0x0f),
    // 1 nibble
    u32x8::splat(0x0f),
    // 2 nibbles, etc.
    u32x8::splat(0x0ff),
    u32x8::splat(0x0fff),
    u32x8::splat(0x0ffff),
    u32x8::splat(0x0f_ffff),
    u32x8::splat(0x0ff_ffff),
    u32x8::splat(0x0fff_ffff),
    u32x8::splat(0xffff_ffff),
];

const U32_SIMD_ZEROES: u32x8 = u32x8::splat(0);

// Shuffles used in unpacking.  Given input bitmask, it calculates the shuffle
// matrix needed to "expand" or move the elements to the right place given null elements.
// from is the source element number.  NOTE: lazy_static was too slow so these constants were
// generated using the following code
// lazy_static! {
//     static ref SHUFFLE_UNPACK_IDX_U32: [u32x8; 256] = {
//         let mut shuffle_indices = [u32x8::splat(0); 256];
//         for bitmask in 0usize..256 {
//             let mut from_pos = 0;
//             let mut indices = [0u32; 8];
//             for to_pos in 0..8 {
//                 // If bit in bitmask is on, then map from_pos to current pos
//                 if bitmask & (1 << to_pos) != 0 {
//                     indices[to_pos] = from_pos;
//                     from_pos += 1;
//                 // If bit is off, then use the last index into which 0 is stuffed.
//                 } else {
//                     indices[to_pos] = 7;
//                 }
//             }
//             shuffle_indices[bitmask as usize] = u32x8::from(indices);
//         }
//         shuffle_indices
//     };
// }

const SHUFFLE_UNPACK_IDX_U32: [u32x8; 256] = [
    u32x8::new(7, 7, 7, 7, 7, 7, 7, 7),
    u32x8::new(0, 7, 7, 7, 7, 7, 7, 7),
    u32x8::new(7, 0, 7, 7, 7, 7, 7, 7),
    u32x8::new(0, 1, 7, 7, 7, 7, 7, 7),
    u32x8::new(7, 7, 0, 7, 7, 7, 7, 7),
    u32x8::new(0, 7, 1, 7, 7, 7, 7, 7),
    u32x8::new(7, 0, 1, 7, 7, 7, 7, 7),
    u32x8::new(0, 1, 2, 7, 7, 7, 7, 7),
    u32x8::new(7, 7, 7, 0, 7, 7, 7, 7),
    u32x8::new(0, 7, 7, 1, 7, 7, 7, 7),
    u32x8::new(7, 0, 7, 1, 7, 7, 7, 7),
    u32x8::new(0, 1, 7, 2, 7, 7, 7, 7),
    u32x8::new(7, 7, 0, 1, 7, 7, 7, 7),
    u32x8::new(0, 7, 1, 2, 7, 7, 7, 7),
    u32x8::new(7, 0, 1, 2, 7, 7, 7, 7),
    u32x8::new(0, 1, 2, 3, 7, 7, 7, 7),
    u32x8::new(7, 7, 7, 7, 0, 7, 7, 7),
    u32x8::new(0, 7, 7, 7, 1, 7, 7, 7),
    u32x8::new(7, 0, 7, 7, 1, 7, 7, 7),
    u32x8::new(0, 1, 7, 7, 2, 7, 7, 7),
    u32x8::new(7, 7, 0, 7, 1, 7, 7, 7),
    u32x8::new(0, 7, 1, 7, 2, 7, 7, 7),
    u32x8::new(7, 0, 1, 7, 2, 7, 7, 7),
    u32x8::new(0, 1, 2, 7, 3, 7, 7, 7),
    u32x8::new(7, 7, 7, 0, 1, 7, 7, 7),
    u32x8::new(0, 7, 7, 1, 2, 7, 7, 7),
    u32x8::new(7, 0, 7, 1, 2, 7, 7, 7),
    u32x8::new(0, 1, 7, 2, 3, 7, 7, 7),
    u32x8::new(7, 7, 0, 1, 2, 7, 7, 7),
    u32x8::new(0, 7, 1, 2, 3, 7, 7, 7),
    u32x8::new(7, 0, 1, 2, 3, 7, 7, 7),
    u32x8::new(0, 1, 2, 3, 4, 7, 7, 7),
    u32x8::new(7, 7, 7, 7, 7, 0, 7, 7),
    u32x8::new(0, 7, 7, 7, 7, 1, 7, 7),
    u32x8::new(7, 0, 7, 7, 7, 1, 7, 7),
    u32x8::new(0, 1, 7, 7, 7, 2, 7, 7),
    u32x8::new(7, 7, 0, 7, 7, 1, 7, 7),
    u32x8::new(0, 7, 1, 7, 7, 2, 7, 7),
    u32x8::new(7, 0, 1, 7, 7, 2, 7, 7),
    u32x8::new(0, 1, 2, 7, 7, 3, 7, 7),
    u32x8::new(7, 7, 7, 0, 7, 1, 7, 7),
    u32x8::new(0, 7, 7, 1, 7, 2, 7, 7),
    u32x8::new(7, 0, 7, 1, 7, 2, 7, 7),
    u32x8::new(0, 1, 7, 2, 7, 3, 7, 7),
    u32x8::new(7, 7, 0, 1, 7, 2, 7, 7),
    u32x8::new(0, 7, 1, 2, 7, 3, 7, 7),
    u32x8::new(7, 0, 1, 2, 7, 3, 7, 7),
    u32x8::new(0, 1, 2, 3, 7, 4, 7, 7),
    u32x8::new(7, 7, 7, 7, 0, 1, 7, 7),
    u32x8::new(0, 7, 7, 7, 1, 2, 7, 7),
    u32x8::new(7, 0, 7, 7, 1, 2, 7, 7),
    u32x8::new(0, 1, 7, 7, 2, 3, 7, 7),
    u32x8::new(7, 7, 0, 7, 1, 2, 7, 7),
    u32x8::new(0, 7, 1, 7, 2, 3, 7, 7),
    u32x8::new(7, 0, 1, 7, 2, 3, 7, 7),
    u32x8::new(0, 1, 2, 7, 3, 4, 7, 7),
    u32x8::new(7, 7, 7, 0, 1, 2, 7, 7),
    u32x8::new(0, 7, 7, 1, 2, 3, 7, 7),
    u32x8::new(7, 0, 7, 1, 2, 3, 7, 7),
    u32x8::new(0, 1, 7, 2, 3, 4, 7, 7),
    u32x8::new(7, 7, 0, 1, 2, 3, 7, 7),
    u32x8::new(0, 7, 1, 2, 3, 4, 7, 7),
    u32x8::new(7, 0, 1, 2, 3, 4, 7, 7),
    u32x8::new(0, 1, 2, 3, 4, 5, 7, 7),
    u32x8::new(7, 7, 7, 7, 7, 7, 0, 7),
    u32x8::new(0, 7, 7, 7, 7, 7, 1, 7),
    u32x8::new(7, 0, 7, 7, 7, 7, 1, 7),
    u32x8::new(0, 1, 7, 7, 7, 7, 2, 7),
    u32x8::new(7, 7, 0, 7, 7, 7, 1, 7),
    u32x8::new(0, 7, 1, 7, 7, 7, 2, 7),
    u32x8::new(7, 0, 1, 7, 7, 7, 2, 7),
    u32x8::new(0, 1, 2, 7, 7, 7, 3, 7),
    u32x8::new(7, 7, 7, 0, 7, 7, 1, 7),
    u32x8::new(0, 7, 7, 1, 7, 7, 2, 7),
    u32x8::new(7, 0, 7, 1, 7, 7, 2, 7),
    u32x8::new(0, 1, 7, 2, 7, 7, 3, 7),
    u32x8::new(7, 7, 0, 1, 7, 7, 2, 7),
    u32x8::new(0, 7, 1, 2, 7, 7, 3, 7),
    u32x8::new(7, 0, 1, 2, 7, 7, 3, 7),
    u32x8::new(0, 1, 2, 3, 7, 7, 4, 7),
    u32x8::new(7, 7, 7, 7, 0, 7, 1, 7),
    u32x8::new(0, 7, 7, 7, 1, 7, 2, 7),
    u32x8::new(7, 0, 7, 7, 1, 7, 2, 7),
    u32x8::new(0, 1, 7, 7, 2, 7, 3, 7),
    u32x8::new(7, 7, 0, 7, 1, 7, 2, 7),
    u32x8::new(0, 7, 1, 7, 2, 7, 3, 7),
    u32x8::new(7, 0, 1, 7, 2, 7, 3, 7),
    u32x8::new(0, 1, 2, 7, 3, 7, 4, 7),
    u32x8::new(7, 7, 7, 0, 1, 7, 2, 7),
    u32x8::new(0, 7, 7, 1, 2, 7, 3, 7),
    u32x8::new(7, 0, 7, 1, 2, 7, 3, 7),
    u32x8::new(0, 1, 7, 2, 3, 7, 4, 7),
    u32x8::new(7, 7, 0, 1, 2, 7, 3, 7),
    u32x8::new(0, 7, 1, 2, 3, 7, 4, 7),
    u32x8::new(7, 0, 1, 2, 3, 7, 4, 7),
    u32x8::new(0, 1, 2, 3, 4, 7, 5, 7),
    u32x8::new(7, 7, 7, 7, 7, 0, 1, 7),
    u32x8::new(0, 7, 7, 7, 7, 1, 2, 7),
    u32x8::new(7, 0, 7, 7, 7, 1, 2, 7),
    u32x8::new(0, 1, 7, 7, 7, 2, 3, 7),
    u32x8::new(7, 7, 0, 7, 7, 1, 2, 7),
    u32x8::new(0, 7, 1, 7, 7, 2, 3, 7),
    u32x8::new(7, 0, 1, 7, 7, 2, 3, 7),
    u32x8::new(0, 1, 2, 7, 7, 3, 4, 7),
    u32x8::new(7, 7, 7, 0, 7, 1, 2, 7),
    u32x8::new(0, 7, 7, 1, 7, 2, 3, 7),
    u32x8::new(7, 0, 7, 1, 7, 2, 3, 7),
    u32x8::new(0, 1, 7, 2, 7, 3, 4, 7),
    u32x8::new(7, 7, 0, 1, 7, 2, 3, 7),
    u32x8::new(0, 7, 1, 2, 7, 3, 4, 7),
    u32x8::new(7, 0, 1, 2, 7, 3, 4, 7),
    u32x8::new(0, 1, 2, 3, 7, 4, 5, 7),
    u32x8::new(7, 7, 7, 7, 0, 1, 2, 7),
    u32x8::new(0, 7, 7, 7, 1, 2, 3, 7),
    u32x8::new(7, 0, 7, 7, 1, 2, 3, 7),
    u32x8::new(0, 1, 7, 7, 2, 3, 4, 7),
    u32x8::new(7, 7, 0, 7, 1, 2, 3, 7),
    u32x8::new(0, 7, 1, 7, 2, 3, 4, 7),
    u32x8::new(7, 0, 1, 7, 2, 3, 4, 7),
    u32x8::new(0, 1, 2, 7, 3, 4, 5, 7),
    u32x8::new(7, 7, 7, 0, 1, 2, 3, 7),
    u32x8::new(0, 7, 7, 1, 2, 3, 4, 7),
    u32x8::new(7, 0, 7, 1, 2, 3, 4, 7),
    u32x8::new(0, 1, 7, 2, 3, 4, 5, 7),
    u32x8::new(7, 7, 0, 1, 2, 3, 4, 7),
    u32x8::new(0, 7, 1, 2, 3, 4, 5, 7),
    u32x8::new(7, 0, 1, 2, 3, 4, 5, 7),
    u32x8::new(0, 1, 2, 3, 4, 5, 6, 7),
    u32x8::new(7, 7, 7, 7, 7, 7, 7, 0),
    u32x8::new(0, 7, 7, 7, 7, 7, 7, 1),
    u32x8::new(7, 0, 7, 7, 7, 7, 7, 1),
    u32x8::new(0, 1, 7, 7, 7, 7, 7, 2),
    u32x8::new(7, 7, 0, 7, 7, 7, 7, 1),
    u32x8::new(0, 7, 1, 7, 7, 7, 7, 2),
    u32x8::new(7, 0, 1, 7, 7, 7, 7, 2),
    u32x8::new(0, 1, 2, 7, 7, 7, 7, 3),
    u32x8::new(7, 7, 7, 0, 7, 7, 7, 1),
    u32x8::new(0, 7, 7, 1, 7, 7, 7, 2),
    u32x8::new(7, 0, 7, 1, 7, 7, 7, 2),
    u32x8::new(0, 1, 7, 2, 7, 7, 7, 3),
    u32x8::new(7, 7, 0, 1, 7, 7, 7, 2),
    u32x8::new(0, 7, 1, 2, 7, 7, 7, 3),
    u32x8::new(7, 0, 1, 2, 7, 7, 7, 3),
    u32x8::new(0, 1, 2, 3, 7, 7, 7, 4),
    u32x8::new(7, 7, 7, 7, 0, 7, 7, 1),
    u32x8::new(0, 7, 7, 7, 1, 7, 7, 2),
    u32x8::new(7, 0, 7, 7, 1, 7, 7, 2),
    u32x8::new(0, 1, 7, 7, 2, 7, 7, 3),
    u32x8::new(7, 7, 0, 7, 1, 7, 7, 2),
    u32x8::new(0, 7, 1, 7, 2, 7, 7, 3),
    u32x8::new(7, 0, 1, 7, 2, 7, 7, 3),
    u32x8::new(0, 1, 2, 7, 3, 7, 7, 4),
    u32x8::new(7, 7, 7, 0, 1, 7, 7, 2),
    u32x8::new(0, 7, 7, 1, 2, 7, 7, 3),
    u32x8::new(7, 0, 7, 1, 2, 7, 7, 3),
    u32x8::new(0, 1, 7, 2, 3, 7, 7, 4),
    u32x8::new(7, 7, 0, 1, 2, 7, 7, 3),
    u32x8::new(0, 7, 1, 2, 3, 7, 7, 4),
    u32x8::new(7, 0, 1, 2, 3, 7, 7, 4),
    u32x8::new(0, 1, 2, 3, 4, 7, 7, 5),
    u32x8::new(7, 7, 7, 7, 7, 0, 7, 1),
    u32x8::new(0, 7, 7, 7, 7, 1, 7, 2),
    u32x8::new(7, 0, 7, 7, 7, 1, 7, 2),
    u32x8::new(0, 1, 7, 7, 7, 2, 7, 3),
    u32x8::new(7, 7, 0, 7, 7, 1, 7, 2),
    u32x8::new(0, 7, 1, 7, 7, 2, 7, 3),
    u32x8::new(7, 0, 1, 7, 7, 2, 7, 3),
    u32x8::new(0, 1, 2, 7, 7, 3, 7, 4),
    u32x8::new(7, 7, 7, 0, 7, 1, 7, 2),
    u32x8::new(0, 7, 7, 1, 7, 2, 7, 3),
    u32x8::new(7, 0, 7, 1, 7, 2, 7, 3),
    u32x8::new(0, 1, 7, 2, 7, 3, 7, 4),
    u32x8::new(7, 7, 0, 1, 7, 2, 7, 3),
    u32x8::new(0, 7, 1, 2, 7, 3, 7, 4),
    u32x8::new(7, 0, 1, 2, 7, 3, 7, 4),
    u32x8::new(0, 1, 2, 3, 7, 4, 7, 5),
    u32x8::new(7, 7, 7, 7, 0, 1, 7, 2),
    u32x8::new(0, 7, 7, 7, 1, 2, 7, 3),
    u32x8::new(7, 0, 7, 7, 1, 2, 7, 3),
    u32x8::new(0, 1, 7, 7, 2, 3, 7, 4),
    u32x8::new(7, 7, 0, 7, 1, 2, 7, 3),
    u32x8::new(0, 7, 1, 7, 2, 3, 7, 4),
    u32x8::new(7, 0, 1, 7, 2, 3, 7, 4),
    u32x8::new(0, 1, 2, 7, 3, 4, 7, 5),
    u32x8::new(7, 7, 7, 0, 1, 2, 7, 3),
    u32x8::new(0, 7, 7, 1, 2, 3, 7, 4),
    u32x8::new(7, 0, 7, 1, 2, 3, 7, 4),
    u32x8::new(0, 1, 7, 2, 3, 4, 7, 5),
    u32x8::new(7, 7, 0, 1, 2, 3, 7, 4),
    u32x8::new(0, 7, 1, 2, 3, 4, 7, 5),
    u32x8::new(7, 0, 1, 2, 3, 4, 7, 5),
    u32x8::new(0, 1, 2, 3, 4, 5, 7, 6),
    u32x8::new(7, 7, 7, 7, 7, 7, 0, 1),
    u32x8::new(0, 7, 7, 7, 7, 7, 1, 2),
    u32x8::new(7, 0, 7, 7, 7, 7, 1, 2),
    u32x8::new(0, 1, 7, 7, 7, 7, 2, 3),
    u32x8::new(7, 7, 0, 7, 7, 7, 1, 2),
    u32x8::new(0, 7, 1, 7, 7, 7, 2, 3),
    u32x8::new(7, 0, 1, 7, 7, 7, 2, 3),
    u32x8::new(0, 1, 2, 7, 7, 7, 3, 4),
    u32x8::new(7, 7, 7, 0, 7, 7, 1, 2),
    u32x8::new(0, 7, 7, 1, 7, 7, 2, 3),
    u32x8::new(7, 0, 7, 1, 7, 7, 2, 3),
    u32x8::new(0, 1, 7, 2, 7, 7, 3, 4),
    u32x8::new(7, 7, 0, 1, 7, 7, 2, 3),
    u32x8::new(0, 7, 1, 2, 7, 7, 3, 4),
    u32x8::new(7, 0, 1, 2, 7, 7, 3, 4),
    u32x8::new(0, 1, 2, 3, 7, 7, 4, 5),
    u32x8::new(7, 7, 7, 7, 0, 7, 1, 2),
    u32x8::new(0, 7, 7, 7, 1, 7, 2, 3),
    u32x8::new(7, 0, 7, 7, 1, 7, 2, 3),
    u32x8::new(0, 1, 7, 7, 2, 7, 3, 4),
    u32x8::new(7, 7, 0, 7, 1, 7, 2, 3),
    u32x8::new(0, 7, 1, 7, 2, 7, 3, 4),
    u32x8::new(7, 0, 1, 7, 2, 7, 3, 4),
    u32x8::new(0, 1, 2, 7, 3, 7, 4, 5),
    u32x8::new(7, 7, 7, 0, 1, 7, 2, 3),
    u32x8::new(0, 7, 7, 1, 2, 7, 3, 4),
    u32x8::new(7, 0, 7, 1, 2, 7, 3, 4),
    u32x8::new(0, 1, 7, 2, 3, 7, 4, 5),
    u32x8::new(7, 7, 0, 1, 2, 7, 3, 4),
    u32x8::new(0, 7, 1, 2, 3, 7, 4, 5),
    u32x8::new(7, 0, 1, 2, 3, 7, 4, 5),
    u32x8::new(0, 1, 2, 3, 4, 7, 5, 6),
    u32x8::new(7, 7, 7, 7, 7, 0, 1, 2),
    u32x8::new(0, 7, 7, 7, 7, 1, 2, 3),
    u32x8::new(7, 0, 7, 7, 7, 1, 2, 3),
    u32x8::new(0, 1, 7, 7, 7, 2, 3, 4),
    u32x8::new(7, 7, 0, 7, 7, 1, 2, 3),
    u32x8::new(0, 7, 1, 7, 7, 2, 3, 4),
    u32x8::new(7, 0, 1, 7, 7, 2, 3, 4),
    u32x8::new(0, 1, 2, 7, 7, 3, 4, 5),
    u32x8::new(7, 7, 7, 0, 7, 1, 2, 3),
    u32x8::new(0, 7, 7, 1, 7, 2, 3, 4),
    u32x8::new(7, 0, 7, 1, 7, 2, 3, 4),
    u32x8::new(0, 1, 7, 2, 7, 3, 4, 5),
    u32x8::new(7, 7, 0, 1, 7, 2, 3, 4),
    u32x8::new(0, 7, 1, 2, 7, 3, 4, 5),
    u32x8::new(7, 0, 1, 2, 7, 3, 4, 5),
    u32x8::new(0, 1, 2, 3, 7, 4, 5, 6),
    u32x8::new(7, 7, 7, 7, 0, 1, 2, 3),
    u32x8::new(0, 7, 7, 7, 1, 2, 3, 4),
    u32x8::new(7, 0, 7, 7, 1, 2, 3, 4),
    u32x8::new(0, 1, 7, 7, 2, 3, 4, 5),
    u32x8::new(7, 7, 0, 7, 1, 2, 3, 4),
    u32x8::new(0, 7, 1, 7, 2, 3, 4, 5),
    u32x8::new(7, 0, 1, 7, 2, 3, 4, 5),
    u32x8::new(0, 1, 2, 7, 3, 4, 5, 6),
    u32x8::new(7, 7, 7, 0, 1, 2, 3, 4),
    u32x8::new(0, 7, 7, 1, 2, 3, 4, 5),
    u32x8::new(7, 0, 7, 1, 2, 3, 4, 5),
    u32x8::new(0, 1, 7, 2, 3, 4, 5, 6),
    u32x8::new(7, 7, 0, 1, 2, 3, 4, 5),
    u32x8::new(0, 7, 1, 2, 3, 4, 5, 6),
    u32x8::new(7, 0, 1, 2, 3, 4, 5, 6),
    u32x8::new(0, 1, 2, 3, 4, 5, 6, 7),
];

// mask for SIMD gather/pointer reading based on number of nonzeroes in group of 8.
// Only read from memory for which values are guaranteed to exist.
const U32_SIMD_READMASKS: [m32x8; 9] = [
    m32x8::splat(false),
    m32x8::new(true, false, false, false, false, false, false, false),
    m32x8::new(true, true, false, false, false, false, false, false),
    m32x8::new(true, true, true, false, false, false, false, false),
    m32x8::new(true, true, true, true, false, false, false, false),
    m32x8::new(true, true, true, true, true, false, false, false),
    m32x8::new(true, true, true, true, true, true, false, false),
    m32x8::new(true, true, true, true, true, true, true, false),
    m32x8::new(true, true, true, true, true, true, true, true),
];

// Used for when we aren't sure there's enough space to use preload_simd
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

#[inline]
fn preload_u32x8_nibbles(buf: &[u8],
                         num_nibbles: usize,
                         nonzeroes: u32) -> Result<(u32x8, u32), CodingError> {
    let total_bytes = (num_nibbles * nonzeroes as usize + 1) / 2;
    let mut i = 0;
    let mut off = 2;
    let simd_word = u32x8::splat(0);
    while i < 8 && off < (total_bytes + 2) {
        let inword = direct_read_uint_le(buf, off)?;
        // Safe because we are checking boundaries in while loop conditions
        unsafe { simd_word.replace_unchecked(i, inword as u32) };
        let shift2 = (num_nibbles * 4) / 8 * 8;  // round off shift to lower byte boundary
        unsafe { simd_word.replace_unchecked(i + 1, (inword >> shift2) as u32) };
        i += 2;
        off += num_nibbles;
    }
    Ok((simd_word, total_bytes as u32))
}

/// SIMD GATHER/cptr based loading of SIMD u32x8 register, fast for 3+ nibbles
/// Can be used to load from any number of nibbles for u32
// TODO: only enable this for x86* and architectures with safe unaligned reads?
#[inline(always)]
unsafe fn preload_u32x8_simd(buf: &[u8],
                             num_nibbles: u8,
                             nonzeroes: u32) -> u32x8 {
    // Get pointer to beginning of buf encoded bytes section.  This is safe due to length check above
    let first_byte = buf.as_ptr().offset(2);
    let u8_ptrs = cptrx8::splat(first_byte);

    // Add variable offsets so we read from right parts of buffer for each word
    let u8_offset = u8_ptrs.offset(U32_SIMD_PTR_OFFSETS[num_nibbles as usize]);
    // Change type from *u8 to *u32 and force unaligned reads
    let u32_offsets: cptrx8<u32> = std::mem::transmute(u8_offset);

    // Read with mask
    let loaded: u32x8 = u32_offsets.read(U32_SIMD_READMASKS[nonzeroes as usize], ZEROES_U32X8);
    // Ensure little endian.  This should be NOP on x86 and other LE architectures
    loaded.to_le()
}

// Optimized shuffle using AVX2 instruction, which is not available in packed_simd for some reason ??
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),
          target_feature = "avx2"))]
#[inline(always)]
fn unpack_shuffle(input: u32x8, nonzero_mask: u8) -> u32x8 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::_mm256_permutevar8x32_epi32;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::_mm256_permutevar8x32_epi32;

    let shifted1 = input.replace(7, 0);  // Stuff 0 into unused final slot
    unsafe {
        std::mem::transmute(
            _mm256_permutevar8x32_epi32(
                std::mem::transmute(shifted1),
                std::mem::transmute(SHUFFLE_UNPACK_IDX_U32[nonzero_mask as usize])
            )
        )
    }
}

// Unoptimized using packed_simd which doesn't support above instruction
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"),
          target_feature = "avx2")))]
#[inline(always)]
fn unpack_shuffle(input: u32x8, nonzero_mask: u8) -> u32x8 {
    let shifted1 = input.replace(7, 0);  // Stuff 0 into unused final slot
    shifted1.shuffle1_dyn(SHUFFLE_UNPACK_IDX_U32[nonzero_mask as usize])
}

// Max number of bytes that a U32 nibblepacked 8 inputs could take up: 2 + 8*4;
pub const MAX_U32_NIBBLEPACKED_LEN: usize = 34;

/// SIMD-based decoding of NibblePacked data to u32x8.  Errors out if number of nibbles exceeds 8.
/// Checks that the input buffer has enough room to decode.
/// Really fast for 1-2 nibbles, but still fast for 3-8 nibbles.
#[inline]
pub fn unpack8_u32_simd<'a, Output: Sink<u32x8>>(
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
        let trailing_zeros = (inbuf[1] & 0x0f) * 4;

        // First step: load encoded bytes in parallel to SIMD registers
        // Also figure out how many bytes are taken up by packed nibbles
        let (simd_inputs, num_bytes) = match num_nibbles {
            // NOTE: the code for 1/2 nibbles is faster than preload_simd, but for 3+ nibbles preload_simd is faster
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
            3..=8 => {
                if inbuf.len() >= MAX_U32_NIBBLEPACKED_LEN {
                    let total_bytes = (num_nibbles as usize * nonzero_count as usize + 1) / 2;
                    // Call below is safe since we have checked length above
                    (unsafe { preload_u32x8_simd(inbuf, num_nibbles, nonzero_count) }, total_bytes as u32)
                } else if num_nibbles <= 4 {
                    preload_u32x8_3_4_nibble(inbuf, num_nibbles as usize, nonzero_count)?
                } else {
                    preload_u32x8_nibbles(inbuf, num_nibbles as usize, nonzero_count)?
                }
            },
            _ => return Err(CodingError::InvalidFormat(
                            format!("{:?} nibbles is too many for u32 decoder", num_nibbles))),
        };

        let shuffled = simd_unpack_inner(simd_inputs, num_nibbles, trailing_zeros,
                                         nonzero_count, nonzero_mask);

        // Step 6. Send to sink, and advance input slice
        output.process(shuffled);
        Ok(&inbuf[(2 + num_bytes as usize)..])
    }
}

// Inner SIMD decoding steps, produces a final shuffled 8 u32's
#[inline(always)]
fn simd_unpack_inner(simd_inputs: u32x8, num_nibbles: u8, trailing_zeros: u8,
                     nonzero_count: u32,
                     nonzero_mask: u8) -> u32x8 {
    // Step 2. Variable right shift to shift each set of nibbles in right place
    let shifted = simd_inputs.shr(U32_SIMD_SHIFTS[num_nibbles as usize]);

    // Step 3. AND mask to strip upper bits, so each lane left with its own value
    let anded = shifted.bitand(U32_SIMD_ANDMASK[num_nibbles as usize]);

    // Step 4. Left shift for trailing zeroes, if needed
    let leftshifted = if (trailing_zeros == 0) { anded } else { anded.shl(trailing_zeros as u32) };

    // Step 5. Shuffle inputs based on nonzero mask to proper places
    if (nonzero_count == 8) { leftshifted } else { unpack_shuffle(leftshifted, nonzero_mask) }
}


#[test]
fn test_unpack_u32simd_1_2nibbles() {
    let mut buf = [55u8; 512];

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

#[test]
fn test_unpack_u32simd_3_4nibbles() {
    // Tests edge case where 4 nibbles (16 bits) pack edge
    // 4 nibbles = 2^16, so values < 65536
    let inputs = [65535u64; 8];
    let mut buf = [0u8; 512];
    let written = nibble_pack8(&inputs, &mut buf, 0).unwrap();

    let mut sink = U32_256Sink::new();
    let _rest = unpack8_u32_simd(&buf[..written], &mut sink).unwrap();

    assert_eq!(sink.values[..8], [65535u32; 8]);

    // case 2 - first 8 use 3 nibbles, and then 4 nibbles.
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
                          (nbits in 4usize..30, chance in 0.1f32..0.6)
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
