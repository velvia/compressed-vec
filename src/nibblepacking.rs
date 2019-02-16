#![allow(unused)]  // needed for dbg!() macro, but folks say this should not be needed

extern crate byteorder;

use self::byteorder::{ReadBytesExt, LittleEndian};

#[derive(Debug, PartialEq)]
pub enum NibblePackError {
    InputTooShort,
}

///
/// NibblePacking is an encoding technique for packing 8 u64's tightly into the same number of nibbles.
/// It can be combined with a prediction algorithm to efficiency encode floats and long values.
/// Please see http://github.com/filodb/FiloDB/doc/compression.md for more answers.
///
/// # Arguments
/// * `inputs` - ref to 8 u64 values to pack, could be the output of a predictor
/// * `out_buffer` - a vec to write the encoded output to.  Bytes are added at the end - vec is not cleared.
///
/// # Return Value
/// The number of bytes encoded, or an encoding error
///
pub fn nibble_pack8(inputs: &[u64; 8], out_buffer: &mut Vec<u8>) -> Result<u16, NibblePackError> {
    if inputs.len() < 8 { return Err(NibblePackError::InputTooShort); }

    // Compute the nonzero bitmask.  TODO: use SIMD here
    let mut nonzero_mask = 0u8;
    for i in 0..8 {
        if inputs[i] != 0 {
            nonzero_mask |= 1 << i;
        }
    }
    out_buffer.push(nonzero_mask);

    // if no nonzero values, we're done!
    if nonzero_mask != 0 {  // TODO: use SIMD here
        // otherwise, get min of leading and trailing zeros, encode it
        // NOTE: code below is actually slower than the iterator!  Fearless use of abstractions FTW!!!
        // let mut min_leading_zeros = 64u32;
        // let mut min_trailing_zeros = 64u32;
        // for i in 0..8 {
        //     min_leading_zeros = u32::min(min_leading_zeros, inputs[i].leading_zeros());
        //     min_trailing_zeros = u32::min(min_trailing_zeros, inputs[i].trailing_zeros());
        // }
        let min_leading_zeros = inputs.into_iter().map(|x| x.leading_zeros()).min().unwrap();
        let min_trailing_zeros = inputs.into_iter().map(|x| x.trailing_zeros()).min().unwrap();

        // Convert min leading/trailing to # nibbles.  Start packing!
        // NOTE: num_nibbles cannot be 0; that would imply every input was zero
        let trailing_nibbles = min_trailing_zeros / 4;
        let num_nibbles = 16 - (min_leading_zeros / 4) - trailing_nibbles;
        let nibble_word = (((num_nibbles - 1) << 4) | trailing_nibbles) as u8;
        out_buffer.push(nibble_word);

        if (num_nibbles % 2) == 0 {
            pack_to_even_nibbles(inputs, out_buffer, num_nibbles, trailing_nibbles);
        } else {
            pack_universal(inputs, out_buffer, num_nibbles, trailing_nibbles);
        }
    }

    // TODO: fix this or truncate to actual length.  Buffer length might have pad bytes at the end!
    Ok(out_buffer.len() as u16)
}

///
/// Inner function to pack the raw inputs to nibbles when # nibbles is even (always # bytes)
/// It's somehow really fast, perhaps because it is really simple.
///
/// # Arguments
/// * `trailing_zero_nibbles` - the min # of trailing zero nibbles across all inputs
/// * `num_nibbles` - the max # of nibbles having nonzero bits in all inputs
#[inline]
fn pack_to_even_nibbles(inputs: &[u64; 8],
                        out_buffer: &mut Vec<u8>,
                        num_nibbles: u32,
                        trailing_zero_nibbles: u32) {
    // In the future, explore these optimizations: functions just for specific nibble widths
    let shift = trailing_zero_nibbles * 4;
    assert!(num_nibbles % 2 == 0);
    let num_bytes_each = (num_nibbles / 2) as usize;

    // for each nonzero input, shift and write out exact # of bytes
    inputs.into_iter().for_each(|&x| if (x != 0) {
        direct_write_uint_le(out_buffer, x >> shift, num_bytes_each)
    });
}

/// This code is inspired by bitpacking crate: https://github.com/tantivy-search/bitpacking/
/// but modified for the NibblePacking algorithm.  No macros, so slightly less efficient.
/// TODO: consider using macros like in bitpacking to achieve even more speed :D
#[inline]
fn pack_universal(inputs: &[u64; 8],
                  out_buffer: &mut Vec<u8>,
                  num_nibbles: u32,
                  trailing_zero_nibbles: u32) {
    let trailing_shift = trailing_zero_nibbles * 4;
    let num_bits = num_nibbles * 4;
    let mut out_word = 0u64;
    let mut bit_cursor = 0;

    inputs.into_iter().for_each(|&x| if (x != 0) {
        let remaining = 64 - bit_cursor;
        let shifted_input = x >> trailing_shift;

        // This is least significant portion of input
        out_word |= shifted_input << bit_cursor;

        // Write out current word if we've used up all 64 bits
        if remaining <= num_bits {
            direct_write_uint_le(out_buffer, out_word, 8);  // Replace with faster write?
            if remaining < num_bits {
                // Most significant portion left over from previous word
                out_word = shifted_input >> (remaining as i32);
            } else {
                out_word = 0;   // reset for 64-bit input case
            }
        }

        bit_cursor = (bit_cursor + num_bits) % 64;
    });

    // Write remainder word if there are any bits remaining, and only advance buffer right # of bytes
    if bit_cursor > 0 {
        direct_write_uint_le(out_buffer, out_word, ((bit_cursor + 7) / 8) as usize);
    }
}

/// Function to write a u64 to memory quickly using unaligned writes.  The Vec state/len is updated & capacity checked.
/// Equivalent of sun.misc.Unsafe, but it checks Vec has enough space so in theory it should be safe
/// It is 2-3x faster than the equivalent code from byteorder, which uses memcpy instead.
/// TODO: write a method which works on multiple 64-bit inputs or partial inputs so the pointer state, reservation etc
///       can be amortized and the below can be a cheaper write.
#[inline]
fn direct_write_uint_le(out_buffer: &mut Vec<u8>, value: u64, numbytes: usize) {
    out_buffer.reserve(8);
    unsafe {   // We have checked the capacity so this is OK
        unsafe_write_uint_le(out_buffer, value, numbytes);
    }
}

#[inline(always)]
unsafe fn unsafe_write_uint_le(out_buffer: &mut Vec<u8>, value: u64, numbytes: usize) {
    let cur_len = out_buffer.len();
    let ptr = out_buffer.as_mut_ptr().offset(cur_len as isize) as *mut u64;
    std::ptr::write_unaligned(ptr, value.to_le());
    out_buffer.set_len(cur_len + numbytes);
}

///
/// A trait for processing data during unpacking.  Used to combine with predictors to create final output.
pub trait Sink {
    fn process(&mut self, data: u64);
}

pub struct LongSink {
    vec: Vec<u64>
}

const DEFAULT_CAPACITY: usize = 64;

impl LongSink {
    fn new() -> LongSink {
        LongSink { vec: Vec::with_capacity(DEFAULT_CAPACITY) }
    }
}

impl Sink for LongSink {
    fn process(&mut self, data: u64) {
        self.vec.push(data);
    }
}

/// Unpacks 8 u64's packed using nibble_pack8.
/// Returns "remainder" byteslice or unpacking error (say if one ran out of space)
///
/// # Arguments
/// * `inbuf` - NibblePacked compressed byte slice containing "remaining" bytes, starting with bitmask byte
/// * `output` - a Trait which processes each resulting u64
// NOTE: The 'a is a lifetime annotation.  When you use two references in Rust, and return one, Rust needs
//       annotations to help it determine to which input the output lifetime is related to, so Rust knows
//       that the output of the slice will live as long as the reference to the input slice is valid.
#[inline]
fn nibble_unpack8<'a, Output: Sink>(inbuf: &'a [u8], output: &mut Output) -> Result<&'a [u8], NibblePackError> {
    let nonzero_mask = inbuf[0];
    if nonzero_mask == 0 {
        // All 8 words are 0; skip further processing
        for _ in 0..8 {
            output.process(0);
        }
        Ok(&inbuf[1..])
    } else {
        let num_bits = ((inbuf[1] >> 4) + 1) * 4;
        let trailing_zeros = (inbuf[1] & 0x0f) * 4;
        let total_bytes = 2 + (num_bits as u32 * nonzero_mask.count_ones() + 7) / 8;
        let mask: u64 = if num_bits >= 64 {
                            std::u64::MAX
                        } else {
                            (1u64 << num_bits) - 1u64
                        };
        let mut buf_index = 2;
        let mut bit_cursor = 0;

        // Read in first word
        let mut in_word = direct_read_uint_le(inbuf, buf_index);
        buf_index += 8;

        for bit in 0..8 {
            if (nonzero_mask & (1 << bit)) != 0 {
                let remaining = 64 - bit_cursor;

                // Shift and read in LSB (or entire nibbles if they fit)
                let shifted_in = in_word >> bit_cursor;
                let mut out_word = shifted_in & mask;

                // If remaining bits are in next word, read next word -- if there's space
                // We don't want to read the next word though if we're already at the end
                if remaining <= num_bits && buf_index < total_bytes {
                    if ((buf_index as usize) < inbuf.len()) {
                        // Read in MSB bits from next wordå
                        in_word = direct_read_uint_le(inbuf, buf_index);
                        buf_index += 8;   // figure out right amount!
                        if remaining < num_bits {
                            let shifted = in_word << remaining;
                            out_word |= shifted & mask;
                        }
                    } else {
                        return Err(NibblePackError::InputTooShort);
                    }
                }

                output.process(out_word << trailing_zeros);

                // Update other indices
                bit_cursor = (bit_cursor + num_bits) % 64;
            } else {
                output.process(0);
            }
        }

        // Return the "remaining slice" - the rest of input buffer after we've parsed our bytes.
        // This allows for easy and clean chaining of nibble_unpack8 calls with no mutable state
        Ok(&inbuf[(total_bytes as usize)..])
    }
}

/// Safe but fast read from inbuf.  If it can read 64 bits then uses fast unaligned read, otherwise
/// uses byteorder crate.  Also does Endianness conversion.
fn direct_read_uint_le(inbuf: &[u8], index: u32) -> u64 {
    if ((index as usize) + 8) <= inbuf.len() {
        unsafe {
            let ptr = inbuf.as_ptr().offset(index as isize) as *const u64;
            u64::from_le(std::ptr::read_unaligned(ptr))
        }
    } else {
        // Less than 8 bytes left.  Use Byteorder implementation which can read limited # of bytes.
        // This ensures we don't read from a space we are not allowed to.
        let mut cursor = std::io::Cursor::new(inbuf);
        cursor.set_position(index as u64);
        cursor.read_uint::<LittleEndian>(inbuf.len() - index as usize).unwrap()
    }
}

#[test]
fn nibblepack8_all_zeroes() {
    let mut buf = Vec::with_capacity(512);
    let inputs = [0u64; 8];
    let res = nibble_pack8(&inputs, &mut buf);
    assert_eq!(res, Ok(1));
    assert_eq!(buf[..], [0u8]);
}

#[test]
fn nibblepack8_all_evennibbles() {
    // All 8 are nonzero, even # nibbles
    let mut buf = Vec::with_capacity(512);
    let inputs = [0x0000_00fe_dcba_0000u64, 0x0000_0033_2211_0000u64,
                  0x0000_0044_3322_0000u64, 0x0000_0055_4433_0000u64,
                  0x0000_0066_5544_0000u64, 0x0000_0076_5432_0000u64,
                  0x0000_0087_6543_0000u64, 0x0000_0098_7654_0000u64,];
    let res = nibble_pack8(&inputs, &mut buf);

    // Expected result:
    let expected_buf = [0xffu8, // Every input is nonzero, all bits on
                        0x54u8, // six nibbles wide, four zero nibbles trailing
                        0xbau8, 0xdcu8, 0xfeu8, 0x11u8, 0x22u8, 0x33u8,
                        0x22u8, 0x33u8, 0x44u8, 0x33u8, 0x44u8, 0x55u8,
                        0x44u8, 0x55u8, 0x66u8, 0x32u8, 0x54u8, 0x76u8,
                        0x43u8, 0x65u8, 0x87u8, 0x54u8, 0x76u8, 0x98u8,
                        ];
    assert_eq!(res, Ok(2 + 3*8));
    assert_eq!(buf[..], expected_buf);
}

// Even nibbles with different combos of partial
#[test]
fn nibblepack8_partial_evennibbles() {
    // All 8 are nonzero, even # nibbles
    let mut buf = Vec::with_capacity(1024);
    let inputs = [0u64, 0x0000_0033_2211_0000u64,
                  0x0000_0044_3322_0000u64, 0x0000_0055_4433_0000u64,
                  0x0000_0066_5544_0000u64, 0u64,
                  0u64, 0u64,];
    let res = nibble_pack8(&inputs, &mut buf);

    // Expected result:
    let expected_buf = [0b0001_1110u8, // only some bits on
                        0x54u8, // six nibbles wide, four zero nibbles trailing
                        0x11u8, 0x22u8, 0x33u8,
                        0x22u8, 0x33u8, 0x44u8, 0x33u8, 0x44u8, 0x55u8,
                        0x44u8, 0x55u8, 0x66u8, ];
    assert_eq!(res, Ok(2 + 3*4));
    assert_eq!(buf[..], expected_buf);
}

// Odd nibbles with different combos of partial
#[test]
fn nibblepack8_partial_oddnibbles() {
    // All 8 are nonzero, even # nibbles
    let mut buf = Vec::with_capacity(1024);
    let inputs = [0u64, 0x0000_0033_2210_0000u64,
                  0x0000_0044_3320_0000u64, 0x0000_0055_4430_0000u64,
                  0x0000_0066_5540_0000u64, 0x0000_0076_5430_0000u64,
                  0u64, 0u64,];
    let res = nibble_pack8(&inputs, &mut buf);

    // Expected result:
    let expected_buf = [0b0011_1110u8, // only some bits on
                        0x45u8, // five nibbles wide, five zero nibbles trailing
                        0x21u8, 0x32u8, 0x23u8, 0x33u8, 0x44u8,  // First two values
                        0x43u8, 0x54u8, 0x45u8, 0x55u8, 0x66u8,
                        0x43u8, 0x65u8, 0x07u8, ];
    assert_eq!(res, Ok(expected_buf.len() as u16));
    assert_eq!(buf[..], expected_buf);
}

// Odd nibbles > 8 nibbles
#[test]
fn nibblepack8_partial_oddnibbles_large() {
    // All 8 are nonzero, even # nibbles
    let mut buf = Vec::with_capacity(1024);
    let inputs = [0u64, 0x0005_4433_2211_0000u64,
                  0x0000_0044_3320_0000u64, 0x0007_6655_4433_0000u64,
                  0x0000_0066_5540_0000u64, 0x0001_9876_5430_0000u64,
                  0u64, 0u64,];
    let res = nibble_pack8(&inputs, &mut buf);

    // Expected result:
    let expected_buf = [0b0011_1110u8, // only some bits on
                        0x84u8, // nine nibbles wide, four zero nibbles trailing
                        0x11u8, 0x22u8, 0x33u8, 0x44u8, 0x05u8, 0x32u8, 0x43u8, 0x04u8, 0,
                        0x33u8, 0x44u8, 0x55u8, 0x66u8, 0x07u8, 0x54u8, 0x65u8, 0x06u8, 0,
                        0x30u8, 0x54u8, 0x76u8, 0x98u8, 0x01u8];
    assert_eq!(res, Ok(expected_buf.len() as u16));
    assert_eq!(buf[..], expected_buf);
}

#[test]
fn nibblepack8_64bit_numbers() {
    let mut buf = Vec::with_capacity(1024);
    let inputs = [0, 0, -1i32 as u64, -2i32 as u64, 0, -100234i32 as u64, 0, 0];
    let res = nibble_pack8(&inputs, &mut buf);

    let expected_buf = [0b0010_1100u8,
                        0xf0u8,   // all 16 nibbles wide, zero nibbles trailing
                        0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8,
                        0xfeu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8,
                        0x76u8, 0x78u8, 0xfeu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8, 0xffu8 ];
    assert_eq!(res, Ok(expected_buf.len() as u16));
    assert_eq!(buf[..], expected_buf);
}

#[test]
fn unpack8_all_zeroes() {
    let compressed_array = [0x00u8];
    let mut sink = LongSink::new();
    let res = nibble_unpack8(&compressed_array, &mut sink);
    assert_eq!(res.unwrap().is_empty(), true);
    assert_eq!(sink.vec.len(), 8);
    assert_eq!(sink.vec[..], [0u64; 8]);
}

#[test]
fn unpack8_input_too_short() {
    let compressed = [0b0011_1110u8, // only some bits on
                      0x84u8, // nine nibbles wide, four zero nibbles trailing
                      0x11u8, 0x22u8, 0x33u8, 0x44u8, 0x05u8, 0x32u8, 0x43u8, 0x04u8, 0,
                      0x33u8, 0x44u8, 0x55u8, 0x66u8, 0x07u8 ];  // too short!!
    let mut sink = LongSink::new();
    let res = nibble_unpack8(&compressed, &mut sink);
    assert_eq!(res, Err(NibblePackError::InputTooShort));
}

// Tests the case where nibbles lines up with 64-bit boundaries - edge case
#[test]
fn unpack8_4nibbles_allfull() {
    // 4 nibbles = 2^16, so values < 65536
    let inputs = [65535u64; 8];
    let mut buf = Vec::with_capacity(1024);
    nibble_pack8(&inputs, &mut buf).expect("Should have encoded OK");

    let mut sink = LongSink::new();
    let res = nibble_unpack8(&buf[..], &mut sink);
    assert_eq!(res.unwrap().len(), 0);
    assert_eq!(sink.vec[..], inputs);
}

#[test]
fn unpack8_partial_oddnibbles() {
    let compressed = [0b0011_1110u8, // only some bits on
                      0x84u8, // nine nibbles wide, four zero nibbles trailing
                      0x11u8, 0x22u8, 0x33u8, 0x44u8, 0x05u8, 0x32u8, 0x43u8, 0x04u8, 0,
                      0x33u8, 0x44u8, 0x55u8, 0x66u8, 0x07u8, 0x54u8, 0x65u8, 0x06u8, 0,
                      0x30u8, 0x54u8, 0x76u8, 0x98u8, 0x01u8,
                      0x00u8 ];  // extra padding... just to test the return value
    let mut sink = LongSink::new();
    let res = nibble_unpack8(&compressed, &mut sink);
    assert_eq!(res.unwrap().len(), 1);
    assert_eq!(sink.vec.len(), 8);

    let orig = [0u64, 0x0005_4433_2211_0000u64,
                0x0000_0044_3320_0000u64, 0x0007_6655_4433_0000u64,
                0x0000_0066_5540_0000u64, 0x0001_9876_5430_0000u64,
                0u64, 0u64,];

    assert_eq!(sink.vec[..], orig);
}

// NOTE: cfg(test) is needed so that proptest can just be a "dev-dependency" and not linked for final library
// NOTE2: somehow cargo is happier when we put props tests in its own module
#[cfg(test)]
mod props {
    extern crate proptest;

    use super::*;
    use self::proptest::prelude::*;

    // Generators (Arb's) for numbers of given # bits with fractional chance of being zero.
    // Also input arrays of 8 with the given properties above.
    prop_compose! {
        /// zero_chance: 0..1.0 chance of obtaining a zero
        fn arb_maybezero_nbits_u64(nbits: usize, zero_chance: f32)
                                  (n in 0u64..(1 << nbits), chance in 0.0..(1.0/zero_chance)) -> u64 {
            if chance <= 1.0 { 0 } else { n }
        }
    }

    // Try different # bits and # nonzero elements
    prop_compose! {
        fn arb_8longs_nbits()
                           (nbits in 4usize..64, chance in 0.2f32..0.8)
                           (input in prop::array::uniform8(arb_maybezero_nbits_u64(nbits, chance))) -> [u64; 8] {
                               input
                           }
    }

    proptest! {
        #[test]
        fn prop_pack_unpack_identity(input in arb_8longs_nbits()) {
            let mut buf = Vec::with_capacity(256);
            nibble_pack8(&input, &mut buf).expect("Should have encoded OK");

            let mut sink = LongSink::new();
            let res = nibble_unpack8(&buf[..], &mut sink);
            assert_eq!(sink.vec[..], input);
        }
    }
}
