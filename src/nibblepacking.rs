#![allow(unused)]  // needed for dbg!() macro, but folks say this should not be needed

extern crate byteorder;

use self::byteorder::{WriteBytesExt, LittleEndian};


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
/// * `out_buffer` - a vec to write the encoded output to
///
/// # Return Value
/// The number of bytes encoded, or an encoding error
///
pub fn nibble_pack8(inputs: &[u64], out_buffer: &mut Vec<u8>) -> Result<u16, NibblePackError> {
    if inputs.len() < 8 { return Err(NibblePackError::InputTooShort); }

    // reset the out_buffer...
    out_buffer.clear();

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
        let mut min_leading_zeros = 64u32;
        let mut min_trailing_zeros = 64u32;
        for i in 0..8 {
            min_leading_zeros = u32::min(min_leading_zeros, inputs[i].leading_zeros());
            min_trailing_zeros = u32::min(min_trailing_zeros, inputs[i].trailing_zeros());
        }

        // Convert min leading/trailing to # nibbles.  Start packing!
        let trailing_nibbles = min_trailing_zeros / 4;
        let num_nibbles = 16 - (min_leading_zeros / 4) - trailing_nibbles;
        let nibble_word = ((num_nibbles << 4) | trailing_nibbles) as u8;
        out_buffer.push(nibble_word);

        if (num_nibbles % 2) == 0 {
            pack_to_even_nibbles(inputs, out_buffer, num_nibbles, trailing_nibbles);
        } else if num_nibbles < 8 { // TODO: distinguish larger odd nibbles
            pack_to_odd_nibbles1(inputs, out_buffer, num_nibbles, trailing_nibbles);
        } else {
            pack_to_odd_nibbles2(inputs, out_buffer, num_nibbles, trailing_nibbles);
        }
    }

    // TODO: fix this or truncate to actual length.  Buffer length might have pad bytes at the end!
    Ok(out_buffer.len() as u16)
}

///
/// Inner function to pack the raw inputs to nibbles when # nibbles is even (always # bytes)
///
/// # Arguments
/// * `trailing_zero_nibbles` - the min # of trailing zero nibbles across all inputs
/// * `num_nibbles` - the max # of nibbles having nonzero bits in all inputs
fn pack_to_even_nibbles(inputs: &[u64],
                        out_buffer: &mut Vec<u8>,
                        num_nibbles: u32,
                        trailing_zero_nibbles: u32) {
    // In the future, explore these optimizations: functions just for specific nibble widths
    let shift = trailing_zero_nibbles * 4;
    assert!(num_nibbles % 2 == 0);
    let num_bytes_each = (num_nibbles / 2) as usize;

    // for each nonzero input, shift and write out exact # of bytes
    for i in 0..8 {
        if inputs[i] != 0 {
            out_buffer.write_uint::<LittleEndian>(inputs[i] >> shift, num_bytes_each).unwrap();
        }
    }
}

/// Pack raw inputs when # nibbles is odd but < 8.  Lets us pack two inputs together.
/// TODO: maybe optimize search for next nonzero input these ways:
///  - pass in the bitmask, then shift and find next nonzero bit
///  - or just use an iterator.  Iterators in Rust are zero-cost abstractions.  Use higher-level functions!!
///  - use slice.chunks(2) to return.  Can also use itertools.chunks() which gives you chunks method on iterators
/// LOVE: how Rust extends existing tools like Vec Write API (byteorder), or itertools extends iter with chunks etc
///       as well as zero-cost abstractions
fn pack_to_odd_nibbles1(inputs: &[u64],
                        out_buffer: &mut Vec<u8>,
                        num_nibbles: u32,
                        trailing_zero_nibbles: u32) {
    let shift = trailing_zero_nibbles * 4;
    assert!(num_nibbles % 2 == 1);
    let mut i = 0;
    while i < 8 {
        if inputs[i] == 0 {
            i += 1;
            continue;
        }

        // if nonzero, shift first value into place
        let mut packedword = inputs[i] >> shift;
        let mut bytespacked = (num_nibbles + 1) / 2;

        // find second value, shift into upper place
        i += 1;
        while i < 8 {
            if inputs[i] == 0 {
                i += 1;
                continue;
            }
            packedword |= (inputs[i] >> shift) << (num_nibbles * 4);
            bytespacked = num_nibbles;
            break;
        }

        // write out both values together
        out_buffer.write_uint::<LittleEndian>(packedword, bytespacked as usize).unwrap();
        i += 1;
    }
}

/// Pack raw inputs when # nibbles is odd but >= 8.
fn pack_to_odd_nibbles2(inputs: &[u64],
                        out_buffer: &mut Vec<u8>,
                        num_nibbles: u32,
                        trailing_zero_nibbles: u32) {
    let shift = trailing_zero_nibbles * 4;
    let word2_lshift = (num_nibbles * 4) - shift;
    let word2_rshift = 64 - (num_nibbles * 4) + shift;
    assert!(num_nibbles % 2 == 1);

    let mut input_it = inputs.into_iter().filter({ |&&x| x != 0u64 });
    loop {
        match (input_it.next(), input_it.next()) {
            (Some(a), Some(b)) => {   // Pack first and second words together
                // mix first word + LSBits of second word together in 64 bits and write it out
                // (because byteorder doesn't let us write out less than width of data)
                let packedword = (a >> shift) | (b << word2_lshift);
                out_buffer.write_u64::<LittleEndian>(packedword).unwrap();

                // Now shift upper bits of second word and write those out only
                let packedword = b >> word2_rshift;
                out_buffer.write_uint::<LittleEndian>(packedword, (num_nibbles - 8) as usize).unwrap();
            },
            (Some(a), None)    => {   // Only need to pack the first word as its the last
                let numbytes = (num_nibbles + 1) / 2;
                out_buffer.write_uint::<LittleEndian>(a >> shift, numbytes as usize).unwrap();
                break;
            },
            (None,    None)    => break,
            (None,    Some(_)) => break,
        }
    }
}

#[test]
fn nibblepack8_all_zeroes() {
    let mut buf = Vec::with_capacity(1024);
    let inputs = [0u64; 8];
    let res = nibble_pack8(&inputs, &mut buf);
    assert_eq!(res, Ok(1));
    assert_eq!(buf[..], [0u8]);
}

#[test]
fn nibblepack8_all_evennibbles() {
    // All 8 are nonzero, even # nibbles
    let mut buf = Vec::with_capacity(1024);
    let inputs = [0x0000_00fe_dcba_0000u64, 0x0000_0033_2211_0000u64,
                  0x0000_0044_3322_0000u64, 0x0000_0055_4433_0000u64,
                  0x0000_0066_5544_0000u64, 0x0000_0076_5432_0000u64,
                  0x0000_0087_6543_0000u64, 0x0000_0098_7654_0000u64,];
    let res = nibble_pack8(&inputs, &mut buf);

    // Expected result:
    let expected_buf = [0xffu8, // Every input is nonzero, all bits on
                        0x64u8, // six nibbles wide, four zero nibbles trailing
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
                        0x64u8, // six nibbles wide, four zero nibbles trailing
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
                        0x55u8, // five nibbles wide, five zero nibbles trailing
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
                        0x94u8, // nine nibbles wide, four zero nibbles trailing
                        0x11u8, 0x22u8, 0x33u8, 0x44u8, 0x05u8, 0x32u8, 0x43u8, 0x04u8, 0,
                        0x33u8, 0x44u8, 0x55u8, 0x66u8, 0x07u8, 0x54u8, 0x65u8, 0x06u8, 0,
                        0x30u8, 0x54u8, 0x76u8, 0x98u8, 0x01u8];
    assert_eq!(res, Ok(expected_buf.len() as u16));
    assert_eq!(buf[..], expected_buf);
}

// TODO: look into Rust equivalent of Quickcheck for generating ... maybe when we have two-way encoding/decoding