extern crate byteorder;

use self::byteorder::{LittleEndian, ReadBytesExt};

// TODO: change these to be Vec implementations?  or Writer?

/// Function to write a u64 to memory quickly using unaligned writes.  The Vec state/len is updated & capacity checked.
/// Equivalent of sun.misc.Unsafe, but it checks Vec has enough space so in theory it should be safe
/// It is 2-3x faster than the equivalent code from byteorder, which uses memcpy instead.
/// TODO: write a method which works on multiple 64-bit inputs or partial inputs so the pointer state, reservation etc
///       can be amortized and the below can be a cheaper write.
#[inline]
pub fn direct_write_uint_le(out_buffer: &mut Vec<u8>, value: u64, numbytes: usize) {
    out_buffer.reserve(8);
    unsafe {
        // We have checked the capacity so this is OK
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

/// Safe but fast read from inbuf.  If it can read 64 bits then uses fast unaligned read, otherwise
/// uses byteorder crate.  Also does Endianness conversion.
#[inline(always)]
pub fn direct_read_uint_le(inbuf: &[u8], index: u32) -> u64 {
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

/// Writes a u64 directly to a vec<u64>, without checking for or reserving more space.
///
/// This method is unsafe as it does not check that the buf has enough space.
/// It is expected for the user to check this themselves via reserve() earlier on.
#[inline]
pub unsafe fn unchecked_write_u64_u64_le(out_buffer: &mut Vec<u64>, value: u64) {
    let cur_len = out_buffer.len();
    let ptr = out_buffer.as_mut_ptr().offset(cur_len as isize);
    std::ptr::write(ptr, value.to_le());
    out_buffer.set_len(cur_len + 1);
}

/// Writes eight u64 values in rapid succession to the Vec, directly, using unsafe,
/// for performance reasons / to avoid checks on every single write.
/// It is safe because of checks done on the overall write.
#[inline]
pub fn write8_u64_le(out_buffer: &mut Vec<u64>, value: u64) {
    out_buffer.reserve(8);
    unsafe {
        let cur_len = out_buffer.len();
        let ptr = out_buffer.as_mut_ptr().offset(cur_len as isize);
        for i in 0..8 {
            std::ptr::write(ptr.add(i), value.to_le());
        }
        out_buffer.set_len(cur_len + 8);
    }
}