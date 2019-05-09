extern crate byteorder;

use self::byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

/// Fast write of u64.  numbytes least significant bytes are written.  May overwrite, but it's fine
/// because this is a Vec which can be extended.  Vec.len() is always advanced numbytes.
#[inline]
pub fn direct_write_uint_le(out_buffer: &mut Vec<u8>, value: u64, numbytes: usize) {
    if numbytes == 8 {
        out_buffer.write_u64::<LittleEndian>(value).unwrap();
    } else {
        let orig_len = out_buffer.len();
        out_buffer.write_u64::<LittleEndian>(value).unwrap();
        out_buffer.truncate(orig_len + numbytes);
    }
}

/// Reads u64 value, even if there are less than 8 bytes left.  Reads are little endian.
/// Cursor state is modified.  Will never read beyond end of inbuf.
#[inline(always)]
pub fn direct_read_uint_le(cursor: &mut Cursor<&[u8]>, inbuf: &[u8]) -> u64 {
    let pos = cursor.position() as usize;
    if (pos + 8) <= inbuf.len() {
        // TODO: use Result's unwrap_or_else to handle if input is short
        cursor.read_u64::<LittleEndian>().unwrap()
    } else {
        // Less than 8 bytes left.  Use Byteorder implementation which can read limited # of bytes.
        // This ensures we don't read from a space we are not allowed to.
        cursor.read_uint::<LittleEndian>(inbuf.len() - pos).unwrap()
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