use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

#[derive(Debug, PartialEq)]
pub enum NibblePackError {
    InputTooShort,
}


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
pub fn direct_read_uint_le(cursor: &mut Cursor<&[u8]>, inbuf: &[u8]) -> Result<u64, NibblePackError> {
    let pos = cursor.position();
    cursor.read_u64::<LittleEndian>()
        // Not enough space.  Use read_uint to never read beyond remaining bytes - be safe
        .or_else(|_e| {
            let remaining = (inbuf.len() as isize) - (pos as isize);
            if remaining > 0 {
                cursor.set_position(pos);
                cursor.read_uint::<LittleEndian>(remaining as usize)
                    .map_err(|_e| NibblePackError::InputTooShort)
            } else {
                Err(NibblePackError::InputTooShort)
            }
        })
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
