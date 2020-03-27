use byteorder::{LittleEndian, ReadBytesExt};
use scroll::{Pwrite, LE};
use std::io::Cursor;

#[derive(Debug, PartialEq)]
pub enum NibblePackError {
    InputTooShort,
    BufferTooShort(usize),
}


/// Fast write of u64.  numbytes least significant bytes are written.
/// Writes into out_buffer[offset..offset+numbytes].
/// Returns offset+numbytes
#[inline]
pub fn direct_write_uint_le(out_buffer: &mut [u8],
                            offset: usize,
                            value: u64,
                            numbytes: usize) -> Result<usize, NibblePackError> {
    // By default, write all 8 bytes checking that there's enough space.
    // We only adjust offset by numbytes, so the happy path is pretty fast.
    let _num_written = out_buffer.pwrite_with(value, offset, LE)
        .or_else(|err| match err {
            _ => {
                if out_buffer.len() < offset + numbytes {
                    Err(NibblePackError::BufferTooShort(offset + numbytes - out_buffer.len()))
                } else {
                    // Copy only numbytes bytes to be memory safe
                    let bytes = value.to_le_bytes();
                    out_buffer[offset..offset+numbytes].copy_from_slice(&bytes[0..numbytes]);
                    Ok(numbytes)
                }
            },
        })?;
    Ok(offset + numbytes)
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
