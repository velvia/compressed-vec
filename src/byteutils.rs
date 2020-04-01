use crate::error::CodingError;

use scroll::{Pread, Pwrite, LE};

/// Fast write of u64.  numbytes least significant bytes are written.
/// Writes into out_buffer[offset..offset+numbytes].
/// Returns offset+numbytes
#[inline]
pub fn direct_write_uint_le(out_buffer: &mut [u8],
                            offset: usize,
                            value: u64,
                            numbytes: usize) -> Result<usize, CodingError> {
    // By default, write all 8 bytes checking that there's enough space.
    // We only adjust offset by numbytes, so the happy path is pretty fast.
    let _num_written = out_buffer.pwrite_with(value, offset, LE)
        .or_else(|err| match err {
            _ => {
                if out_buffer.len() < offset + numbytes {
                    Err(CodingError::NotEnoughSpace)
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
/// Will never read beyond end of inbuf.  Pos is position within inbuf.
#[inline(always)]
pub fn direct_read_uint_le(inbuf: &[u8], pos: usize) -> Result<u64, CodingError> {
    inbuf.pread_with::<u64>(pos, LE)
        .or_else(|err| match err {
            _ => {
                let remaining = inbuf.len() as isize - (pos as isize);
                if remaining > 0 {
                    let mut buf = [0u8; 8];
                    buf[0..remaining as usize].copy_from_slice(&inbuf[pos..]);
                    Ok(u64::from_le_bytes(buf))
                } else {
                    Err(CodingError::NotEnoughSpace)
                }
            }
        })
}
