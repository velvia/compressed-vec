use plain::Plain;
use crate::nibblepacking::*;

#[derive(Debug)]
#[repr(u8)]
pub enum BinHistogramFormat {
    Empty = 0x00,
    GeometricDelta = 0x01,
    Geometric1Delta = 0x02,
}

/// Header for a compressed histogram, not including any length prefix bytes.  A compressed histogram
/// contains bucket definitions and compressed bucket values, usually compressed using nibblepacking.
#[repr(C, packed)]
#[derive(Debug)]
struct BinHistogramHeader {
    format_code: BinHistogramFormat,
    bucket_def_len: u16,
    num_buckets: u16,
}

unsafe impl Plain for BinHistogramHeader {}

impl BinHistogramHeader {
    fn from_bytes(buf: &[u8]) -> &BinHistogramHeader {
        plain::from_bytes(buf).expect("The buffer is either too short or not aligned!")
    }

    // Returns the byte slice for the compressed binary bucket values
    fn values_byteslice<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        let values_index = offset_of!(BinHistogramHeader, num_buckets) + self.bucket_def_len as usize;
        &buf[values_index..]
    }
}

#[repr(C, packed)]
#[derive(Debug)]
struct PackedGeometricBuckets {
    initial_bucket: f64,
    multiplier: f64,
}

unsafe impl Plain for PackedGeometricBuckets {}

///
/// Compresses raw histogram values with geometric bucket definitions and non-increasing bucket values as a delta-
/// encoded compressed histogram -- ie, the raw values will be considered deltas and parsed as increasing buckets.
///
/// This method should be called to convert non-increasing histogram buckets to the internal increasing bucket
/// format.  The outbuf must have been cleared already though it can have other data in it.
pub fn compress_geom_nonincreasing(num_buckets: u16,
                                   initial_bucket: f64,
                                   multiplier: f64,
                                   format_code: BinHistogramFormat,
                                   bucket_values: &[u64],
                                   outbuf: &mut Vec<u8>) {
    // First, write out BinHistogramHeader
    let bucket_def_len = mem::size_of::<PackedGeometricBuckets>() as u16 + 2;
    let header = BinHistogramHeader { format_code, bucket_def_len, num_buckets };
    unsafe { outbuf.extend_from_slice(plain::as_bytes(&header)) };

    // Then, write out geometric values
    let geom_buckets = PackedGeometricBuckets { initial_bucket, multiplier };
    unsafe { outbuf.extend_from_slice(plain::as_bytes(&geom_buckets)) };

    // Finally, pack the values
    pack_u64(bucket_values.into_iter().cloned(), outbuf);
}

use std::mem;

#[test]
fn dump_header_structure() {
    let header = BinHistogramHeader {
        format_code: BinHistogramFormat::GeometricDelta,
        bucket_def_len: 2,
        num_buckets: 16,
    };

    println!("size of header: {:?}", mem::size_of::<BinHistogramHeader>());
    println!("align of header: {:?}", mem::align_of::<BinHistogramHeader>());
    println!("span of bucket_def_len: {:?}", span_of!(BinHistogramHeader, bucket_def_len));

    unsafe {
        let slice = plain::as_bytes(&header);
        assert_eq!(slice, [0x01u8, 0x02, 0, 16, 0]);

        let new_header = BinHistogramHeader::from_bytes(slice);
        println!("new_header: {:?}", new_header);
    }
}