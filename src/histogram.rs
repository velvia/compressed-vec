use packed_simd_2::u64x8;
use plain::Plain;
use crate::nibblepacking::*;
use crate::sink::Sink;

#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum BinHistogramFormat {
    Empty = 0x00,
    GeometricDelta = 0x01,
    Geometric1Delta = 0x02,
}

/// Header for a compressed histogram, not including any length prefix bytes.  A compressed histogram
/// contains bucket definitions and compressed bucket values, usually compressed using nibblepacking.
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BinHistogramHeader {
    format_code: BinHistogramFormat,
    bucket_def_len: u16,
    num_buckets: u16,
}

unsafe impl Plain for BinHistogramHeader {}

impl BinHistogramHeader {
    #[allow(dead_code)]
    pub fn from_bytes(buf: &[u8]) -> &BinHistogramHeader {
        plain::from_bytes(buf).expect("The buffer is either too short or not aligned!")
    }

    // Returns the byte slice for the compressed binary bucket values
    #[allow(dead_code)]
    pub fn values_byteslice<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        let values_index = offset_of!(BinHistogramHeader, num_buckets) + self.bucket_def_len as usize;
        &buf[values_index..]
    }
}

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
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
                                   outbuf: &mut [u8]) {
    // First, write out BinHistogramHeader
    let bucket_def_len = mem::size_of::<PackedGeometricBuckets>() as u16 + 2;
    let header = BinHistogramHeader::from_mut_bytes(outbuf).unwrap();
    header.format_code = format_code;
    header.bucket_def_len = bucket_def_len;
    header.num_buckets = num_buckets;

    // Then, write out geometric values
    let header_size = mem::size_of::<BinHistogramHeader>();
    let geom_buckets = PackedGeometricBuckets::from_mut_bytes(&mut outbuf[header_size..]).unwrap();
    geom_buckets.initial_bucket = initial_bucket;
    geom_buckets.multiplier = multiplier;

    // Finally, pack the values
    pack_u64(bucket_values.into_iter().cloned(), outbuf, (bucket_def_len + 3) as usize).unwrap();
}

///
/// A sink used for increasing histogram counters.  In one shot:
/// - Unpacks a delta-encoded NibblePack compressed Histogram
/// - Subtracts the values from lastHistValues, noting if the difference is not >= 0 (means counter reset)
/// - Packs the subtracted values
/// - Updates lastHistValues to the latest unpacked values so this sink can be used again
///
/// Meant to be used again and again to parse next histogram, thus the last_hist_deltas
/// state is reused to compute the next set of deltas.
/// If the new set of values is less than last_hist_deltas then the new set of values is
/// encoded instead of the diffs.
/// For more details, see the "2D Delta" section in [compression.md](doc/compression.md)
#[derive(Default)]
#[derive(Debug)]
pub struct DeltaDiffPackSink<'a> {
    value_dropped: bool,
    i: usize,
    last_hist_deltas: Vec<u64>,
    pack_array: [u64; 8],
    out_offset: usize,
    out_buf: &'a mut [u8],
}

impl<'a> DeltaDiffPackSink<'a> {
    /// Creates new DeltaDiffPackSink
    pub fn new(num_buckets: usize, out_buf: &'a mut [u8]) -> Self {
        let mut last_hist_deltas = Vec::<u64>::with_capacity(num_buckets);
        last_hist_deltas.resize(num_buckets, 0);
        Self { last_hist_deltas, out_buf, ..Default::default() }
    }

    pub fn reset_out_buf(&mut self) {
        self.out_offset = 0;
    }

    /// Call this to finish packing the remainder of the deltas and reset for next go
    #[inline]
    pub fn finish(&mut self) {
        // TODO: move this to a pack_remainder function?
        if self.i != 0 {
            for j in self.i..8 {
                self.pack_array[j] = 0;
            }
            self.out_offset = nibble_pack8(&self.pack_array, self.out_buf, self.out_offset).unwrap();
        }
        self.i = 0;
        self.value_dropped = false;
    }
}

impl<'a> Sink<u64x8> for DeltaDiffPackSink<'a> {
    #[inline]
    fn process(&mut self, data: u64x8) {
        let maxlen = self.last_hist_deltas.len();
        let looplen = if self.i + 8 <= maxlen { 8 } else { maxlen - self.i };
        for n in 0..looplen {
            let last_value = self.last_hist_deltas[self.i + n];
            // If data dropped from last, write data instead of diff
            // TODO: actually try to use the SIMD
            let data_item = data.extract(n);
            if data_item < last_value {
                self.value_dropped = true;
                self.pack_array[n] = data_item;
            } else {
                self.pack_array[n] = data_item - last_value;
            }
        }
        // copy data wholesale to last_hist_deltas
        for n in self.i..(self.i+looplen) {
            self.last_hist_deltas[n] = data.extract(n - self.i);
        }
        // if numElems < 8, zero out remainder of packArray
        for n in looplen..8 {
            self.pack_array[n] = 0;
        }
        self.out_offset = nibble_pack8(&self.pack_array, self.out_buf, self.out_offset).unwrap();
        self.i += 8;
    }

    fn process_zeroes(&mut self) {
        todo!();
    }

    // Resets everythin, even the out_buf.  Probably should be used only for testing
    #[inline]
    fn reset(&mut self) {
        self.i = 0;
        self.value_dropped = false;
        for elem in self.last_hist_deltas.iter_mut() {
            *elem = 0;
        }
        self.out_offset = 0;
    }
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

#[test]
fn delta_diffpack_sink_test() {
    let inputs = [ [0u64, 1000, 1001, 1002, 1003, 2005, 2010, 3034, 4045, 5056, 6067, 7078],
                   [3u64, 1004, 1006, 1008, 1009, 2012, 2020, 3056, 4070, 5090, 6101, 7150],
                   // [3u64, 1004, 1006, 1008, 1009, 2010, 2020, 3056, 4070, 5090, 6101, 7150],
                   [7u64, 1010, 1016, 1018, 1019, 2022, 2030, 3078, 4101, 5122, 6134, 7195] ];
    let diffs = inputs.windows(2).map(|pair| {
        pair[1].iter().zip(pair[0].iter()).map(|(nb, na)| nb - na ).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    // Compress each individual input into its own buffer
    let compressed_inputs: Vec<[u8; 256]> = inputs.iter().map(|input| {
        let mut buf = [0u8; 256];
        pack_u64_delta(&input[..], &mut buf).unwrap();
        buf
    }).collect();

    let mut out_buf = [0u8; 1024];
    let mut sink = DeltaDiffPackSink::new(inputs[0].len(), &mut out_buf);

    // Verify delta on first one (empty diffs) yields back the original
    let _res = unpack(&compressed_inputs[0], &mut sink, inputs[0].len());
    // assert_eq!(res.unwrap().len(), 0);
    sink.finish();

    let mut dsink = DeltaSink::new();
    let _res = unpack(sink.out_buf, &mut dsink, inputs[0].len());
    assert_eq!(dsink.output_vec()[..inputs[0].len()], inputs[0]);

    // Second and subsequent inputs shouyld correspond to diffs
    for i in 1..3 {
        sink.reset_out_buf();   // need to reset output
        let _res = unpack(&compressed_inputs[i], &mut sink, inputs[0].len());
        // assert_eq!(res.unwrap().len(), 0);
        assert_eq!(sink.value_dropped, false);  // should not have dropped?
        sink.finish();
        // dbg!(&sink.out_vec);

        let mut dsink = DeltaSink::new();
        let _res = unpack(sink.out_buf, &mut dsink, inputs[0].len());
        assert_eq!(dsink.output_vec()[..inputs[0].len()], diffs[i - 1][..]);
    }
}
