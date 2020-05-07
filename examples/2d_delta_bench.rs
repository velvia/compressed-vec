use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use compressed_vec::{histogram, nibblepacking};
use compressed_vec::nibblepacking::Sink;

///
/// 2d_delta_bench <<CSV histogram file>>
/// An example benchmark that reads histogram data from a file and repeatedly decodes delta-encoded
/// histograms using DeltaDiffPackSink (ie re-encoding into 2D diff encoded histograms that aren't
/// increasing any more)
/// Each line of the file is CSV, no headers, and is expected to be Prom style (ie increasing bucket to bucket
/// and increasing over time).
///
/// NOTE: be sure to compile in release mode for benchmarking, ie cargo build --release --example 2d_delta_bench
fn main() {
    const NUM_BUCKETS: usize = 64;
    const NUM_LOOPS: usize = 1000;

    let filename = std::env::args().nth(1).expect("No filename given");
    let file = File::open(filename).unwrap();
    let mut srcbuf = [0u8; 65536];
    let mut num_lines = 0;
    let mut offset = 0;

    for line in BufReader::new(&file).lines() {
        // Split and trim lines, parsing into u64.  Delta encode
        let mut last = 0u64;
        let line = line.expect("Could not parse line");
        let num_iter = line.split(',')
                           .map(|s| s.trim().parse::<u64>().unwrap())
                           .map(|n| {
                               let delta = n.saturating_sub(last);
                               last = n;
                               delta
                           });
        offset = nibblepacking::pack_u64(num_iter, &mut srcbuf, offset).unwrap();
        num_lines += 1;
    }

    println!("Finished reading and compressing {} histograms, now running {} iterations of 2D Delta...",
             num_lines, NUM_LOOPS);

    let mut out_buf = [0u8; 4096];
    let mut sink = histogram::DeltaDiffPackSink::new(NUM_BUCKETS, &mut out_buf);
    let start = Instant::now();

    for _ in 0..NUM_LOOPS {
        // Reset out_buf and sink last_deltas state
        sink.reset();
        let mut slice = &srcbuf[..];

        for _ in 0..num_lines {
            let res = nibblepacking::unpack(slice, &mut sink, NUM_BUCKETS);
            sink.finish();
            slice = res.unwrap();
        }
    }

    let elapsed_millis = start.elapsed().as_millis();
    let rate = (num_lines * NUM_LOOPS * 1000) as u128 / elapsed_millis;
    println!("{} encoded in {} ms = {} histograms/sec", num_lines * NUM_LOOPS, elapsed_millis, rate);
}