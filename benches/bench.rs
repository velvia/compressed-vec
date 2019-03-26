#[macro_use]
extern crate criterion;
extern crate rustfilo;

use criterion::Criterion;
use rustfilo::nibblepacking;

fn nibblepack8_varlen(c: &mut Criterion) {
    // This method from Criterion allows us to run benchmarks and vary some variable.
    // Thus we can discover how much effect the # of bits and # of nonzero elements make.
    // Two discoveries:
    //  1) Varying the # of nonzero elements matters, but not really the # of bits per element.
    //  2) For some odd reason, running just a single benchmark speeds it up significantly, running two slows down each one by half
    c.bench_function_over_inputs("nibblepack8 varying nonzeroes", |b, &&nonzeroes| {
        let mut inputbuf = [0u64; 8];
        let mut buf = Vec::with_capacity(1024);
        for i in 0..nonzeroes {
            inputbuf[i] = 0x1234u64 + i as u64;
        }
        b.iter(|| {
            buf.clear();
            nibblepacking::nibble_pack8(&inputbuf, &mut buf)
        })
    }, &[0, 2, 4, 6, 8]);
}


fn nibblepack8_varnumbits(c: &mut Criterion) {
    c.bench_function_over_inputs("nibblepack8 varying # bits", |b, &&numbits| {
        let mut inputbuf = [0u64; 8];
        let mut buf = Vec::with_capacity(1024);
        for i in 0..4 {
            inputbuf[i] = (1u64 << numbits) - 1 - (i as u64);
        }
        b.iter(|| {
            buf.clear();
            nibblepacking::nibble_pack8(&inputbuf, &mut buf)
        })
    }, &[8, 12, 16, 20, 24, 32]);
}

fn make_nonzeroes_u64x64(num_nonzeroes: usize) -> [u64; 64] {
    let mut inputs = [0u64; 64];
    for i in 1..=num_nonzeroes {
        inputs[i] = (((i as f32) * std::f32::consts::PI / (num_nonzeroes as f32)).sin() * 1000.0) as u64
    }
    inputs
}

fn increasing_nonzeroes_u64x64(num_nonzeroes: usize) -> [u64; 64] {
    let mut inputs = make_nonzeroes_u64x64(num_nonzeroes);
    for i in 1..64 {
        inputs[i] = inputs[i - 1] + inputs[i];
    }
    inputs
}

// Pack 64 u64's, variable number of them are 0
fn pack_delta_u64s_varlen(c: &mut Criterion) {
    c.bench_function_over_inputs("pack delta u64s varying nonzero numbers", |b, &&nonzeroes| {
        let inputs = increasing_nonzeroes_u64x64(nonzeroes);
        let mut buf = Vec::with_capacity(1024);
        b.iter(|| {
            buf.clear();
            nibblepacking::pack_u64_delta(&inputs, &mut buf)
        })
    }, &[2, 4, 8, 16]);
}

fn unpack_delta_u64s(c: &mut Criterion) {
    c.bench_function("unpack delta u64s", |b| {
        let inputs = increasing_nonzeroes_u64x64(16);
        let mut buf = Vec::with_capacity(1024);
        nibblepacking::pack_u64_delta(&inputs, &mut buf);
        // println!("Packed inputs of {} bytes to buffer of {} bytes", inputs.len() * 8, buf.len());

        let mut sink = nibblepacking::DeltaSink::new();
        b.iter(|| {
            sink.clear();
            nibblepacking::unpack(&buf[..], &mut sink, inputs.len()).unwrap();
        })
    });
}

criterion_group!(benches, //nibblepack8_varlen,
                          nibblepack8_varnumbits,
                          pack_delta_u64s_varlen,
                          unpack_delta_u64s);
criterion_main!(benches);
