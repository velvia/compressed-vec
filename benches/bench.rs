#[macro_use]
extern crate criterion;
extern crate compressed_vec;

use criterion::{Criterion, Benchmark, Throughput};
use compressed_vec::{histogram, nibblepacking, nibblepack_simd, section};

fn nibblepack8_varlen(c: &mut Criterion) {
    // This method from Criterion allows us to run benchmarks and vary some variable.
    // Thus we can discover how much effect the # of bits and # of nonzero elements make.
    // Two discoveries:
    //  1) Varying the # of nonzero elements matters, but not really the # of bits per element.
    //  2) For some odd reason, running just a single benchmark speeds it up significantly, running two slows down each one by half
    c.bench_function_over_inputs("nibblepack8 varying nonzeroes", |b, &&nonzeroes| {
        let mut inputbuf = [0u64; 8];
        let mut buf = [0u8; 1024];
        for i in 0..nonzeroes {
            inputbuf[i] = 0x1234u64 + i as u64;
        }
        b.iter(|| {
            nibblepacking::nibble_pack8(&inputbuf, &mut buf, 0).unwrap();
        })
    }, &[0, 2, 4, 6, 8]);
}


fn nibblepack8_varnumbits(c: &mut Criterion) {
    c.bench_function_over_inputs("nibblepack8 varying # bits", |b, &&numbits| {
        let mut inputbuf = [0u64; 8];
        let mut buf = [0u8; 1024];
        for i in 0..4 {
            inputbuf[i] = (1u64 << numbits) - 1 - (i as u64);
        }
        b.iter(|| {
            nibblepacking::nibble_pack8(&inputbuf, &mut buf, 0).unwrap();
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

fn sinewave_varnonzeros_u32(fract_nonzeroes: f32, len: usize) -> Vec<u32> {
    let amp = 7.0 / fract_nonzeroes;
    let dist = 15.0 - amp;
    // Sinusoid between -1 and 1 with period of ~20
    (0..len).map(|i| ((i as f32) * std::f32::consts::PI / 10.0).sin())
    // Change amplitude to 8/fract_nonzeroes; center so max is 16:
    // If value is negative, turn it into a zero
            .map(|i| {
                let new_value = i * amp + dist;
                if new_value >= 0.0 { new_value as u32 } else { 0 }
            }).collect()
}

// Pack 64 u64's, variable number of them are 0
fn pack_delta_u64s_varlen(c: &mut Criterion) {
    c.bench_function_over_inputs("pack delta u64s varying nonzero numbers", |b, &&nonzeroes| {
        let inputs = increasing_nonzeroes_u64x64(nonzeroes);
        let mut buf = [0u8; 1024];
        b.iter(|| {
            nibblepacking::pack_u64_delta(&inputs, &mut buf).unwrap();
        })
    }, &[2, 4, 8, 16]);
}

fn unpack_delta_u64s(c: &mut Criterion) {
    c.bench_function("unpack delta u64s", |b| {
        let inputs = increasing_nonzeroes_u64x64(16);
        let mut buf = [0u8; 1024];
        nibblepacking::pack_u64_delta(&inputs, &mut buf).unwrap();

        let mut sink = nibblepacking::DeltaSink::new();
        b.iter(|| {
            sink.clear();
            nibblepacking::unpack(&buf[..], &mut sink, inputs.len()).unwrap();
        })
    });
}

use section::FixedSectionWriter;

fn section32_decode_dense_lowcard_varnonzeroes(c: &mut Criterion) {
    c.bench_function_over_inputs("decode u32 section: dense low-card: varying nonzero %", |b, &&nonzero_f| {
        let inputs = sinewave_varnonzeros_u32(nonzero_f, 256);
        let mut buf = [0u8; 1024];
        section::NibblePackU32MedFixedSect::write(&mut buf, 0, &inputs[..]).unwrap();

        b.iter(|| {
            let mut sink = nibblepack_simd::U32_256Sink::new();
            section::NibblePackU32MedFixedSect::decode_to_sink(&buf, &mut sink).unwrap();
        });
    }, &[0.05, 0.25, 0.5, 0.9, 1.0]);
}

const BATCH_SIZE: usize = 100;

fn repack_2d_deltas(c: &mut Criterion) {
    c.bench("repack 2D diff deltas",
            Benchmark::new("100x u64", |b| {

        let orig = increasing_nonzeroes_u64x64(16);
        let mut inputs = [0u64; 64];
        let mut srcbuf = [0u8; 1024];
        for i in 0..BATCH_SIZE {
            for j in 0..orig.len() {
                inputs[j] = orig[j] + ((j + i) as u64);
            }
            nibblepacking::pack_u64_delta(&inputs, &mut srcbuf).unwrap();
        }

        let mut out_buf = [0u8; 4096];
        let mut sink = histogram::DeltaDiffPackSink::new(inputs.len(), &mut out_buf);

        b.iter(|| {
            // Reset out_buf and sink last_deltas state
            sink.reset();
            let mut slice = &srcbuf[..];

            for _ in 0..BATCH_SIZE {
                let res = nibblepacking::unpack(slice, &mut sink, 64);
                sink.finish();
                slice = res.unwrap();
            }
        })
    }).throughput(Throughput::Elements(BATCH_SIZE as u32)));
}

criterion_group!(benches, //nibblepack8_varlen,
                          // nibblepack8_varnumbits,
                          pack_delta_u64s_varlen,
                          unpack_delta_u64s,
                          section32_decode_dense_lowcard_varnonzeroes,
                          // repack_2d_deltas,
                          );
criterion_main!(benches);
