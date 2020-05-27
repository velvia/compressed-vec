#[macro_use]
extern crate criterion;
extern crate compressed_vec;

use criterion::{Criterion, Benchmark, BenchmarkId, Throughput};
use compressed_vec::*;
use compressed_vec::sink::{Sink, U32_256Sink};
use compressed_vec::section::{FixedSectReader, NibblePackU32MedFixedSect};

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

// About 50% nonzeroes, but vary number of bits
fn sinewave_varnumbits_u32(max_numbits: u8, len: usize) -> Vec<u32> {
    let amp = 2usize.pow(max_numbits as u32) - 1;
    // Sinusoid between -1 and 1 with period of ~20
    (0..len).map(|i| ((i as f32) * std::f32::consts::PI / 10.0).sin())
    // If value is negative, turn it into a zero
            .map(|i| {
                let new_value = i * amp as f32;
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
            sink.reset();
            nibblepacking::unpack(&buf[..], &mut sink, inputs.len()).unwrap();
        })
    });
}

use section::FixedSectionWriter;

fn section32_decode_dense_lowcard_varnonzeroes(c: &mut Criterion) {
    let mut group = c.benchmark_group("section u32 decode");
    group.throughput(Throughput::Elements(256));

    for nonzero_f in [0.05, 0.25, 0.5, 0.9, 1.0].iter() {
        let inputs = sinewave_varnonzeros_u32(*nonzero_f, 256);
        let mut buf = [0u8; 1024];
        NibblePackU32MedFixedSect::write(&mut buf, 0, &inputs[..]).unwrap();

        group.bench_with_input(BenchmarkId::new("dense low card, nonzero%: ", *nonzero_f), &buf,
                               |b, buf| b.iter(|| {
            let mut sink = U32_256Sink::new();
            NibblePackU32MedFixedSect::try_from(buf).unwrap().decode_to_sink(&mut sink).unwrap();
        }));
    }
}

fn section32_decode_dense_varnumbits(c: &mut Criterion) {
    let mut group = c.benchmark_group("section u32 decode numbits");
    group.throughput(Throughput::Elements(256));

    for numbits in [4, 8, 12, 16, 20].iter() {
        let inputs = sinewave_varnumbits_u32(*numbits, 256);
        let mut buf = [0u8; 1024];
        NibblePackU32MedFixedSect::write(&mut buf, 0, &inputs[..]).unwrap();

        group.bench_with_input(BenchmarkId::new("dense low card, numbits: ", *numbits), &buf,
                               |b, buf| b.iter(|| {
            let mut sink = U32_256Sink::new();
            NibblePackU32MedFixedSect::try_from(buf).unwrap().decode_to_sink(&mut sink).unwrap();
        }));
    }
}

const VECTOR_LENGTH: usize = 10000;

fn dense_lowcard_vector() -> Vec<u8> {
    let inputs = sinewave_varnonzeros_u32(1.0, VECTOR_LENGTH);
    let mut appender = vector::VectorU32Appender::new(8192).unwrap();
    inputs.iter().for_each(|a| appender.append(*a).unwrap());
    appender.finish(VECTOR_LENGTH).unwrap()
}

fn sparse_lowcard_vector(num_nonzeroes: usize) -> Vec<u8> {
    let nonzeroes = sinewave_varnonzeros_u32(1.0, num_nonzeroes/2);
    let nulls = VECTOR_LENGTH - num_nonzeroes;

    let mut appender = vector::VectorU32Appender::new(8192).unwrap();
    appender.append_nulls(nulls/4).unwrap();
    nonzeroes.iter().for_each(|a| appender.append(*a).unwrap());
    appender.append_nulls(nulls/2).unwrap();
    nonzeroes.iter().for_each(|a| appender.append(*a).unwrap());
    appender.finish(VECTOR_LENGTH).unwrap()
}

fn bench_filter_vect(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector filtering");
    group.throughput(Throughput::Elements(VECTOR_LENGTH as u64));

    let dense_vect = dense_lowcard_vector();
    let sparse_vect = sparse_lowcard_vector(100);
    let dense_reader = vector::VectorReader::<u32>::try_new(&dense_vect[..]).unwrap();
    let sparse_reader = vector::VectorReader::<u32>::try_new(&sparse_vect[..]).unwrap();

    group.bench_function("lowcard u32", |b| b.iter(|| {
        let filter_iter = dense_reader.filter_iter(filter::EqualsSink::<u32>::new(&3));
        filter::count_hits(filter_iter);
    }));
    group.bench_function("very sparse lowcard u32", |b| b.iter(|| {
        let filter_iter = sparse_reader.filter_iter(filter::EqualsSink::<u32>::new(&15));
        filter::count_hits(filter_iter);
    }));
    group.bench_function("dense + sparse lowcard combo", |b| b.iter(|| {
        let dense_iter = dense_reader.filter_iter(filter::EqualsSink::<u32>::new(&3));
        let sparse_iter = sparse_reader.filter_iter(filter::EqualsSink::<u32>::new(&15));
        let filter_iter = filter::MultiVectorFilter::new(vec![dense_iter, sparse_iter]);
        filter::count_hits(filter_iter);
    }));
    group.bench_function("sparse + dense lowcard combo", |b| b.iter(|| {
        let dense_iter = dense_reader.filter_iter(filter::EqualsSink::<u32>::new(&3));
        let sparse_iter = sparse_reader.filter_iter(filter::EqualsSink::<u32>::new(&15));
        let filter_iter = filter::MultiVectorFilter::new(vec![sparse_iter, dense_iter]);
        filter::count_hits(filter_iter);
    }));

    group.finish();
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
    }).throughput(Throughput::Elements(BATCH_SIZE as u64)));
}

criterion_group!(benches, //nibblepack8_varlen,
                          pack_delta_u64s_varlen,
                          unpack_delta_u64s,
                          section32_decode_dense_lowcard_varnonzeroes,
                          section32_decode_dense_varnumbits,
                          bench_filter_vect,
                          // repack_2d_deltas,
                          );
criterion_main!(benches);
