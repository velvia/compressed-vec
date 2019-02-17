#[macro_use]
extern crate criterion;
extern crate rustfilo;

use criterion::Criterion;
use rustfilo::nibblepacking;

fn nibblepack_partial_even(c: &mut Criterion) {
    let mut buf = Vec::with_capacity(1024);
    let inputs = [
        0u64,
        0x0000_0033_2211_0000u64,
        0x0000_0044_3322_0000u64,
        0x0000_0055_4433_0000u64,
        0x0000_0066_5544_0000u64,
        0u64,
        0u64,
        0u64,
    ];
    c.bench_function("nibblepack partial even", move |b| {
        b.iter(|| {
            buf.clear();
            nibblepacking::nibble_pack8(&inputs, &mut buf)
        })
    });
}

//TODO: see if we can combine several bench functions together
fn nibblepack_partial_odd(c: &mut Criterion) {
    let mut buf = Vec::with_capacity(1024);
    let inputs = [
        0u64,
        0x0000_0033_2210_0000u64,
        0x0000_0044_3320_0000u64,
        0x0000_0055_4430_0000u64,
        0x0000_0066_5540_0000u64,
        0x0000_0076_5430_0000u64,
        0u64,
        0u64,
    ];
    c.bench_function("nibblepack partial odd", move |b| {
        b.iter(|| {
            buf.clear();
            nibblepacking::nibble_pack8(&inputs, &mut buf)
        })
    });
}

criterion_group!(benches, nibblepack_partial_even, nibblepack_partial_odd);
criterion_main!(benches);
