## compressed_vec

[![crate](https://img.shields.io/crates/v/compressed_vec.svg)](https://crates.io/crates/compressed_vec)
[![documentation](https://docs.rs/compressed_vec/badge.svg)](https://docs.rs/compressed_vec)
[![CircleCI](https://circleci.com/gh/velvia/compressed-vec.svg?style=shield)](https://circleci.com/gh/velvia/compressed-vec)

Floating point and integer compressed vector library, SIMD-enabled for fast processing/iteration over compressed representations.

This is a *compressed vec* library, rather than a *compression* library.  What does that mean?  A compression library takes some uncompressed data and provides essentially compress() and decompress() functions.  Typically you have to decompress data to be able to do anything with it, resulting in extra latency and allocations.

On the other hand, this *compressed vec* library allows you to iterate over and process the compressed representations directly.  It is designed to balance fast iteration and SIMD processing/filtering, while compressing vectors to within 2x of the best columnar compression technology such as Apache Parquet, using techniques such as delta and XOR encoding.  Applications:

* Database engines
* Large in-memory data processing
* Games and other apps needing fast access to large quantities of FP vectors/matrices

### Performance Numbers

Numbers are from my laptop: 2.9 GHz Core i9, 6/12 cores, 12MB L3, AVX2; from running `cargo bench vector`, which benchmarks a filter-and-count-matches operation directly on encoded/compressed vectors.

| Vector type(s) | Elements/sec | Raw GBs per sec |
| -------------- | ------------ | --------------- |
| u32 dense (no sparsity) | 1.7 Gelems/s  | 6.8 GB/s  |
| u32 sparse (99% zeros)  | 13.9 Gelems/s | 55.6 GB/s |
| Two u32 vectors (sparse + dense)* |  1.3-5.2 Gelems/s | 5-20 GB/s |
| u64 vector, dense       |  955M - 1.1 Gelems/s        | 7.6 - 9.1 GB/s |
| f32, XOR encoded, 60% density |  985 Melems/s         | 3.9 GB/s       |

* The two u32 vector filtering speed (using `MultiVectorFilter`) depends on the order of the vectors.  It is faster to filter the sparse vector first.

### Creation, Iteration

To create an f32 compressed vector:

```rust
    use compressed_vec::VectorF32XorAppender;
    let mut appender = VectorF32XorAppender::try_new(2048).unwrap();
    let encoded_bytes = appender.encode_all(vec![1.0, 1.5, 2.0, 2.5]).unwrap();
```

The simplest way to iterate on this compressed vector (note this does not allocate at all):

```rust
    use compressed_vec::VectorReader;
    let reader = VectorReader::<f32>::try_new(&encoded_bytes[..]).unwrap();
    let sum = reader.iterate().sum::<f32>();   // Yay, no allocations!
```

### Filtering and SIMD Processing

`iterate()` is the easiest API to go through individual elements of the compressed vector, but it is not the fastest.  Fast data processing, such as done in the filter-and-count benchmarks above, are performed using `Sink`s, which are defined in the `sink` module.  Sinks operate on a SIMD word at a time, and the sink API is designed for inlining.

For example, let's say that we want to add 2.5 to the f32 vector above, and then write out the results to a `Vec<f32>`.  Internally, XOR encoding and decoding is performed (using a sink).  The sinks can be stacked during decoding, for an almost entirely SIMD pipeline:
    - `XorSink` (used automatically for f32 decoding)
    - `AddConstSink`  (to add 2.5, again done using SIMD)
    - `VecSink`  (writes output to a normal Vec)

```rust
    use compressed_vec::{VectorReader, AddConstSink, VecSink};
    let reader = VectorReader::<f32>::try_new(&encoded_bytes[..]).unwrap();
    let mut vecsink = VecSink::<f32>::new();
    let mut addsink = AddConstSink::new(2.5f32, &mut vecsink);
    reader.decode_to_sink(&mut addsink).unwrap();
    println!("And the transformed vector is: {:?}", vecsink.vec);
```

### Vector Format

Details of the vector format can be found [here](vector_format.md).

The vector format follows columnar compression techniques used throughout the big data and database world, and roughly follows the Google [Procella](https://blog.acolyer.org/2019/09/11/procella/) paper with its custom Artus format:

* Compression within 2x of ZSTD while operating directly on the data
    * Compression for this format is within 2x of Parquet, but is written to allow filtering and operating on the data directly without needing a separate decompression step for the entire vector
* Multi-pass encoding
    * The `VectorAppender` collects min/max and other stats on the raw data and uses it to decide on the best encoding strategy (delta, etc.)
* Exposing dictionary indices to query engine and aggressive pushdown to the data format
    * The format is designed to filter over dictionary codes, which speeds up filtering
    * The use of sections allows for many optimizations for filtering.  For example, null sections and constant sections allow for very fast filter short-circuiting.

### Collaboration

Please reach out to me to collaborate!