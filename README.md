## Compressed-Vec

This is a *compressed vec* library, rather than a *compression* library.  What does that mean?

A compression library takes some uncompressed data and provides essentially compress() and decompress() functions.  Typically you have to decompress data to be able to do anything with it, resulting in extra latency and allocations.

On the other hand, this *compressed vec* library gives you a `Vec` (or list, or array) like interface for working with compressed representations (vectors) of specific kinds of data (in this case, integers and floating point data).  This allows you to efficiently store large quantities of certain data in memory while being able to iterate over it, without having to decompress large quantities.  It is designed for fast iteration (not necessarily random access) of compressed vectors, accelerated by SIMD optimizations.  Additional SIMD-enabled filtering and other vector capabilities are also provided.

### Creation, Iteration

### Filtering

### Vector Format

Details of the vector format can be found [here](vector_format.md).