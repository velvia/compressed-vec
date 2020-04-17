## compressed_vec Vector Format

The format for each vector is defined in detail below.  It is loosely based on the vector format in [FiloDB](https://github.com/filodb/FiloDB) used for histograms.  Each vector is divided into sections of 256 elements.

The vector bytes are wire-format ready, they can be written to and read from disk or network and interpreted/read with no further transformations needed.

### Header

The header is 16 bytes.  The first 6 bytes are binary compatible with FiloDB vectors.  The structs are defined in `src/vector.rs` in the `BinaryVector` and `FixedSectStats` structs.

| offset | description |
| ------ | ----------- |
| +0     | u32: total number of bytes in this vector, NOT including these 4 length bytes |
| +4     | u8: Major vector type, see the `VectorType` enum for details  |
| +5     | u8: Vector subtype, see the `VectorSubType` enum for details  |
| +8     | u32: total number of elements in this vector                  |
| +12    | u16: number of null sections in this vector, used for quickly determining relative sparsity  |

For the vectors produced by this crate, the major type code used is `VectorType::FixedSection256` (0x10), while the minor type code is `Primitive`.

### Sections

Following the header are one or more sections of fixed 256 elements each.  If the last section does not have 256 elements, nulls are added until the section has 256 elements.

The first byte of a section contains the section code.  See the `SectionType` enum in `src/section.rs` for the up to date list of codes, but it is included here for convenience:

```rust
pub enum SectionType {
    Null = 0,                // n unavailable or null elements in a row
    NibblePackedU64Medium = 1,   // Nibble-packed u64's, total size < 64KB
    NibblePackedU32Medium = 2,   // Nibble-packed u32's, total size < 64KB
}
```

### Null Sections

A null section represents 256 zeroes, and just contains the single Null section type byte.

Null sections are key to encoding sparse vectors efficiently, and should be leveraged as much as possible.

### NibblePacked U64/U32 sections

The NibblePacked section codes (1/2) represent 256 values (u32 or u64), packed in groups of 8 using the [NibblePacking](https://github.com/filodb/FiloDB/blob/develop/doc/compression.md#predictive-nibblepacking) algorithm from FiloDB.  NibblePacking uses only 1 bit for zero values, and stores the minimum number of nibbles only.  From the start of the section, there are 3 header bytes, followed by the NibblePack-encoded data.

| offset | description |
| ------ | ----------- |
| +0     | u8: section type code: 1 or 2 |
| +1     | u16: number of bytes of this section, excluding these 3 header bytes  |
| +3     | Start of NibblePack-encoded data, back to back.   This starts with the bitmask byte, then the number of nibbles byte, then the nibbles, repeated for every group of 8 u64's/u32's |

### Filtering and Vector Processing

Fast filtering and vector processing of multiple vectors is enabled by the following:
* All sections are the same number of elements across vectors, thus
* we can iterate through sections and process the same set of elements across multiple vectors at the same time
* processing of sparse vectors with null sections can be optimized and special cased

We can see examples of this in `src/filter.rs`, with the `VectorFilter` and `MultiVectorFilter` structs.
