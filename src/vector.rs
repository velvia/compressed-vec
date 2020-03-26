use std::mem;

/// BinaryVector: a compressed vector storing data of the same type
///   enabling high speed operations on compressed data without
///   the need for decompressing (in many cases, exceptions noted)
///
/// A BinaryVector MAY consist of multiple sections.  Each section can represent
/// potentially different encoding parameters (bit widths, sparsity, etc.) and
/// has its own header to allow for quickly skipping ahead even when different
/// sections are encoded differently.
#[repr(C)]
pub struct BinaryVector {
    num_bytes: u32,         // Number of bytes in vector following this length
    major_type: VectorType, // These should probably be enums no?
    minor_type: VectorSubType,
    num_elements: u16,
}

#[repr(u8)]
enum VectorType {
    Empty = 0x01,
    SIMPLE = 0x02,
    DICT = 0x03,
    CONST = 0x04,
    DIFF = 0x05,
    BinSimple = 0x06,
    BinDict = 0x07,
    Delta2 = 0x08, // Delta-delta encoded
}

#[repr(u8)]
enum VectorSubType {
    PRIMITIVE = 0x00,
    STRING = 0x01,
    UTF8 = 0x02,
    FIXEDMAXUTF8 = 0x03, // fixed max size per blob, length byte
    DATETIME = 0x04,
    PrimitiveNoMask = 0x05,
    REPEATED = 0x06, // vectors.ConstVector
    INT = 0x07,      // Int gets special type because Longs and Doubles may be encoded as Int
    IntNoMask = 0x08,
}

impl BinaryVector {
    /// Returns the length of the BinaryVector including the length bytes
    pub fn whole_length(&self) -> u32 {
        self.num_bytes + (mem::size_of::<u32>() as u32)
    }
}

#[test]
fn test_whole_length() {
    let vect1 = BinaryVector {
        num_bytes: 100,
        major_type: VectorType::Delta2,
        minor_type: VectorSubType::INT,
        num_elements: 0,
    };
    assert_eq!(vect1.whole_length(), 104);

    println!("Size of BinaryVector struct: {}", mem::size_of::<BinaryVector>());
    println!("Span of num_elements: {:?}", span_of!(BinaryVector, num_elements))
}
