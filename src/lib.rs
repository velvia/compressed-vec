#[macro_use]
extern crate memoffset;

pub mod nibblepacking;
mod nibblepack_simd;
pub mod byteutils;
pub mod vector;
pub mod histogram;
pub mod section;
pub mod error;

#[no_mangle]
pub extern "C" fn double_input(input: i32) -> i32 {
    input * 2
}
