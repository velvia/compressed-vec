#[derive(Debug, PartialEq)]
pub enum CodingError {
    NotEnoughSpace,
    InputTooShort,
    BadOffset(usize),
    InvalidSectionType(u8),
    InvalidFormat(String),
    ScrollErr(String),
}

impl From<scroll::Error> for CodingError {
    fn from(err: scroll::Error) -> CodingError {
        match err {
            scroll::Error::TooBig { .. }  => CodingError::NotEnoughSpace,
            scroll::Error::BadOffset(off) => CodingError::BadOffset(off),
            _ => CodingError::ScrollErr(err.to_string()),
        }
    }
}
