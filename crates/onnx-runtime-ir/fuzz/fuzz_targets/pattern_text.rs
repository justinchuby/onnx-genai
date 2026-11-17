//! Raw text: byte 0 selects ORT/native x inline/library; the rest is source.
//! Invalid UTF-8 cannot reach the `&str` API, so it is decoded lossily to
//! still exercise multi-byte (U+FFFD) characters at token and span edges.
#![no_main]

mod common;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((&selector, bytes)) = data.split_first() else {
        return;
    };
    match std::str::from_utf8(bytes) {
        Ok(source) => common::run(selector, source),
        Err(_) => common::run(selector, &String::from_utf8_lossy(bytes)),
    }
});
