//! Grammar-aware: each byte picks a lexeme, so mutations explore token
//! sequences (nesting, alternatives, annotations, spreads, graph lists,
//! definitions) instead of being rejected by the lexer.
#![no_main]

mod common;

use libfuzzer_sys::fuzz_target;

const LEXEMES: &[&str] = &[
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ",",
    ";",
    ":",
    "=",
    "=>",
    "->",
    "|",
    "...",
    "*",
    "<",
    ">",
    "?",
    ".",
    "::",
    "return",
    "pattern",
    "interface",
    "implements",
    "const",
    "None",
    "_",
    "tensor",
    "values",
    "slots",
    "float",
    "float16",
    "int64",
    "T",
    "N",
    "x",
    "y",
    "z",
    "w",
    "xs",
    "P",
    "Q",
    "R",
    "I",
    "J",
    "Add",
    "Mul",
    "Relu",
    "Identity",
    "Clip",
    "Dropout",
    "If",
    "Loop",
    "Split",
    "com.microsoft",
    "ai.onnx",
    "Foo",
    "alpha",
    "axis",
    "then_branch",
    "else_branch",
    "body",
    "0",
    "1",
    "2",
    "64",
    "-3",
    "2.5",
    "1e40",
    "-0.0",
    "9223372036854775808",
    "18446744073709551616",
    "\"s\"",
    "''",
    "\"\\\"\"",
    "'\u{e9}'",
    "# comment\n",
    "\n",
    "\u{e9}",
    "\u{1F600}",
];

fuzz_target!(|data: &[u8]| {
    let Some((&selector, picks)) = data.split_first() else {
        return;
    };
    let mut source = String::new();
    for &pick in picks {
        source.push_str(LEXEMES[pick as usize % LEXEMES.len()]);
        source.push(' ');
    }
    common::run(selector, &source);
});
