# C-locale regex bracket oracles

`oracle.tsv` (libc++) and `oracle-libstdcxx.tsv` (libstdc++) each contain the same
504 hex-encoded ECMAScript patterns. Every valid pattern is matched against all
256 one-byte strings; invalid patterns are recorded as `invalid`. A dash (`-`)
in the third column means no byte matches. Hex preserves NUL, controls
and UTF-8 spellings. The normal Rust test needs no C++ compiler.

Coverage includes ASCII byte names, named punctuation/control/digit aliases,
GNU control names, negation, mixed classes, ranges, independent equivalence/POSIX
terms, embedded NUL names, multibyte names, primary-key edge cases and malformed
delimiters. Existing trigger capture/repetition fixtures remain separate.

The actual libraries disagree: libc++ accepts direct byte names and additional
aliases; libstdc++ accepts more named controls and folds ASCII case in primary
equivalence; range-endpoint and bracket-termination validity also differ. The
implementation uses an explicit `GrammarCollation`, selected
by the nxrt `grammar_trigger_collation` request extension. Omission/null selects
`libcxx`; `libstdcxx` selects GNU classic-locale bracket behavior. No ambient
locale/compiler detection or global state controls production matching.

Generated on macOS ARM64 with Apple clang 21.0.0 (`clang-2100.4.1.1`)/libc++ and
Homebrew g++-16 16.2.0/libstdc++, C++17, explicitly using
`std::locale::classic()`. These are functional checks, not performance, Linux
execution or arbitrary-locale Unicode-collation claims. Primary references:
[LLVM names](https://github.com/llvm/llvm-project/blob/main/libcxx/src/regex.cpp),
[LLVM regex traits](https://github.com/llvm/llvm-project/blob/main/libcxx/include/regex),
[GNU names](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/regex.tcc),
[GNU primary keys](https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/regex.h).
The llama.cpp trigger baseline is `8d60873dfb8619f2b5ed31adfbab07cccb40db5c`.

Reproduce from this directory, substituting the selected compiler and fixture:

```sh
awk -F '\t' '!/^#/ { print $1 }' oracle.tsv > /tmp/llama-collation-patterns.hex
clang++ -std=c++17 oracle.cpp -o /tmp/llama-collation-oracle
/tmp/llama-collation-oracle < /tmp/llama-collation-patterns.hex > /tmp/llama-collation-actual.tsv
grep -v '^#' oracle.tsv > /tmp/llama-collation-expected.tsv
diff -u /tmp/llama-collation-expected.tsv /tmp/llama-collation-actual.tsv
```

For GNU, use `g++-16` and `oracle-libstdcxx.tsv`. Both fixtures use identical
pattern order, allowing direct comparison of names and matching sets.
