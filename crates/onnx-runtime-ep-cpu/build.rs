fn main() {
    const RE2_SOURCES: &[&str] = &[
        "bitstate.cc",
        "compile.cc",
        "dfa.cc",
        "filtered_re2.cc",
        "mimics_pcre.cc",
        "nfa.cc",
        "onepass.cc",
        "parse.cc",
        "perl_groups.cc",
        "prefilter.cc",
        "prefilter_tree.cc",
        "prog.cc",
        "re2.cc",
        "regexp.cc",
        "set.cc",
        "simplify.cc",
        "stringpiece.cc",
        "tostring.cc",
        "unicode_casefold.cc",
        "unicode_groups.cc",
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .define("NDEBUG", None)
        .define("NOMINMAX", None)
        .define("WIN32_LEAN_AND_MEAN", None)
        .include("vendor/re2")
        .warnings(false);
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build.flag("/EHsc").flag("/utf-8");
    } else {
        build.std("c++11");
    }
    for source in RE2_SOURCES {
        build.file(format!("vendor/re2/re2/{source}"));
    }
    build
        .file("vendor/re2/util/rune.cc")
        .file("vendor/re2/util/strutil.cc")
        .file("src/kernels/re2_bridge.cc")
        .compile("nxrt_re2");

    println!("cargo:rerun-if-changed=src/kernels/re2_bridge.cc");
    println!("cargo:rerun-if-changed=vendor/re2");
}
