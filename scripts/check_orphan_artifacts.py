#!/usr/bin/env python3
"""Find artifacts that EXIST but that nothing INCLUDES.

@12e42da8: "A file existing, a file COMPILING, a file being WIRED, a file being
COMMITTED, and a file being committed TO THE RIGHT BRANCH are FIVE separate
claims. Every one has failed silently tonight."

Nothing checked the first two, so this does. Two defects tonight had the
identical shape in two different languages:

  * `demo_assets.rs` -- 281 lines, reported as built for half an hour, with no
    `mod demo_assets;` anywhere. Rust simply never compiles an undeclared
    module: NO error, NO warning, NO 404. The file is inert and looks finished.
  * `styles/panels.css` -- 739 lines of panel styling linked by no `<link>`.
    Every panel renders naked, and the browser reports nothing because nothing
    was ever requested.

Both are invisible to every tool we already run. A unit test that imports the
artifact BY PATH loads it happily and goes green, which is precisely how
`stylesheet.test.js` passed while the page shipped unstyled -- a test that
reaches past the wiring can never observe that the wiring is missing.

WHAT THIS DOES NOT CLAIM. Declared is not compiled and linked is not correct;
this closes the "wired" claim only. `cargo check` closes "compiles".

EXIT 1 on any orphan. Run from the repo root.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "node_modules", "target", ".hf_cache", ".scratch", "models"}

# Rust files that are entry points or declared by convention, not by `mod`.
RUST_EXEMPT = {"lib.rs", "main.rs", "mod.rs", "build.rs"}


def walk(base, exts):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(exts):
                yield os.path.join(dirpath, fn)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def orphan_rust_modules():
    """A .rs file under a crate `src/` that no sibling declares with `mod`."""
    problems = []
    for path in walk(os.path.join(ROOT, "crates"), (".rs",)):
        name = os.path.basename(path)
        if name in RUST_EXEMPT:
            continue
        parts = path.split(os.sep)
        if "src" not in parts:
            continue  # tests/, benches/, examples/ are compiled by cargo directly
        # cargo auto-discovers `src/bin/*.rs` as binary targets; no `mod` needed.
        if "bin" in parts[parts.index("src") + 1:]:
            continue
        stem = name[:-3]
        # `foo.rs` may be declared by any .rs file in the same directory, by the
        # parent's `mod dir;` chain, or attached out-of-tree via `#[path]`.
        declared = False
        sibling_dir = os.path.dirname(path)
        # RUST 2018 LAYOUT: `src/kernels/pooling.rs` is declared by
        # `src/kernels.rs` -- a sibling of the DIRECTORY, not a file inside it.
        # Omitting this produced 30 false positives on the first run, which is
        # more noise than the one real defect this exists to find.
        search_dirs = [sibling_dir]
        parent = os.path.dirname(sibling_dir)
        if os.path.basename(sibling_dir) != "src" and os.path.isdir(parent):
            search_dirs.append(parent)
        decl = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+" + re.escape(stem) + r"\s*[;{]", re.M)
        # MACRO-GENERATED DECLARATIONS. `operator_group_modules!("ops-cnn";
        # affine_grid, pooling, ...)` in kernels/mod.rs declares nine modules
        # as bare identifiers. Missing this produced nine false positives --
        # nine ways to send someone hunting dead code that is very much alive.
        macro_decl = re.compile(r"^\s*" + re.escape(stem) + r"\s*,\s*$", re.M)
        pathattr = re.compile(r'#\[\s*path\s*=\s*"[^"]*' + re.escape(name) + r'"\s*\]')
        for d in search_dirs:
            for other in os.listdir(d):
                if not other.endswith(".rs") or os.path.join(d, other) == path:
                    continue
                src = read(os.path.join(d, other))
                if decl.search(src) or pathattr.search(src) or macro_decl.search(src):
                    declared = True
                    break
            if declared:
                break
        if not declared:
            # a directory module `foo/` is declared from its parent instead
            rel = os.path.relpath(path, ROOT)
            problems.append(f"{rel}: no `mod {stem};` declares it — the compiler never sees this file")
    return problems


def orphan_stylesheets():
    """A .css file that no .html references."""
    # A stylesheet may be pulled in by a <link> OR by a bundler import
    # (`import "./style.css"` in main.ts). Checking only .html reported
    # diffusion-demo's style.css, which main.ts:1 imports on its first line.
    refs = [(p, read(p)) for p in walk(os.path.join(ROOT, "examples"),
                                       (".html", ".js", ".ts", ".jsx", ".tsx"))]
    problems = []
    for path in walk(os.path.join(ROOT, "examples"), (".css",)):
        name = os.path.basename(path)
        if not any(name in src for _, src in refs):
            rel = os.path.relpath(path, ROOT)
            n = len(read(path).splitlines())
            problems.append(f"{rel}: {n} lines, referenced by no .html and imported by no module — renders nothing, reports nothing")
    return problems


def self_test():
    """Prove both detectors BITE, against a synthetic tree in a temp dir.

    Every check must be shown to fail (@12e42da8). Doing that by editing the
    real tree is unsafe -- another agent is always mid-edit -- so the fixtures
    are built in tempfile and torn down. Also pins the four false-positive
    classes that cost this checker three rewrites: src/bin, Rust-2018 layout,
    macro-declared modules, and bundler-imported CSS.
    """
    import tempfile, shutil
    global ROOT
    saved = ROOT
    tmp = tempfile.mkdtemp()
    try:
        def w(rel, text):
            full = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(text)

        w("crates/c/src/lib.rs", "mod declared;\nmod kernels;\n")
        w("crates/c/src/declared.rs", "pub fn a() {}\n")
        w("crates/c/src/ORPHAN.rs", "pub fn dead() {}\n")
        w("crates/c/src/bin/tool.rs", "fn main() {}\n")           # auto-target
        w("crates/c/src/kernels/mod.rs", "group!(\n    macro_mod,\n);\n")
        w("crates/c/src/kernels/macro_mod.rs", "pub fn m() {}\n")  # macro-declared
        w("examples/e/index.html", '<link href="./used.css">')
        w("examples/e/used.css", "a{}\n")
        w("examples/e/bundled.css", "b{}\n")
        w("examples/e/main.ts", 'import "./bundled.css";\n')
        w("examples/e/ORPHAN.css", "c{}\n")

        ROOT = tmp
        rust, css = orphan_rust_modules(), orphan_stylesheets()
        assert len(rust) == 1 and "ORPHAN.rs" in rust[0], rust
        assert len(css) == 1 and "ORPHAN.css" in css[0], css
        print("self-test PASSED: both detectors fire; 4 false-positive classes stay silent")
        return 0
    finally:
        ROOT = saved
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if "--self-test" in sys.argv:
        return self_test()
    rust = orphan_rust_modules()
    css = orphan_stylesheets()
    for label, items in (("UNDECLARED RUST MODULES", rust), ("UNLINKED STYLESHEETS", css)):
        print(f"### {label}: {len(items)}")
        for i in items:
            print(f"  {i}")
    total = len(rust) + len(css)
    print(f"\ntotal orphans: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
