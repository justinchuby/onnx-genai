//! The public C header is part of the contract, so it is tested like one.
//!
//! Two things can silently drift apart here: the sizes recorded in the header
//! and the Rust definitions, and the header and a real C compiler's opinion of
//! it. Both are checked.
//!
//! Neither test is allowed to pass by doing nothing. The layout test fails if
//! it does not find every annotation it expects, and the compile test fails
//! loudly when no C compiler can be found rather than quietly skipping.

use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::Command;

use onnx_runtime_memory_abi::*;

fn header_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("include/nxmem_memory_abi.h")
}

fn example_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/minimal_plugin.c")
}

/// Every `NXMEM_LAYOUT: <name> size=<n>` annotation in the header.
fn annotated_sizes() -> Vec<(String, usize)> {
    let header = std::fs::read_to_string(header_path()).expect("the public header is readable");
    header
        .lines()
        .filter_map(|line| {
            let rest = line.split_once("NXMEM_LAYOUT:")?.1.trim();
            let (name, size) = rest.split_once("size=")?;
            let size = size
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("malformed layout annotation: {line}"));
            Some((name.trim().to_string(), size))
        })
        .collect()
}

/// Every `NXMEM_LAYOUT_FIELD: <type>.<field> offset=<n>` annotation.
///
/// Offsets are annotated only for the structs the ABI reads by *prefix*, where
/// a field's position is load-bearing rather than incidental.
fn annotated_offsets() -> Vec<(String, String, usize)> {
    let header = std::fs::read_to_string(header_path()).expect("the public header is readable");
    header
        .lines()
        .filter_map(|line| {
            let rest = line.split_once("NXMEM_LAYOUT_FIELD:")?.1.trim();
            let (path, offset) = rest.split_once("offset=")?;
            let (type_name, field) = path.trim().split_once('.').unwrap_or_else(|| {
                panic!("malformed field annotation, expected `Type.field`: {line}")
            });
            let offset = offset
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("malformed field annotation: {line}"));
            Some((
                type_name.trim().to_string(),
                field.trim().to_string(),
                offset,
            ))
        })
        .collect()
}

/// The Rust offset of one annotated field, or `None` if this test does not
/// know the field.
///
/// Spelled out rather than derived, because the point is to notice when the
/// Rust definition moves: a table generated from the definition would agree
/// with it by construction and prove nothing.
fn rust_offset(type_name: &str, field: &str) -> Option<usize> {
    use std::mem::offset_of;
    macro_rules! table {
        ($($ty:ident { $($field:ident),+ $(,)? }),+ $(,)?) => {
            match (type_name, field) {
                $($((stringify!($ty), stringify!($field)) => Some(offset_of!($ty, $field)),)+)+
                _ => None,
            }
        };
    }
    table! {
        NxmemHostCallbacks {
            struct_size, abi_minor, host_ctx, request_reclaim, release_completed,
        },
        NxmemVirtualBackingVtable {
            struct_size, abi_minor, mechanism_id, ctx, allocate_committed, commit_range,
            decommit_range, mapped_bytes_for_ranges, mapped_bytes_for_allocation, committed_bytes,
        },
        NxmemSharedMappingVtable {
            struct_size, abi_minor, mechanism_id, ctx, create_shared_prefix, retain_shared_prefix,
            release_shared_prefix, incremental_owned_bytes, commit_shared_prefix,
        },
        NxmemAllocatorVtable {
            struct_size, abi_minor, mechanism_id, device, capability_flags, name, ctx, allocate,
            deallocate, retain, release, virtual_backing, shared_mapping, enqueue_release,
            drain_releases, pending_release_count, release_allocation,
        },
        NxmemAllocatorFactoryVtable {
            struct_size, abi_minor, name, device, capability_flags, ctx, open_allocator, release,
        },
    }
}

/// The Rust size of every type the header annotates.
///
/// Adding a `#[repr(C)]` type to the header without adding it here fails the
/// test below, which is the point: the two lists must move together.
fn rust_sizes() -> Vec<(&'static str, usize)> {
    vec![
        ("NxmemStatus", size_of::<NxmemStatus>()),
        ("NxmemDeviceId", size_of::<NxmemDeviceId>()),
        ("NxmemAllocation", size_of::<NxmemAllocation>()),
        ("NxmemByteRange", size_of::<NxmemByteRange>()),
        ("NxmemAllocRequest", size_of::<NxmemAllocRequest>()),
        ("NxmemAllocResult", size_of::<NxmemAllocResult>()),
        ("NxmemRangeRequest", size_of::<NxmemRangeRequest>()),
        ("NxmemReleaseOutcome", size_of::<NxmemReleaseOutcome>()),
        (
            "NxmemReleaseCompletion",
            size_of::<NxmemReleaseCompletion>(),
        ),
        ("NxmemReclaimRequest", size_of::<NxmemReclaimRequest>()),
        ("NxmemUnloadReport", size_of::<NxmemUnloadReport>()),
        (
            "NxmemSharedPrefixHandle",
            size_of::<NxmemSharedPrefixHandle>(),
        ),
        (
            "NxmemSharedPrefixCommitRequest",
            size_of::<NxmemSharedPrefixCommitRequest>(),
        ),
        (
            "NxmemSharedPrefixCommitInfo",
            size_of::<NxmemSharedPrefixCommitInfo>(),
        ),
        ("NxmemHostCallbacks", size_of::<NxmemHostCallbacks>()),
        ("NxmemOpenRequest", size_of::<NxmemOpenRequest>()),
        (
            "NxmemVirtualBackingVtable",
            size_of::<NxmemVirtualBackingVtable>(),
        ),
        (
            "NxmemSharedMappingVtable",
            size_of::<NxmemSharedMappingVtable>(),
        ),
        ("NxmemAllocatorVtable", size_of::<NxmemAllocatorVtable>()),
        (
            "NxmemAllocatorFactoryVtable",
            size_of::<NxmemAllocatorFactoryVtable>(),
        ),
        ("NxmemVersionRange", size_of::<NxmemVersionRange>()),
        ("NxmemNegotiateRequest", size_of::<NxmemNegotiateRequest>()),
        (
            "NxmemNegotiateResponse",
            size_of::<NxmemNegotiateResponse>(),
        ),
    ]
}

/// The header's recorded sizes match the Rust definitions.
///
/// Sizes are only equal like this on a 64-bit target, where the header's
/// pointer-sized members and Rust's agree; on anything else the annotations
/// describe a different machine and comparing them would be meaningless.
#[test]
#[cfg(target_pointer_width = "64")]
fn header_layout_matches_the_rust_definitions() {
    let annotated = annotated_sizes();
    assert!(
        annotated.len() >= 20,
        "the header lost its layout annotations: found only {}",
        annotated.len()
    );

    for (name, rust_size) in rust_sizes() {
        let (_, header_size) = annotated
            .iter()
            .find(|(annotated_name, _)| annotated_name == name)
            .unwrap_or_else(|| panic!("the header has no NXMEM_LAYOUT annotation for {name}"));
        assert_eq!(
            *header_size, rust_size,
            "{name}: the header says {header_size} bytes, Rust says {rust_size}"
        );
    }

    // And nothing is annotated that this test does not know about, so the
    // header cannot grow a type that silently goes unchecked.
    for (name, _) in &annotated {
        assert!(
            rust_sizes().iter().any(|(known, _)| known == name),
            "the header annotates {name}, which this test does not check"
        );
    }
}

/// A C compiler and the flag dialect it speaks.
struct CCompiler {
    path: PathBuf,
    /// MSVC takes `/c`, `/I`, `/Fo` where everything else takes `-c`, `-I`,
    /// `-o`. Nothing else about these invocations is platform-specific.
    msvc: bool,
}

impl CCompiler {
    /// Compile `source` to an object file, returning the compiler's stderr on
    /// failure.
    fn compile(&self, source: &Path, include_dir: &Path, out: &Path) -> Result<(), String> {
        let mut command = Command::new(&self.path);
        if self.msvc {
            command
                .arg("/c")
                .arg("/std:c11")
                .arg("/W4")
                .arg("/WX")
                .arg("/nologo")
                .arg(format!("/I{}", include_dir.display()))
                .arg(source)
                .arg(format!("/Fo{}", out.display()));
        } else {
            command
                .arg("-c")
                .arg("-std=c11")
                .arg("-Wall")
                .arg("-Wextra")
                .arg("-Werror")
                .arg("-fPIC")
                .arg("-I")
                .arg(include_dir)
                .arg(source)
                .arg("-o")
                .arg(out);
        }
        let output = command.output().expect("the C compiler runs");
        if output.status.success() {
            return Ok(());
        }
        // MSVC writes diagnostics to stdout, everyone else to stderr.
        Err(format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// The first C compiler on this machine, if any.
///
/// `cl` is tried too, and deliberately so: MSVC is the toolchain most likely
/// to disagree with Rust about `#[repr(C)]` packing, and this crate's entire
/// deliverable is a binary contract that has to hold across toolchains. A
/// layout test that never runs on the compiler most likely to break it is not
/// a layout test.
fn find_cc() -> Option<CCompiler> {
    if let Ok(cc) = std::env::var("CC")
        && !cc.is_empty()
    {
        let msvc = Path::new(&cc)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .is_some_and(|stem| stem.eq_ignore_ascii_case("cl"));
        return Some(CCompiler {
            path: PathBuf::from(cc),
            msvc,
        });
    }
    for candidate in ["cc", "clang", "gcc"] {
        if let Ok(output) = Command::new(candidate).arg("--version").output()
            && output.status.success()
        {
            return Some(CCompiler {
                path: PathBuf::from(candidate),
                msvc: false,
            });
        }
    }
    // `cl` has no `--version`; it prints a banner and its usage for `/?`.
    if let Ok(output) = Command::new("cl").arg("/?").output()
        && output.status.success()
    {
        return Some(CCompiler {
            path: PathBuf::from("cl"),
            msvc: true,
        });
    }
    None
}

/// The message every compiler-backed test fails with when there is none.
///
/// Skipping instead would let the binary contract go unchecked on exactly the
/// platform where it is most likely to be wrong, while the suite still
/// reported success.
const NO_CC: &str = "no C compiler found (tried $CC, cc, clang, gcc, cl); the public C header \
                     cannot be checked without one";

/// A real C compiler accepts the header and the example plugin.
///
/// This is what backs the "one minimal plugin example that documents the
/// contract without requiring workspace linking" claim: the example is
/// compiled with nothing on the include path but the header itself.
#[test]
fn nxmem_c_example_compiles() {
    let cc = find_cc().expect(NO_CC);
    let out = std::env::temp_dir().join(format!("nxmem_minimal_plugin_{}.o", std::process::id()));
    let result = cc.compile(
        &example_path(),
        header_path().parent().expect("include directory"),
        &out,
    );
    let _ = std::fs::remove_file(&out);
    if let Err(diagnostics) = result {
        panic!("the minimal C plugin must compile against the public header alone:\n{diagnostics}");
    }
}

/// A C compiler agrees with Rust about every annotated struct size **and every
/// annotated field offset**.
///
/// The layout test above compares Rust against a *comment*. This one compares
/// Rust against the compiler that will actually build a plugin, which is the
/// opinion that decides whether a real plugin works.
///
/// Offsets matter as much as sizes here. Prefix negotiation is built on the
/// claim that a newer struct is a byte-for-byte superset of an older one, and
/// the minimum prefix sizes are *derived* from field offsets — so a field
/// inserted in the middle keeps the total size plausible while silently moving
/// the boundary every older peer reads up to.
///
/// Not gated on Unix. These are plain `-c`/`/c` invocations, and MSVC is
/// precisely the compiler most likely to disagree.
#[test]
#[cfg(target_pointer_width = "64")]
fn a_c_compiler_agrees_with_the_rust_layouts() {
    let cc = find_cc().expect(NO_CC);

    let mut source = String::from("#include <stddef.h>\n#include \"nxmem_memory_abi.h\"\n");
    for (name, size) in rust_sizes() {
        source.push_str(&format!(
            "_Static_assert(sizeof({name}) == {size}, \"{name} disagrees with Rust\");\n"
        ));
    }
    let offsets = annotated_offsets();
    assert!(
        !offsets.is_empty(),
        "the header must annotate field offsets; finding none means the annotation format \
         changed and this test stopped checking anything"
    );
    for (type_name, field, offset) in &offsets {
        source.push_str(&format!(
            "_Static_assert(offsetof({type_name}, {field}) == {offset}, \
             \"{type_name}.{field} disagrees with the header\");\n"
        ));
    }
    source.push_str("int nxmem_layout_probe(void) { return 0; }\n");

    let dir = std::env::temp_dir().join(format!("nxmem_layout_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp directory");
    let probe = dir.join("probe.c");
    std::fs::write(&probe, source).expect("probe source is writable");

    let result = cc.compile(
        &probe,
        header_path().parent().expect("include directory"),
        &dir.join("probe.o"),
    );
    let _ = std::fs::remove_dir_all(&dir);
    if let Err(diagnostics) = result {
        panic!("a C compiler disagrees with the Rust struct layouts:\n{diagnostics}");
    }
}

/// Rust agrees with every field offset the header annotates.
///
/// The C probe above proves the *header* and a C compiler agree. This proves
/// the header and *Rust* agree, without needing a C compiler at all — so the
/// offsets stay pinned even on a machine that cannot run the probe.
#[test]
fn the_header_field_offsets_match_rust() {
    let annotated = annotated_offsets();
    assert!(
        !annotated.is_empty(),
        "the header must annotate field offsets; finding none means the annotation format \
         changed and this test stopped checking anything"
    );
    for (type_name, field, offset) in annotated {
        let actual = rust_offset(&type_name, &field).unwrap_or_else(|| {
            panic!("the header annotates {type_name}.{field}, which this test does not know")
        });
        assert_eq!(
            actual, offset,
            "{type_name}.{field}: the header says offset {offset}, Rust says {actual}"
        );
    }
}
