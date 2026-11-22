"""Run or reanalyze the ARM64 Erf/GELU wall diagnostic; never infer CPU savings."""

import argparse
import csv
import fcntl
import gzip
import hashlib
import io
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

TEST = "kernels::simd_activations::neon_erf::tests::neon_erf_wall_ab"
LENGTHS = (32, 257, 4096, 65537, 4194307)
FIELDS = [
    "sweep",
    "width",
    "mode",
    "op",
    "n",
    "iters",
    "rep",
    "arm",
    "wall_ns",
    "checksum",
]
LOCK = Path("/tmp/onnx-genai-hostlock-darwin.flock")

# Historical result formats are explicit measurement schemas, not runtime API
# aliases. Never infer an arm's meaning from its position or row count.
ARM_SCHEMAS = {
    1: {"0": "scalar_before", "1": "native", "2": "native_aa", "3": "mlas"},
    2: {
        "0": "scalar_before",
        "1": "native",
        "2": "native_aa",
        "3": "mlas",
        "4": "neon_guarded_before",
    },
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def analyze(raw, metadata):
    schema = metadata["schema"]
    require(schema in ARM_SCHEMAS, "unsupported diagnostic schema")
    arm_labels = ARM_SCHEMAS[schema]
    arm_count = len(arm_labels)
    if schema == 2:
        require(
            metadata["arms"] == arm_labels, "explicit arm identities differ from schema"
        )
    require(sha(raw) == metadata["raw_sha256"], "raw CSV checksum differs")
    reps, sweeps = metadata["repetitions"], metadata["sweeps"]
    require(1 <= reps <= 101 and 1 <= sweeps <= 9, "invalid repetition/sweep limits")
    axes = list(
        itertools.product(
            range(1, sweeps + 1),
            (1, 4, 8),
            ("serial", "runtime"),
            ("erf", "gelu"),
            LENGTHS,
        )
    )
    receipts = metadata["receipts"]
    require(
        len(receipts) == sweeps * 3
        and {(r["sweep"], r["width"]) for r in receipts}
        == set(itertools.product(range(1, sweeps + 1), (1, 4, 8))),
        "process receipt identities differ",
    )
    require(
        all(r["exit"] == 0 and r["cpu_s"] > 0 and r["wall_s"] > 0 for r in receipts),
        "child failed or resource receipt is empty",
    )
    checks, hashes = set(), {}
    for receipt in receipts:
        if schema == 2:
            modes = receipt["fpcr"]
            require(
                len(modes) == 2
                and modes[0] == modes[1]
                and modes[0] & ((1 << 1) | (3 << 22) | (1 << 24) | (1 << 25)) == 0,
                "FPCR proof/measurement mode differs from standard IEEE mode",
            )
        for line in receipt["validations"]:
            prefix, width, mode, op, n, arm, digest, error, dispatched, worker_tasks = (
                line.split(",")
            )
            width, n, arm, digest, error = (
                int(width),
                int(n),
                int(arm),
                int(digest),
                float(error),
            )
            key = (receipt["sweep"], width, mode, op, n, arm)
            require(
                prefix == "NEON_ERF_VALIDATE"
                and key not in checks
                and key[:-1] in axes
                and width == receipt["width"]
                and arm in range(arm_count),
                "invalid validation identity",
            )
            require(
                digest > 0 and 0 <= error <= (3e-7 if op == "erf" else 4e-7),
                "pre-timing oracle validation failed",
            )
            require(
                hashes.setdefault((op, n, arm), digest) == digest,
                "full output hash changes with pool/sweep/mode",
            )
            require(
                int(dispatched)
                == int(mode == "runtime" and width > 1 and n >= 1048576),
                "validated dispatch did not use the requested scheduling path",
            )
            require(
                (int(worker_tasks) > 0) == bool(int(dispatched)),
                "parallel validation lacks actual worker participation",
            )
            checks.add(key)
    require(
        checks == {(*axis, arm) for axis in axes for arm in range(arm_count)},
        "missing full-buffer validation",
    )
    for op, n in itertools.product(("erf", "gelu"), LENGTHS):
        require(
            hashes[(op, n, 1)] == hashes[(op, n, 2)],
            "native A/A full-buffer hashes differ",
        )
    reader = csv.DictReader(io.StringIO(raw.decode()))
    require(reader.fieldnames == FIELDS, "CSV schema differs")
    seen, groups, tokens = set(), {}, {}
    for row in reader:
        require(None not in row and None not in row.values(), "incomplete sample row")
        row = {k: v if k in ("mode", "op") else int(v) for k, v in row.items()}
        axis = tuple(row[k] for k in FIELDS[:5])
        key = (*axis, row["rep"], row["arm"])
        require(
            axis in axes
            and key not in seen
            and 0 <= row["rep"] < reps
            and row["arm"] in range(arm_count),
            "duplicate or unexpected sample identity",
        )
        require(
            row["iters"] == min(8192, max(1, 1048576 // row["n"])),
            "sample work batch differs",
        )
        require(row["wall_ns"] > 0 and row["checksum"] > 0, "empty work/timing sample")
        token_key = (row["op"], row["n"], row["arm"])
        require(
            tokens.setdefault(token_key, row["checksum"]) == row["checksum"],
            "sample output token differs across repetitions",
        )
        seen.add(key)
        groups.setdefault(axis, {arm: [] for arm in range(arm_count)})[
            row["arm"]
        ].append(row["wall_ns"] / row["iters"])
    require(
        seen
        == {
            (*axis, rep, arm)
            for axis in axes
            for rep in range(reps)
            for arm in range(arm_count)
        },
        "incomplete sample matrix",
    )
    cells = []
    for axis, arms in sorted(groups.items()):
        medians = {str(arm): statistics.median(values) for arm, values in arms.items()}
        aa = medians["1"] / medians["2"]
        cells.append(
            dict(zip(FIELDS[:5], axis, strict=True))
            | {
                "median_wall_ns": medians,
                "native_aa": aa,
                "aa_within_5pct": 0.95 <= aa <= 1.05,
                "scalar_before_over_native": medians["0"] / medians["1"],
                "mlas_over_native": medians["3"] / medians["1"],
            }
        )
        if schema == 2:
            cells[-1]["guarded_before_over_native"] = medians["4"] / medians["1"]
    for op, n in itertools.product(("erf", "gelu"), LENGTHS):
        require(
            tokens[(op, n, 1)] == tokens[(op, n, 2)], "native A/A sample tokens differ"
        )
    if schema == 2:
        for op, n in itertools.product(("erf", "gelu"), LENGTHS):
            require(
                hashes[(op, n, 4)] == hashes[(op, n, 1)],
                "guarded before/native full output hashes differ",
            )
            require(
                tokens[(op, n, 4)] == tokens[(op, n, 1)],
                "guarded before/native output tokens differ",
            )
    return {
        "schema": schema,
        "samples": len(seen),
        "validations": len(checks),
        "arms": arm_labels,
        "cells": cells,
    }


def run(directory, binary, cargo_receipt, reps, sweeps):
    require(
        sys.platform == "darwin" and platform.machine() == "arm64",
        "runner requires Darwin ARM64",
    )
    require(
        1 <= reps <= 101 and 1 <= sweeps <= 9, "use repetitions 1..101 and sweeps 1..9"
    )
    inventory = subprocess.check_output(
        [binary, TEST, "--exact", "--ignored", "--list"], text=True
    )
    require(
        inventory.count(TEST + ": test") == 1 and "1 test" in inventory,
        "exact diagnostic selector did not resolve to one test",
    )
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        "crates/onnx-runtime-ep-cpu/src/kernels/simd_activations.rs",
        "crates/onnx-runtime-ep-cpu/src/kernels/simd_activations_neon.rs",
    ]
    artifacts = [
        json.loads(line)
        for line in cargo_receipt.read_text().splitlines()
        if line.startswith("{")
    ]
    matches = [
        a
        for a in artifacts
        if a.get("reason") == "compiler-artifact"
        and a.get("executable")
        and Path(a["executable"]).resolve() == binary
    ]
    require(len(matches) == 1, "Cargo receipt must identify the exact executable once")
    artifact = matches[0]
    require(
        artifact["target"]["name"] == "onnx_runtime_ep_cpu"
        and artifact["profile"]["test"]
        and artifact["profile"]["opt_level"] == "3"
        and "mlas" in artifact["features"],
        "require an optimized MLAS lib-test Cargo artifact",
    )
    tracked = [*sources, str(Path(__file__).resolve().relative_to(root))]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *tracked],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "diff", "HEAD", "--exit-code"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    )
    require(
        not status,
        "freeze the entire checkout before measuring; commit or move pending source/artifact files",
    )
    metadata = {
        "schema": 2,
        "arms": ARM_SCHEMAS[2],
        "repetitions": reps,
        "sweeps": sweeps,
        "binary_sha256": sha(binary.read_bytes()),
        "cargo_artifact": artifact,
        "cargo_receipt_sha256": sha(cargo_receipt.read_bytes()),
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
        ).strip(),
        "source_sha256": {name: sha((root / name).read_bytes()) for name in sources},
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "platform": platform.platform(),
        "lock": str(LOCK),
        "test_inventory": inventory,
        "receipts": [],
    }
    raw = io.StringIO()
    writer = csv.writer(raw, lineterminator="\n")
    writer.writerow(FIELDS)
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sweep in range(1, sweeps + 1):
            widths = (1, 4, 8)
            shift = (sweep - 1) % 3
            widths = widths[-shift:] + widths[:-shift] if shift else widths
            for width in widths:
                name = f"sweep{sweep}-width{width}"
                out_path, err_path = (
                    directory / f"{name}.stdout",
                    directory / f"{name}.stderr",
                )
                env = os.environ | {
                    "NXRT_ERF_BENCH_WIDTH": str(width),
                    "NXRT_ERF_BENCH_REPS": str(reps),
                    "NXRT_ERF_BENCH_ORDER": str(sweep - 1),
                }
                command = [str(binary), TEST, "--exact", "--ignored", "--nocapture"]
                load, start = os.getloadavg(), time.monotonic()
                with (
                    out_path.open("wb") as out,
                    err_path.open("wb") as err,
                    subprocess.Popen(
                        command,
                        env=env,
                        stdout=out,
                        stderr=err,
                        pass_fds=(lock.fileno(),),
                    ) as child,
                ):
                    _, status, usage = os.wait4(child.pid, 0)
                    child.returncode = os.waitstatus_to_exitcode(status)
                validations = [
                    line
                    for line in err_path.read_text().splitlines()
                    if line.startswith("NEON_ERF_VALIDATE,")
                ]
                fpcr = [
                    int(line.split(",")[1])
                    for line in err_path.read_text().splitlines()
                    if line.startswith("NEON_ERF_FPCR,")
                ]
                receipt = {
                    "sweep": sweep,
                    "width": width,
                    "pid": child.pid,
                    "exit": child.returncode,
                    "wall_s": time.monotonic() - start,
                    "cpu_s": usage.ru_utime + usage.ru_stime,
                    "maxrss_bytes": usage.ru_maxrss,
                    "load_before": load,
                    "load_after": os.getloadavg(),
                    "command": command,
                    "validations": validations,
                    "fpcr": fpcr,
                }
                metadata["receipts"].append(receipt)
                (directory / "metadata.json").write_text(
                    json.dumps(metadata, indent=2) + "\n"
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in receipt.items()
                            if k not in ("validations", "command")
                        }
                    ),
                    flush=True,
                )
                require(child.returncode == 0, f"diagnostic failed: inspect {err_path}")
                output = out_path.read_text()
                require(
                    f"test {TEST} ... ok" in output,
                    "the exact diagnostic did not report a pass",
                )
                for line in output.splitlines():
                    if line.startswith("NEON_ERF_SAMPLE,"):
                        writer.writerow([sweep, *line.split(",")[1:]])
    data = raw.getvalue().encode()
    metadata["raw_sha256"] = sha(data)
    (directory / "samples.csv.gz").write_bytes(gzip.compress(data, mtime=0))
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = analyze(data, metadata)
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {result['samples']} samples and {len(result['cells'])} cells")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "analyze"))
    p.add_argument("directory", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--cargo-receipt", type=Path)
    p.add_argument("--repetitions", type=int, default=21)
    p.add_argument("--sweeps", type=int, default=3)
    args = p.parse_args()
    if args.action == "run":
        require(
            args.binary is not None and args.cargo_receipt is not None,
            "run requires --binary and its --cargo-receipt from a JSON-format release MLAS build",
        )
        run(
            args.directory.resolve(),
            args.binary.resolve(),
            args.cargo_receipt.resolve(),
            args.repetitions,
            args.sweeps,
        )
    else:
        metadata = json.loads((args.directory / "metadata.json").read_text())
        result = analyze(
            gzip.decompress((args.directory / "samples.csv.gz").read_bytes()), metadata
        )
        require(
            result == json.loads((args.directory / "results.json").read_text()),
            "saved results differ from raw-data reanalysis",
        )
        print(f"Validated {result['samples']} samples and {len(result['cells'])} cells")


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        csv.Error,
        subprocess.SubprocessError,
    ) as e:
        sys.exit(
            f"NEON Erf diagnostic failed: {e}. Inspect the input/artifacts; run again only in a fresh output directory."
        )
