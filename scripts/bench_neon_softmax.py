"""Measure native ARM64 Softmax against scalar/A-A/MLAS; wall latency only."""

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

TEST = "kernels::softmax::softmax_neon::tests::neon_softmax_wall_ab"
SHAPES = ((1, 4), (1, 7), (1, 257), (32, 1024), (128, 4099), (16, 65539))
FIELDS = [
    "sweep",
    "width",
    "mode",
    "rows",
    "cols",
    "iters",
    "rep",
    "arm",
    "wall_ns",
    "checksum",
]
BASE_ARMS = {"0": "scalar_before", "1": "native", "2": "native_aa", "3": "mlas"}
ARM_SCHEMAS = {
    1: BASE_ARMS,
    2: BASE_ARMS | {"4": "neon_initial_before"},
    3: BASE_ARMS | {"4": "neon_initial_before"},
}

LOCK = Path("/tmp/onnx-genai-hostlock-darwin.flock")


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
    if schema == 3:
        require(metadata["native_neon_min_cols"] == 8, "native NEON row seam differs")
    require(
        metadata["arms"] == arm_labels,
        "explicit diagnostic schema/arms differ",
    )
    require(sha(raw) == metadata["raw_sha256"], "raw CSV checksum differs")
    reps, sweeps = metadata["repetitions"], metadata["sweeps"]
    require(1 <= reps <= 101 and 1 <= sweeps <= 9, "invalid repetition/sweep limits")
    axes = {
        (s, w, m, n, d)
        for s, w, m in itertools.product(
            range(1, sweeps + 1), (1, 4, 8), ("serial", "runtime")
        )
        for n, d in SHAPES
    }
    receipts = metadata["receipts"]
    require(
        len(receipts) == sweeps * 3
        and {(r["sweep"], r["width"]) for r in receipts}
        == set(itertools.product(range(1, sweeps + 1), (1, 4, 8))),
        "process receipt identities differ",
    )
    checks, hashes = set(), {}
    for receipt in receipts:
        require(
            receipt["exit"] == 0 and receipt["wall_s"] > 0 and receipt["cpu_s"] > 0,
            "child failed or resource receipt empty",
        )
        modes = receipt["fpcr"]
        require(
            len(modes) == 2
            and modes[0] == modes[1]
            and modes[0] & ((1 << 1) | (3 << 22) | (1 << 24) | (1 << 25)) == 0,
            "FPCR mode differs from standard IEEE mode",
        )
        for line in receipt["validations"]:
            prefix, width, mode, n, d, arm, digest, error, dispatched, workers = (
                line.split(",")
            )
            width, n, d, arm, digest, dispatched, workers = map(
                int, (width, n, d, arm, digest, dispatched, workers)
            )
            axis = (receipt["sweep"], width, mode, n, d)
            key = (*axis, arm)
            require(
                prefix == "NEON_SOFTMAX_VALIDATE"
                and axis in axes
                and key not in checks
                and width == receipt["width"]
                and arm in range(arm_count),
                "duplicate or unexpected validation identity",
            )
            require(
                0 <= float(error) <= 1e-6 and digest > 0,
                "invalid full-buffer numerical validation",
            )
            parallel = mode == "runtime" and width > 1 and n >= 2 and n * d >= 16384
            require(
                dispatched == int(parallel)
                and workers >= 0
                and (workers > 0) == parallel,
                "actual worker participation differs",
            )
            require(
                hashes.setdefault((n, d, arm), digest) == digest,
                "full output digest changes with sweep/width/mode",
            )
            checks.add(key)
    require(
        checks == {(*axis, a) for axis in axes for a in range(arm_count)},
        "missing full-buffer validation",
    )
    for n, d in SHAPES:
        require(hashes[(n, d, 1)] == hashes[(n, d, 2)], "native A/A full hashes differ")
    reader = csv.DictReader(io.StringIO(raw.decode()))
    require(reader.fieldnames == FIELDS, "CSV schema differs")
    seen, groups, tokens = set(), {}, {}
    for row in reader:
        require(None not in row and None not in row.values(), "incomplete sample row")
        row = {k: v if k == "mode" else int(v) for k, v in row.items()}
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
            row["iters"] == min(2048, max(1, 262144 // (row["rows"] * row["cols"]))),
            "sample work batch differs",
        )
        require(row["wall_ns"] > 0 and row["checksum"] > 0, "empty work/timing sample")
        token_key = (row["rows"], row["cols"], row["arm"])
        require(
            tokens.setdefault(token_key, row["checksum"]) == row["checksum"],
            "sample output tokens change",
        )
        groups.setdefault(axis, {a: [] for a in range(arm_count)})[row["arm"]].append(
            row["wall_ns"] / row["iters"]
        )
        seen.add(key)
    require(
        seen
        == {
            (*axis, r, a)
            for axis in axes
            for r in range(reps)
            for a in range(arm_count)
        },
        "incomplete sample matrix",
    )
    cells = []
    for axis, arms in sorted(groups.items()):
        medians = {str(a): statistics.median(v) for a, v in arms.items()}
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
    for n, d in SHAPES:
        require(
            tokens[(n, d, 1)] == tokens[(n, d, 2)], "native A/A sample tokens differ"
        )
    if schema >= 2:
        for cell in cells:
            cell["initial_neon_over_native"] = (
                cell["median_wall_ns"]["4"] / cell["median_wall_ns"]["1"]
            )
        for n, d in SHAPES:
            before = 0 if schema == 3 and d < 8 else 4
            require(
                hashes[(n, d, before)] == hashes[(n, d, 1)],
                "selected before/native full hashes differ",
            )
            require(
                tokens[(n, d, before)] == tokens[(n, d, 1)],
                "selected before/native tokens differ",
            )
    return {
        "schema": schema,
        "arms": arm_labels,
        "samples": len(seen),
        "validations": len(checks),
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
        "crates/onnx-runtime-ep-cpu/src/kernels/softmax.rs",
        "crates/onnx-runtime-ep-cpu/src/kernels/softmax_neon.rs",
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
        "schema": 3,
        "arms": ARM_SCHEMAS[3],
        "native_neon_min_cols": 8,
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
                    "NXRT_SOFTMAX_BENCH_WIDTH": str(width),
                    "NXRT_SOFTMAX_BENCH_REPS": str(reps),
                    "NXRT_SOFTMAX_BENCH_ORDER": str(sweep - 1),
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
                    if line.startswith("NEON_SOFTMAX_VALIDATE,")
                ]
                fpcr = [
                    int(line.split(",")[1])
                    for line in err_path.read_text().splitlines()
                    if line.startswith("NEON_SOFTMAX_FPCR,")
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
                    if line.startswith("NEON_SOFTMAX_SAMPLE,"):
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
            f"NEON Softmax diagnostic failed: {e}. Inspect the input/artifacts; run again only in a fresh output directory."
        )
