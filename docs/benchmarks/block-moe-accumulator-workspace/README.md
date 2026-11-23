# BlockQuantizedMoE accumulator in the shared execution lane

Conditions: Apple M1 Max, macOS arm64, Rust 1.98.0 / LLVM 22, release native CPU, no MLAS symbols. Synthetic MXFP4 public InferenceSession strict prepare + run + output-owner-drop; these results do not establish model throughput, RSS savings, all-format performance or CUDA behavior.

## Change and accounting boundary

The existing StepScoped layout grows from projection maximum + gathered input to projection maximum + gathered input + final accumulator. The new region is exactly rows*hidden*sizeof(f32), with no expert-count/top-k multiplier. All three regions can be live at once. Serial experts and projections keep reusing their existing regions. Empty output still needs no backing.

The final accumulator is initialized through the existing checked zeroed-prefix primitive; accumulation and final output write order are unchanged. It stays disjoint from gather and streamed projection scratch. Dense execution still leaves the projection region untouched. This removes one call-owned Vec allocation, without adding an allocation/grant interface or another checkout/return. Pool retention remains charged at actual capacity.

Routing containers, conversions, FC intermediate results, compressed strided copies and GEMM-internal storage remain outside this migration. Workspace admission is not a complete process-memory cap. The dense projection bound remains reserved for possible fallback even on a warm resident hit.

## Protocol and source

Base: `388c818906b49580870baee0a96bedf2d1dfd139`. Only block_quantized_moe.rs changes between builds. `provenance.json` records source/probe/compiler/binary hashes and zero MLAS symbols. The null executable is byte-identical to baseline. The driver restores the candidate before the timing sweep.

The full build/sweep holds the stable `/tmp/onnx-genai-hostlock-darwin.flock` shared with llama.cpp work. Three processes per arm/case use rotated arm order and nine timed batches (72 formal processes). Baseline pilots choose iteration counts targeting 100 ms CPU per measured batch. The independent Relu control runs nine batches of 100,000 calls. No formal timing cells are retried. Starting process-load snapshots are gzip-compressed beside raw logs.

A public-runtime metadata inconsistency stopped the initial pilot before any formal measurements: a legal nongated node retained seven input positions including trailing None, while build_node_plan trimmed its ordinary plan to five and the EP received inconsistent node/metadata cardinalities. The measurement probe uses the equivalent legal form omitting unused trailing slots, while preserving holes before wired inputs. The original probe, failure and provenance are retained under initial-runtime-metadata-failure. This is a runtime bug requiring a separate generic plan fix and public-session regression; EP validation must not be weakened. The runtime implementation measured here was not changed for that failure.

Qualification requires all measured/control batches >=20 ms CPU, candidate/null control medians within 5% of baseline, and measured CPU/wall ratio >=0.9. Failed cells remain in the table. These guards and the lock do not establish exclusive core scheduling on macOS.

## Results

CPU units are microseconds per strict prepare/run/drop. Values are medians of three process medians; brackets give the range. Dense means resident prepared f32 expert projections; streamed means zero dense retention. Gated wires the separate third MXFP4 projection.

| Rows/H/I | Threads | Dense | Gated | Baseline us | Candidate us | Delta | Identical-baseline delta | Gates |
|---|---:|---|---|---:|---:|---:|---:|---|
| 1/32/64 | 1 | False | False | 12.426 [12.289, 12.980] | 12.677 [12.637, 12.895] | +2.02% | -0.78% | FAIL |
| 1/64/128 | 1 | False | False | 49.187 [45.192, 49.421] | 48.937 [48.514, 49.380] | -0.51% | -3.02% | pass |
| 8/128/256 | 1 | False | False | 498.373 [491.308, 502.532] | 494.498 [490.363, 502.249] | -0.78% | -0.95% | pass |
| 1/64/128 | 1 | True | False | 12.704 [12.390, 12.887] | 12.650 [12.447, 12.878] | -0.42% | +0.13% | FAIL |
| 8/128/256 | 4 | False | False | 761.618 [677.203, 810.252] | 736.805 [689.732, 786.659] | -3.26% | -8.43% | pass |
| 8/128/256 | 4 | True | False | 384.422 [382.284, 449.763] | 377.879 [375.591, 435.181] | -1.70% | -2.24% | pass |
| 8/128/256 | 4 | False | True | 1316.538 [1236.436, 1339.641] | 1345.244 [1189.923, 1418.936] | +2.18% | -0.59% | pass |
| 8/128/256 | 4 | True | True | 637.757 [593.882, 672.618] | 616.507 [584.721, 636.912] | -3.33% | +7.79% | pass |

Complete outputs and argmaxes match within each case across all 72 processes; all outputs are finite. The manager workspace charge equals the declared physical lane requirement. The candidate adds exactly rows*hidden*4 retained bytes in every case. The probe also reconciles dense retained payloads against the actual selected experts/projections, or zero for streamed mode. Warm retained use does not grow. This does not establish lower peak RSS: formerly transient accumulation becomes retained, charged capacity until idle reclamation.

## One fixed diagnostic of the unqualified tiny cells

The original H32 streamed cell fails its null control ratio (0.94139), and the original H64 dense cell fails its CPU/wall guard (0.67520). A single diagnostic was specified before its pilots: reuse the exact existing executables, target 500 ms CPU batches instead of 100 ms, and use five independent processes per arm instead of three. No further attempts were run. The independent control and its iteration count stay unchanged. Original failed cells are retained above.

| Rows/H/I | Dense | Baseline us | Candidate us | Candidate change | Identical-baseline change | Gates |
|---|---|---:|---:|---:|---:|---|
| 1/32/64 | False | 12.094 | 12.176 | +0.679% | +0.580% | pass |
| 1/64/128 | True | 12.453 | 12.417 | -0.284% | +0.103% | pass |

Both diagnostic cells qualify. Streamed candidate change (+0.679%, about +82 ns) is close to the identical-baseline change (+0.580%, about +70 ns), and process ranges overlap; it does not establish a distinguishable regression. Dense changes -0.284% (about -35 ns), which does not establish a general speedup. The qualified original larger cases also have overlapping ranges; multi-thread identical-baseline shifts up to about 8.4% further limit their interpretation. A small cost is not excluded.

All 102 full outputs and argmaxes across the original and diagnostic studies agree within each case, including across the two studies. All retained capacity and selected dense-payload checks pass. This supports the bounded accounting change without a new EP lifecycle API; it does not prove universally zero inference overhead. The diagnostic raw batches, process snapshots, controls, hashes and ranges are under tiny-diagnostic/.

## Validation

Native BlockQuantizedMoE module: 29 tests passed; MLAS-reference build: 29 passed; neither run ignored any selected test. Listing proves the resource regression was selected, and run logs prove it executed. Native CPU all-target Clippy with -D warnings passes. The updated existing resource test checks exact complete-lane capacity, zero warm issuer callbacks including zero-byte calls, and trim-to-zero; the dense test checks projection scratch remains untouched while gather/accumulator are written. Other module tests cover all 15 formats, mixed formats, optional/gated inputs, strided tensors and empty output. No CUDA runtime tests were performed.

## Reproduction

Use the recorded base and candidate.patch, unchanged probe.rs, and driver.py. The driver creates a temporary path-dependent Cargo package with ep-api, ep-cpu, ir, loader, memory-api, memory-governor, session, libc and rayon. Adjust this host’s absolute checkout, cached ORT 1.32.0 and target paths when reproducing elsewhere. Hold the shared host lock for the entire matrix and retain failed gates. Raw logs, load snapshots, batch records, hashes and the initial runtime metadata failure are included.
