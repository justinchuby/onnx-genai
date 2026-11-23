# CPU stack diagnosis of shared expert FC banks

Candidate `71b105629d3bb65e8c11a4eaba9f5e804143b92b`, baseline kernel modules
from `509144567bc6f74d899385aacee5717d20811cc0`. Native Apple M1 Max CPU,
synthetic streamed separate-FC3 BlockQuantizedMoE R8/H128/I256, four Rayon workers.
One executable/crate chooses the factory only at cold construction. Baseline and
null run identical old code in independent processes. Two fixed orders collect
six samples. All full output bytes, shape/finite checks, declared/actual retained
charges, positive/stable factory dispatch counters and final trim-to-zero agree.
The shared stable host flock covers the build and entire sampling sequence.

After warmup each child announces READY, then executes strict public session
prepare/run/output drop for a 15-second wall window. `/usr/bin/sample` targets
that exact own child PID for five seconds at a requested 1 ms interval, with
`-mayDie` symbol handling. Sampling perturbs execution and call counts are **not
throughput measurements**. Samples include waiting threads and do not equal CPU
on-core time. The top-of-stack section omits symbols with fewer than five samples;
`selected-leaves.json` copies selected raw counts from that censored list. Absence
from that list is not zero execution. Inlining differs across implementations, so
do not compare a named helper's count as a whole activation cost ratio. No CUDA,
model-throughput or full-process-memory result is claimed.

The stacks show decoded block conversion, native GEMM, and Rayon/thread waiting.
The candidate gated row helper is visible in both candidate samples. Inspecting
its actual compiled ARM64 instructions provides the actionable finding: its inner
loop still checks the ActivationBias discriminant and layout/source selection for
each element. Its mode is already known at the callers (Applied for prepared
expert projections, Pending for grouped bias folding). A static representation
can remove the bias-mode branch without adding an EP API or changing ownership,
quota calls, projection order or bank layout. This does not prove how much of the
previous wall-time regression it explains; a separate unsampled comparison is
required. Bias payload presence and shape guards must remain.

`candidate-gated-activation-assembly.txt` is the real candidate helper, selected
by excluding `probe_before` from `nm`. The first exploratory disassembly is also
retained as `gated-activation-assembly.txt`; it is the old module's similarly named
helper, not the helper executed by the old Block path. Do not confuse the two.
Both were extracted with LLVM objdump from the same recorded binary after sampling.

`instrumentation.patch` reconstructs the feature-only factory selector and old
module copies on the candidate commit. `probe.rs`, manifests, protocols, hashes,
raw compressed sampler reports/stdout/stderr/host snapshots and selected leaf
extraction preserve provenance. The binary remains `/tmp/nxrt-bank-cpu-sample/sample-probe`.
Local paths in the manifest need adjustment when reproducing. Temporary runtime
interfaces are removed after archiving. Draft #2469 remains unadopted.
