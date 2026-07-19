# TLA+ Formal Specifications

Formal models for critical distributed protocols in onnx-genai. These specs
verify safety and liveness properties that are difficult to test through
conventional unit/integration testing due to combinatorial interleavings.

## Specifications

### PressureProtocol.tla

**Verifies:** The epoch-based pressure protocol between DeviceGovernors and
HostGovernor is deadlock-free.

**Key properties:**
- `NoDeadlock` — some action is always enabled (no global stuck state)
- `NoLockDuringWait` — no lock is held while a device is in "waiting" state
- `PagesConserved` — total pages invariant (no leaks)
- `EventualSatisfaction` — every waiting device eventually gets its pages (liveness)

**Model parameters:**
- `Devices = {d0, d1, d2}` (3 devices)
- `HostCapacity = 4` (small for exhaustive search)
- `MaxRequest = 1`

### CollectiveOrdering.tla

**Verifies:** The GroupRegistry compile + DagScheduler guarantees all ranks in
a communication group submit collectives in identical order.

**Key properties:**
- `OrderConsistency` — ranks in a group never diverge by more than 1 step
- `NoSkip` — no rank submits step N+1 before completing step N-1
- `AllComplete` — all collectives eventually complete (liveness)

**Model parameters:**
- `Ranks = {r0, r1, r2, r3}` (4 ranks)
- `Groups = {g0, g1}` (2 groups with overlapping membership)
- `CollectiveSeq[g0] = <<s1, s2, s3>>`, `CollectiveSeq[g1] = <<s4, s5>>`

### BufferOwnership.tla

**Verifies:** CommHandle lifecycle ensures no buffer is reused/freed while
communication is in-flight, under arbitrary DAG scheduling orders.

**Key properties:**
- `NoPendingReuse` — buffer in "pending_comm" cannot be acquired by another op
- `NoDoubleUse` — buffer cannot be in_use by two operations simultaneously
- `FreeOnlyAfterSignal` — buffer freed only after CommHandle signals completion
- `AllOpsComplete` — all operations eventually complete (liveness)

**Model parameters:**
- `Buffers = {b0, b1, b2}` (3 buffers)
- `Operations = {op0, op1, op2, op3}` (4 operations, some sharing buffers)

## Running

Install [TLA+ tools](https://github.com/tlaplus/tlaplus/releases) or use
the VS Code TLA+ extension.

```bash
# Check with TLC model checker
java -jar tla2tools.jar -config PressureProtocol.cfg PressureProtocol.tla
```

## Design Context

These specs correspond to:
- `docs/MEMORY_ARCHITECTURE.md` §5.5 (Pressure Protocol)
- `docs/DISTRIBUTED_RUNTIME.md` §3.2 (GroupRegistry), §8.1 (ExecutionPlan DAG)
- `docs/DISTRIBUTED_RUNTIME.md` §3.1 (CommHandle buffer ownership)
