# Adaptive Scheduling & Cost Model Design

> Companion to [ORT2.md](../architecture/ORT2.md). Covers server-level scheduling, session preemption,
> pluggable cost models, and multi-dimensional resource-aware execution.

**Scope:** Dynamic resource scheduling across sessions. ORT2.md §7 covers *intra-session*
graph partitioning (static ILP at compile time). This document covers *inter-session*
scheduling decisions made at runtime by genai-server.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture: Who Decides What](#2-architecture-who-decides-what)
3. [Session Lifecycle](#3-session-lifecycle)
4. [Pluggable Cost Model](#4-pluggable-cost-model)
5. [System Context: Observable Signals](#5-system-context-observable-signals)
6. [Scheduling Policy](#6-scheduling-policy)
7. [Interaction with Memory Governor](#7-interaction-with-memory-governor)
8. [EP Negotiation Protocol](#8-ep-negotiation-protocol)
9. [genai-server Integration](#9-genai-server-integration)
10. [Multi-Plan Compilation Strategy](#10-multi-plan-compilation-strategy)
11. [Prior Art & Differentiation](#11-prior-art--differentiation)
12. [Open Questions](#12-open-questions)

---

## 1. Design Principles

1. **Sessions are static.** A compiled session is a deterministic execution plan. It does
   not adapt internally. All dynamic decisions are made *outside* the session by the
   scheduler.

2. **Cost model is pluggable.** The runtime provides observable signals; the cost function
   that maps signals → decisions is a user-replaceable trait. We ship a good default, not
   a hardcoded policy.

3. **Scheduling is proactive.** React to predicted state (thermal trajectory, queue trends),
   not just current state. By the time GPU is throttling, it's too late.

4. **Preempt, don't destroy.** Releasing a session's resources (weights paged out, scratch
   freed) is always cheaper than recompilation. The Governor handles the physical page
   management; the scheduler just changes session state.

5. **Server is the scheduler.** genai-server sees what individual sessions cannot: queue
   depth, concurrent requests, global utilization, SLA deadlines. It makes inter-session
   decisions.

6. **One model, one plan (usually).** Multi-plan variants are opt-in for specific scenarios
   (e.g., GPU-only vs CPU-fallback). The scheduler decides whether to pre-compile
   alternatives, not the user.

---

## 2. Architecture: Who Decides What

```
┌───────────────────────────────────────────────────────────────────┐
│                    External Orchestrator (k8s, etc.)               │
│   Knows: cluster topology, network, node-level resources          │
├───────────────────────────────────────────────────────────────────┤
│                    genai-server Scheduler                          │
│   Knows: all sessions, queue depth, SLA, global utilization       │
│   Decides: which sessions are active/preempted/terminated         │
│            whether to pre-compile plan variants                   │
│            request routing and priority                           │
├───────────────────────────────────────────────────────────────────┤
│                    DeviceGovernor (memory management)              │
│   Knows: physical page state, per-zone committed bytes            │
│   Decides: page eviction, commit/decommit, pressure signals       │
│   Executes: preemption actions requested by scheduler             │
├───────────────────────────────────────────────────────────────────┤
│                    nxrt Session (static, compiled)                 │
│   Knows: its own graph, placement plan, memory layout             │
│   Decides: nothing at runtime. Executes its plan deterministically│
└───────────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility | Input Signals |
|-------|---------------|---------------|
| Orchestrator | Node selection, replica scaling | Cluster metrics |
| genai-server | Session scheduling, preemption decisions, plan selection | Device metrics, queue state, SLA |
| DeviceGovernor | Physical page management, pressure signals | Committed bytes, zone budgets |
| nxrt session | Execute compiled plan | (none — static) |

**Key invariant:** A session never observes its own runtime environment. It runs when told
to run, stops when told to stop. All intelligence is in the scheduler.

**Separation of concerns:** The scheduler decides *which* session to preempt (business
logic: priority, SLA, idle time). The Governor decides *how* to reclaim pages (eviction
policy, swap vs discard). The scheduler says "preempt session X"; the Governor figures out
whether to swap KV to host or discard and recompute.

---

## 3. Session Lifecycle

```
                 compile()
    ModelDef ─────────────────► Session [READY]
                                    │
                         restore()  │    schedule()
                     ┌──────────────┤──────────────┐
                     │              ▼              │
                     │       Session [ACTIVE]      │
                     │         │         │        │
                     │   infer()│         │preempt()
                     │         ▼         ▼        │
                     │    [executing]  Session [PREEMPTED]
                     │                    │        │
                     │                    │restore()│
                     │                    └────────┘
                     │
                     │  terminate()
                     └───────────► Session [TERMINATED]
```

### States

| State | Compute Resources | Weights in Device Memory | Session State (KV/recurrent) | Compiled Graph | Can Execute |
|-------|-------------------|--------------------------|------------------------------|----------------|-------------|
| READY | Released | Not loaded | None | ✅ Retained | No (needs restore) |
| ACTIVE | Allocated | Loaded (or paging in) | On device | ✅ Retained | ✅ Yes |
| PREEMPTED | Released | Evicted (pages returned to Governor) | Swapped to host or discarded | ✅ Retained | No (needs restore) |
| TERMINATED | Released | Released | Released | Released | No (needs recompile) |

### Transitions

```rust
impl Session {
    /// Transition READY/PREEMPTED → ACTIVE.
    /// Requests Governor to page weights back in, allocates scratch/workspace.
    /// For PREEMPTED sessions: also restores session state (KV/recurrent)
    /// from host swap buffer, or triggers re-prefill if state was discarded.
    pub async fn restore(&mut self) -> Result<(), SchedulingError>;

    /// Transition ACTIVE → PREEMPTED.
    /// Releases scratch memory, Governor reclaims weight and session state pages.
    /// Session state is either swapped to host (long sequences) or discarded
    /// (short sequences where re-prefill is cheaper).
    /// Retains compiled graph (no recompilation needed on restore).
    pub async fn preempt(&mut self) -> Result<(), SchedulingError>;

    /// Transition any → TERMINATED. Releases everything.
    pub fn terminate(&mut self);

    /// Only callable in ACTIVE state.
    pub async fn infer(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, InferenceError>;
}
```

### Preemption Triggers

Preemption can be triggered by three sources:

| Trigger | Who initiates | Example |
|---------|---------------|---------|
| Governor (automatic) | Memory pressure, zone budget exceeded | KV growth exhausts budget → Governor signals scheduler → preempt lowest-priority session |
| Scheduler (automatic) | Idle timeout, thermal emergency, capacity management | Session idle 5 min → preempt to free resources for new requests |
| User (manual) | Explicit API call | User calls `scheduler.preempt(session_id)` to free specific resources |

```rust
impl Scheduler {
    /// User-initiated: preempt a specific session, freeing its resources.
    pub async fn preempt(&mut self, session_id: SessionId) -> Result<(), ScheduleError>;

    /// User-initiated: restore a preempted session.
    pub async fn restore(&mut self, session_id: SessionId) -> Result<(), ScheduleError>;

    /// User-initiated: set priority (affects automatic preemption victim selection).
    pub fn set_priority(&mut self, session_id: SessionId, priority: Priority);
}
```

### Preemption Cost Model

| Operation | Approximate Cost |
|-----------|-----------------|
| preempt (7B model, swap KV to host) | ~100ms (page-out weights + KV, async) |
| restore (7B model, from host) | ~200ms (page-in weights + KV from host) |
| restore (short seq, recompute) | ~50ms (re-prefill 512 tokens, no swap needed) |
| recompile from scratch | ~10-60s (depends on model + optimization level) |

Restore is 50-300x cheaper than recompile. Preemption is always preferable to termination
when the session might be needed again.

---

## 4. Pluggable Cost Model

> **Note:** This is the runtime scheduling cost model. For the compile-time placement cost model used by the ILP solver, see ORT2.md §6 (`PlacementCostModel`). The `SessionInfo.estimated_latency_ms` field is populated from PlacementCostModel's output during session compilation.

### Core Trait

```rust
/// A cost model evaluates how "good" a scheduling decision is.
/// The runtime provides signals; the cost model provides the value judgment.
///
/// Users can implement this trait to define custom optimization objectives.
/// The scheduler calls `evaluate` when considering plan changes.
pub trait SchedulingCostModel: Send + Sync {
    /// Evaluate the cost of running `session` given current system state.
    /// Lower is better. The scheduler picks the action that minimizes total cost.
    fn evaluate(
        &self,
        session: &SessionInfo,
        action: ScheduleAction,
        ctx: &SystemContext,
    ) -> f64;

    /// Optional: called after each inference with measured results.
    /// Enables learned/adaptive cost models that improve over time.
    // Uses interior mutability (e.g., AtomicU64, Mutex<Stats>) for thread safety
    fn observe(&self, _event: &InferenceEvent) {}

    /// Optional: proactive signal — predict cost N milliseconds into the future.
    /// Enables proactive scheduling (e.g., predict thermal throttle before it happens).
    fn predict(
        &self,
        session: &SessionInfo,
        action: ScheduleAction,
        ctx: &SystemContext,
        horizon_ms: u64,
    ) -> f64 {
        // Default: assume current state persists
        self.evaluate(session, action, ctx)
    }
}

/// What the scheduler is considering doing.
#[derive(Clone, Debug)]
pub enum ScheduleAction {
    /// Keep session active and run next inference on it.
    Execute,
    /// Preempt this session to free resources.
    Preempt,
    /// Restore a preempted session.
    Restore,
    /// Preempt this session specifically to make room for another.
    PreemptInFavorOf { beneficiary: SessionId },
}
```

### Default Implementation

```rust
/// Latency-first cost model. Suitable for single-user / interactive scenarios.
pub struct LatencySchedulingCost {
    /// Weight for queue wait time.
    pub queue_weight: f64,
    /// Weight for estimated compute time.
    pub compute_weight: f64,
    /// Weight for transition overhead (preempt/restore).
    pub transition_weight: f64,
}

impl SchedulingCostModel for LatencySchedulingCost {
    fn evaluate(&self, session: &SessionInfo, action: ScheduleAction, ctx: &SystemContext) -> f64 {
        match action {
            ScheduleAction::Execute => {
                self.compute_weight * session.estimated_latency_ms as f64
                    + self.queue_weight * ctx.queue_depth_for(session.id) as f64
            }
            ScheduleAction::Preempt => {
                self.transition_weight * session.preempt_cost_ms as f64
            }
            ScheduleAction::Restore => {
                self.transition_weight * session.restore_cost_ms as f64
            }
            ScheduleAction::PreemptInFavorOf { .. } => {
                self.transition_weight * session.preempt_cost_ms as f64 * 1.5
            }
        }
    }
}
```

### Example: Power-Aware Cost Model (Mobile)

```rust
/// For mobile/edge: balances latency against thermal headroom and battery.
pub struct PowerAwareSchedulingCost {
    pub latency_weight: f64,
    pub power_weight: f64,
    pub thermal_weight: f64,
    /// Thermal threshold (0.0-1.0) above which cost increases exponentially.
    pub thermal_cliff: f64,
}

impl SchedulingCostModel for PowerAwareSchedulingCost {
    fn evaluate(&self, session: &SessionInfo, action: ScheduleAction, ctx: &SystemContext) -> f64 {
        let base_latency = session.estimated_latency_ms as f64;
        let power_cost = ctx.estimated_power_draw_mw(session) as f64 / 1000.0;
        let thermal = ctx.thermal_headroom(); // 0.0 = cool, 1.0 = throttling

        let thermal_cost = if thermal > self.thermal_cliff {
            // Exponential penalty near throttle point
            ((thermal - self.thermal_cliff) / (1.0 - self.thermal_cliff)).powi(2) * 100.0
        } else {
            thermal
        };

        match action {
            ScheduleAction::Execute => {
                self.latency_weight * base_latency
                    + self.power_weight * power_cost
                    + self.thermal_weight * thermal_cost
            }
            _ => base_latency  // simplified
        }
    }

    fn predict(&self, session: &SessionInfo, action: ScheduleAction, ctx: &SystemContext, horizon_ms: u64) -> f64 {
        // Use thermal trajectory to predict future thermal state
        let predicted_thermal = ctx.predict_thermal(horizon_ms);
        let mut future_ctx = ctx.clone();
        future_ctx.override_thermal(predicted_thermal);
        self.evaluate(session, action, &future_ctx)
    }
}
```

### Example: Throughput-Optimized Cost Model (Server)

```rust
/// For genai-server: maximize requests/second under SLA constraints.
pub struct ThroughputSchedulingCost {
    pub sla_deadline_ms: u64,
    pub sla_violation_penalty: f64,
    pub utilization_target: f64,  // e.g., 0.85
}

impl SchedulingCostModel for ThroughputSchedulingCost {
    fn evaluate(&self, session: &SessionInfo, action: ScheduleAction, ctx: &SystemContext) -> f64 {
        match action {
            ScheduleAction::Execute => {
                let ttft = ctx.estimated_time_to_first_token(session);
                let sla_slack = self.sla_deadline_ms as f64 - ttft;
                if sla_slack < 0.0 {
                    // SLA violation — very expensive
                    -sla_slack * self.sla_violation_penalty
                } else {
                    // Under SLA: prefer high utilization
                    let util_gap = (self.utilization_target - ctx.gpu_utilization()).abs();
                    util_gap * 10.0
                }
            }
            ScheduleAction::PreemptInFavorOf { beneficiary } => {
                // Preempt is cheap if the victim is in decode (interruptible)
                // and the beneficiary is a prefill (latency-sensitive)
                let victim_interruptibility = session.decode_progress_ratio();
                let beneficiary_urgency = ctx.sla_slack_for(beneficiary);
                victim_interruptibility * 10.0 - beneficiary_urgency
            }
            _ => 0.0
        }
    }
}
```

---

## 5. System Context: Observable Signals

The scheduler provides a `SystemContext` to the cost model. This is the **read-only view**
of everything observable about the current system state.

```rust
/// Everything the cost model can observe. Updated by the scheduler's monitor loop.
pub struct SystemContext {
    // === Device-level signals ===
    pub devices: Vec<DeviceContext>,

    // === Queue-level signals ===
    pub pending_requests: usize,
    pub active_sessions: usize,
    pub preempted_sessions: usize,

    // === Historical signals (for prediction) ===
    pub thermal_history: RingBuffer<(Instant, f32)>,  // last N thermal readings
    pub utilization_history: RingBuffer<(Instant, f32)>,
    pub inference_latency_p50: Duration,
    pub inference_latency_p99: Duration,
}

pub struct DeviceContext {
    pub id: DeviceId,
    pub kind: DeviceKind,  // GPU, CPU, NPU, etc.

    // Utilization
    pub compute_utilization: f32,    // 0.0–1.0
    pub memory_utilization: f32,     // 0.0–1.0
    pub memory_used_bytes: usize,
    pub memory_total_bytes: usize,

    // Thermal
    pub temperature_celsius: f32,
    pub thermal_headroom: f32,       // 0.0 = throttling, 1.0 = cold
    pub thermal_state: ThermalState, // nominal, fair, serious, critical

    // Power
    pub power_draw_mw: u32,
    pub power_limit_mw: u32,
    pub battery_level: Option<f32>,  // mobile only, 0.0–1.0
    pub is_plugged_in: Option<bool>,

    // Bandwidth
    pub pcie_bandwidth_utilization: f32,  // host↔device transfer pressure
    pub memory_bandwidth_utilization: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThermalState {
    Nominal,   // All good
    Fair,      // Warming up
    Serious,   // Approaching throttle, should reduce load
    Critical,  // Actively throttling
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Npu,       // Neural Processing Unit (Apple ANE, Qualcomm HTP, etc.)
    Custom(u32),
}
```

### Platform-Specific Signal Sources

| Signal | Linux | macOS | Windows | Android |
|--------|-------|-------|---------|---------|
| GPU temp | NVML / sysfs | IOKit SMC | NVML / WMI | thermal HAL |
| GPU utilization | NVML | Metal perf counters | NVML / D3D12 | GPU profiler |
| CPU thermal | `/sys/class/thermal` | IOKit | WMI | thermal HAL |
| Power draw | NVML / RAPL | IOKit (SMC) | NVML | battery stats |
| Memory pressure | `/proc/meminfo` | `os_proc_available_memory` | GlobalMemoryStatusEx | ActivityManager |
| Battery | N/A | IOKit | WMI | BatteryManager |

### Monitor Trait (Platform Abstraction)

```rust
/// Abstraction over platform-specific monitoring APIs.
/// Implementations poll hardware sensors and update SystemContext.
pub trait DeviceMonitor: Send + Sync {
    /// Refresh all device signals. Called periodically by the scheduler.
    fn poll(&mut self) -> Vec<DeviceContext>;

    /// Polling interval hint. Monitor may adjust based on system load.
    fn recommended_interval(&self) -> Duration;

    /// Subscribe to critical events (thermal critical, OOM imminent).
    fn subscribe_alerts(&self) -> tokio::sync::mpsc::Receiver<DeviceAlert>;
}

pub enum DeviceAlert {
    ThermalCritical { device: DeviceId, temperature: f32 },
    MemoryPressure { device: DeviceId, available_bytes: usize },
    PowerLimit { device: DeviceId, throttle_percent: f32 },
}
```

---

## 6. Scheduling Policy

The scheduling policy is the decision-making layer that uses the cost model to determine
what to do.

```rust
/// The scheduling policy decides which sessions to activate, preempt, or terminate.
/// It consumes cost model evaluations and produces scheduling decisions.
pub trait SchedulingPolicy: Send + Sync {
    /// Called periodically (or on event) to produce scheduling decisions.
    fn schedule(
        &self,
        sessions: &[SessionInfo],
        ctx: &SystemContext,
        cost_model: &dyn SchedulingCostModel,
    ) -> Vec<ScheduleDecision>;
}

pub struct ScheduleDecision {
    pub session_id: SessionId,
    pub action: ScheduleAction,
    pub reason: ScheduleReason,
    pub priority: u32,  // execution order if multiple decisions
}

#[derive(Clone, Debug)]
pub enum ScheduleReason {
    /// Normal request routing
    RequestArrived { request_id: RequestId },
    /// Memory pressure from Governor — need to free resources
    MemoryPressure { device: DeviceId, needed_bytes: usize },
    /// Thermal management
    ThermalManagement { device: DeviceId, state: ThermalState },
    /// SLA deadline approaching
    SlaUrgency { request_id: RequestId, slack_ms: i64 },
    /// Idle timeout — no requests for this session in a while
    IdleTimeout { idle_duration: Duration },
    /// Manual override from user/operator
    UserRequested,
}

pub struct SessionInfo {
    pub id: SessionId,
    pub state: SessionState,
    pub model_id: ModelId,
    pub plan_variant: PlanVariant,

    // Resource footprint
    pub weight_size_bytes: usize,
    pub workspace_size_bytes: usize,
    pub session_state_bytes: usize,  // KV cache + recurrent state
    pub device_affinity: DeviceId,

    // Performance characteristics (from compilation / profiling)
    pub estimated_latency_ms: u64,
    pub preempt_cost_ms: u64,   // time to page-out weights + swap state
    pub restore_cost_ms: u64,   // time to page-in weights + restore state

    // Runtime state
    pub last_active: Instant,
    pub total_inferences: u64,
    pub current_request: Option<RequestId>,
    pub priority: Priority,
    pub sequence_length: usize,  // affects swap-vs-recompute decision
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Background,
    Normal,
    High,
    Critical,
}
```

### Default Policy: Priority Queue with Backpressure

```rust
pub struct DefaultSchedulingPolicy {
    /// Max concurrent active sessions (resource budget).
    pub max_active: usize,
    /// Idle timeout before auto-preempt.
    pub idle_timeout: Duration,
    /// Thermal threshold that triggers preemption.
    pub thermal_preempt_threshold: ThermalState,
}

impl SchedulingPolicy for DefaultSchedulingPolicy {
    fn schedule(
        &self,
        sessions: &[SessionInfo],
        ctx: &SystemContext,
        cost_model: &dyn SchedulingCostModel,
    ) -> Vec<ScheduleDecision> {
        let mut decisions = Vec::new();

        // 1. Thermal emergency: preempt sessions on overheating devices
        for session in sessions.iter().filter(|s| s.state == SessionState::Active) {
            let device_ctx = ctx.device(session.device_affinity);
            if device_ctx.thermal_state >= self.thermal_preempt_threshold {
                decisions.push(ScheduleDecision {
                    session_id: session.id,
                    action: ScheduleAction::Preempt,
                    reason: ScheduleReason::ThermalManagement {
                        device: session.device_affinity,
                        state: device_ctx.thermal_state,
                    },
                    priority: 0, // highest priority
                });
            }
        }

        // 2. Idle timeout: preempt sessions with no recent activity
        for session in sessions.iter().filter(|s| s.state == SessionState::Active) {
            if session.last_active.elapsed() > self.idle_timeout {
                decisions.push(ScheduleDecision {
                    session_id: session.id,
                    action: ScheduleAction::Preempt,
                    reason: ScheduleReason::IdleTimeout {
                        idle_duration: session.last_active.elapsed(),
                    },
                    priority: 10,
                });
            }
        }

        // 3. Capacity management: if over max_active, preempt lowest-value sessions
        let active_count = sessions.iter().filter(|s| s.state == SessionState::Active).count();
        if active_count > self.max_active {
            let mut active: Vec<_> = sessions.iter()
                .filter(|s| s.state == SessionState::Active)
                .collect();
            // Sort by cost of keeping active (highest cost = first to preempt)
            active.sort_by(|a, b| {
                let cost_a = cost_model.evaluate(a, ScheduleAction::Execute, ctx);
                let cost_b = cost_model.evaluate(b, ScheduleAction::Execute, ctx);
                cost_b.partial_cmp(&cost_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            for session in active.iter().take(active_count - self.max_active) {
                decisions.push(ScheduleDecision {
                    session_id: session.id,
                    action: ScheduleAction::Preempt,
                    reason: ScheduleReason::MemoryPressure {
                        device: session.device_affinity,
                        needed_bytes: 0, // capacity-driven, not byte-driven
                    },
                    priority: 20,
                });
            }
        }

        // 4. Restore sessions needed for pending requests
        // (handled by request router, not periodic scheduling)

        decisions.sort_by_key(|d| d.priority);
        decisions
    }
}
```

---

## 7. Interaction with Memory Governor

Session preemption and restore are coordinated between the Scheduler and the DeviceGovernor.
The Scheduler decides *who* to preempt; the Governor decides *how* to handle the pages.

### Scheduler → Governor Communication

```rust
/// Governor interface for session-level resource management.
impl DeviceGovernor {
    /// Scheduler requests: "free all resources held by this session."
    /// Governor decides: swap KV to host (long seq) or discard (short seq).
    pub async fn release_session(&mut self, session_id: SessionId) -> ReleaseReport;

    /// Scheduler requests: "restore this session's resources."
    /// Governor pages weights back in and restores session state.
    pub async fn restore_session(&mut self, session_id: SessionId) -> RestoreReport;

    /// Governor → Scheduler signal: "I'm under pressure, need N bytes freed."
    /// Scheduler picks the victim session(s).
    pub fn pressure_signal(&self) -> Option<PressureSignal>;
}

pub struct PressureSignal {
    pub device: DeviceId,
    pub needed_bytes: usize,
    pub urgency: PressureUrgency,
}

pub enum PressureUrgency {
    /// Can wait — preempt at next scheduling tick
    Low,
    /// Should act soon — new request queued but can't start
    Medium,
    /// Must act now — active session's KV growth is blocked
    High,
}

pub struct ReleaseReport {
    pub freed_bytes: usize,
    pub state_action: StateAction,
}

pub enum StateAction {
    /// Session state (KV/recurrent) was swapped to host — can restore cheaply
    SwappedToHost { host_buffer_bytes: usize },
    /// Session state was discarded — must re-prefill on restore
    Discarded { recompute_tokens: usize },
}
```

### Decision Flow

```
Governor detects pressure → signals Scheduler
    ↓
Scheduler evaluates cost model for each active session
    ↓
Scheduler selects victim (lowest priority / highest preempt cost-benefit)
    ↓
Scheduler calls session.preempt()
    ↓
Session calls Governor.release_session()
    ↓
Governor decides swap-vs-discard based on:
  - sequence_length vs recompute_threshold
  - available host memory for swap buffer
  - data type (KV → swap or recompute; recurrent state → always swap)
    ↓
Governor frees physical pages → available for other sessions
```

### Weight Pages: Discard Always

Weight pages are always discarded on preemption (never swapped D2H) because the original
bytes are permanently available via host mmap of the model file. Restore = H2D copy from
the existing host mapping. No D2H save step needed.

### Session State: Swap or Recompute

| Condition | Action | Rationale |
|-----------|--------|-----------|
| `seq_len < recompute_threshold` | Discard + re-prefill on restore | Recompute is faster than D2H + H2D |
| `seq_len >= recompute_threshold` | Swap to host | D2H + H2D cheaper than re-prefill |
| Recurrent state (any size) | Swap to host | Cannot recompute (non-linear accumulation) |
| Host memory insufficient | Discard (forced) | No room to swap; accept re-prefill cost |

The threshold is hardware-dependent:
```
recompute_threshold ≈ (swap_round_trip_bytes / pcie_bandwidth) / per_token_prefill_time

Example (PCIe Gen4 x16, ~25 GB/s, 4090):
  1 GB KV swap round-trip ≈ 80ms
  ~400 tokens prefill ≈ 80ms
  → threshold ≈ 400 tokens
```

---

## 8. EP Negotiation Protocol

The EP negotiation protocol (conditional support, prerequisites) is defined in ORT2.md §4.
Key extensions beyond basic KernelMatch:

- `KernelMatch::ConditionalSupport` — EP can run the op IF runtime applies specified transforms
- `Prerequisite` enum — FuseOps, CastInput, DecomposeOp, CustomTransform, MinBatchSize

The ILP solver accounts for prerequisite costs. See ORT2.md §4 for full API definitions.

---

## 9. genai-server Integration

genai-server is the primary consumer of the scheduling API. It sits between incoming
requests and nxrt sessions.

```
                    Incoming Requests
                         │
                         ▼
              ┌─────────────────────┐
              │   Request Router    │
              │   (SLA, priority)   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Scheduler         │  ← SchedulingPolicy + CostModel
              │   (session mgmt)    │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Session A│   │ Session B│   │ Session C│
    │ [ACTIVE] │   │ [ACTIVE] │   │[PREEMPTED]│
    └──────────┘   └──────────┘   └──────────┘
```

### Server → Scheduler Communication

```rust
/// The server-facing API for the scheduler.
pub trait Scheduler {
    /// Register a new session with the scheduler.
    fn register_session(&mut self, session: SessionHandle, info: SessionInfo);

    /// Request: "I need to run inference on this model. Which session should I use?"
    /// Scheduler may restore a preempted session, or reject if overloaded.
    async fn acquire(&self, model_id: ModelId, request: &RequestInfo) -> Result<SessionHandle, ScheduleError>;

    /// Release: "I'm done with this session for now."
    fn release(&self, session: SessionHandle);

    /// Periodic tick: let the scheduler make proactive decisions.
    async fn tick(&mut self, ctx: &SystemContext) -> Vec<ScheduleDecision>;

    /// Swap cost model at runtime (e.g., switching from interactive to batch mode).
    fn set_cost_model(&mut self, cost_model: Box<dyn SchedulingCostModel>);

    /// Swap scheduling policy at runtime.
    fn set_policy(&mut self, policy: Box<dyn SchedulingPolicy>);
}
```

### Dynamic Resource Limits

genai-server can adjust resource limits at runtime (e.g., when a new model is loaded or
an operator scales down GPU allocation):

```rust
/// Server tells scheduler: "your resource budget changed."
pub struct ResourceBudget {
    pub max_gpu_memory_bytes: usize,
    pub max_active_sessions: usize,
    pub max_power_draw_mw: Option<u32>,
    pub priority_class: PriorityClass,
}

impl Scheduler {
    /// Update resource budget. Scheduler will preempt/terminate sessions to fit.
    fn update_budget(&mut self, budget: ResourceBudget);
}
```

---

## 10. Multi-Plan Compilation Strategy

### When to Compile Multiple Plans

One model usually needs one plan. Multiple plans are justified when:

1. **Heterogeneous fallback:** Primary plan uses GPU; fallback uses CPU when GPU is
   unavailable (thermal, shared with higher-priority workload).
2. **Batch size specialization:** One plan optimized for batch=1 (interactive), another
   for batch=32 (throughput).
3. **Power profiles:** Full-speed plan vs power-saver plan (different EP selections,
   different precision).

### Who Decides

The scheduler (or server admin) requests plan variants, not the user:

```rust
pub struct PlanVariant {
    pub id: PlanVariantId,
    pub label: String,  // e.g., "gpu-full", "cpu-fallback", "power-saver"
    pub constraints: CompilationConstraints,
}

pub struct CompilationConstraints {
    /// Restrict to these devices only.
    pub allowed_devices: Option<Vec<DeviceId>>,
    /// Target batch size for optimization.
    pub target_batch_size: Option<usize>,
    /// Power budget constraint (affects EP selection and precision).
    pub max_power_mw: Option<u32>,
    /// Precision constraint.
    pub min_precision: Option<DataType>,
}
```

### Switching Between Plans

Plan switching = preempt current session + restore alternative session:

```rust
// Scheduler decides GPU is overheating, switch model_A from gpu-plan to cpu-plan
scheduler.preempt(model_a_gpu_session).await?;
scheduler.restore(model_a_cpu_session).await?;
// Next request for model_A routes to cpu session
```

No special "plan switching" mechanism needed — it's just session lifecycle management.
The scheduler's cost model determines when switching is worthwhile (transition cost vs
continued degraded performance).

---

## 11. Prior Art & Differentiation

| System | What It Has | What It Lacks |
|--------|-------------|---------------|
| **vLLM** | KV cache swap/recompute preemption | General session preemption; pluggable cost; thermal awareness |
| **TensorRT** | Multi-profile engines, runtime profile switch | Automatic switching; cost model; thermal/power signals |
| **Core ML** | Compute unit hints; thermal state API | Dynamic switching without reload; scheduling policy |
| **QNN/SNPE** | Performance profiles (DVFS control) | Multi-plan; pluggable cost; automatic switching |
| **TFLite** | Pluggable delegate interface | Scheduling; runtime switching; cost model |

Note: The paged attention design assumes Mobius generates models with ScatterND/GatherElements for KV access (not contiguous past_key_values). Since Mobius is under our control, the runtime can assume optimal model structure.

### NXRT's Unique Combination

1. **Pluggable cost model** — user defines what "optimal" means (latency, power, thermal,
   throughput, or any combination). No other runtime offers this.
2. **Automatic proactive scheduling** — predict thermal trajectory and act before throttle.
   No other runtime does this automatically.
3. **Governor-driven preemption** — unified memory management where physical pages flow
   between sessions and zones. Goes beyond vLLM's KV-cache-only swap to full session
   state management.
4. **Server-level intelligence, session-level simplicity** — clean separation that keeps
   nxrt sessions deterministic and debuggable while enabling sophisticated scheduling.
5. **Three preemption triggers** — automatic (Governor pressure, scheduler policy) and
   manual (user API) all funnel through the same lifecycle state machine.

---

## 12. Open Questions

1. **Feedback loop frequency:** How often should `CostModel::observe()` be called? Every
   inference? Every N inferences? On significant state change only?

2. **Learned cost models:** Can we ship a cost model that learns from historical data
   (e.g., simple regression on latency vs utilization)? What's the cold-start story?

3. **Plan variant budget:** How many pre-compiled plans per model is reasonable? Storage
   and compilation time cost. Is 2-3 enough for most cases?

4. **Progressive restore ordering:** How to determine "critical pages" for early restore?
   First N layers? Or profiling-guided (which layers are hit first)?

5. **Cross-session weight sharing:** If two sessions (same model, different plans) share
   weights, can the Governor deduplicate? Saves memory but complicates page ownership.

6. **Scheduler tick frequency:** How often should the scheduler's periodic `tick()` run?
   Too fast = overhead. Too slow = missed thermal events. Adaptive frequency?

7. **Multi-GPU scheduling:** When model spans multiple GPUs (tensor parallel), does each
   GPU-slice count as a separate session? Or one session with multi-device plan?

8. **Migration vs restart:** If a model needs to move from GPU A to GPU B (e.g., GPU A
   overheating), is it cheaper to preempt+restore-on-B, or recompile-for-B?

9. **SLA specification format:** How do users express SLA requirements? Simple deadline_ms?
   Or richer (p99 < 100ms, throughput > 10 req/s)?

10. **Scheduler fairness:** With multiple tenants, how to prevent one tenant's sessions from
    starving another? Priority classes? Weighted fair scheduling?

---

## Appendix A: Signal Collection Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                 DeviceMonitor                                      │
│                                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐              │
│  │ NVML Plugin │  │ sysfs Plugin │  │ IOKit Plugin│              │
│  │ (NVIDIA GPU)│  │ (Linux CPU)  │  │ (macOS)    │              │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘              │
│         └────────────────┼─────────────────┘                     │
│                          ▼                                        │
│              SystemContext (unified)                               │
│                          │                                        │
└──────────────────────────┼────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       CostModel    SchedulingPolicy   Alerts
       .evaluate()  .schedule()        (thermal critical, OOM)
```
