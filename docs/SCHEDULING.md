# Adaptive Scheduling & Cost Model Design

> Companion to [ORT2.md](./ORT2.md). Covers server-level scheduling, session hibernation,
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
7. [Interaction with Paged Memory](#7-interaction-with-paged-memory)
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

4. **Hibernate, don't destroy.** Suspending a session (offload state, release compute
   resources) is always cheaper than recompilation. Leverage paged memory for weight
   offload.

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
│   Decides: which sessions are active/hibernated/terminated        │
│            whether to pre-compile plan variants                   │
│            request routing and priority                           │
├───────────────────────────────────────────────────────────────────┤
│                    nxrt Session (static, compiled)                 │
│   Knows: its own graph, placement plan, memory layout             │
│   Decides: nothing at runtime. Executes its plan deterministically│
└───────────────────────────────────────────────────────────────────┘
```

| Layer | Responsibility | Input Signals |
|-------|---------------|---------------|
| Orchestrator | Node selection, replica scaling | Cluster metrics |
| genai-server | Session scheduling, hibernation, plan selection | Device metrics, queue state, SLA |
| nxrt session | Execute compiled plan | (none — static) |

**Key invariant:** A session never observes its own runtime environment. It runs when told
to run, sleeps when told to sleep. All intelligence is in the scheduler.

---

## 3. Session Lifecycle

```
                 compile()
    ModelDef ─────────────────► Session [READY]
                                    │
                          wake()    │    schedule()
                     ┌──────────────┤──────────────┐
                     │              ▼              │
                     │       Session [ACTIVE]      │
                     │         │         │        │
                     │   infer()│         │hibernate()
                     │         ▼         ▼        │
                     │    [executing]  Session [HIBERNATED]
                     │                    │        │
                     │                    │ wake() │
                     │                    └────────┘
                     │
                     │  terminate()
                     └───────────► Session [TERMINATED]
```

### States

| State | Compute Resources | Weights in Device Memory | Compiled Graph | Can Execute |
|-------|-------------------|--------------------------|----------------|-------------|
| READY | Released | Not loaded | ✅ Retained | No (needs wake) |
| ACTIVE | Allocated | Loaded (or paged in) | ✅ Retained | ✅ Yes |
| HIBERNATED | Released | Offloaded (paged out) | ✅ Retained | No (needs wake) |
| TERMINATED | Released | Released | Released | No (needs recompile) |

### Transitions

```rust
impl Session {
    /// Transition READY/HIBERNATED → ACTIVE.
    /// Pages weights back in, allocates scratch/workspace memory.
    pub async fn wake(&mut self) -> Result<(), SchedulingError>;

    /// Transition ACTIVE → HIBERNATED.
    /// Releases scratch memory, triggers weight page-out via paged memory system.
    /// Retains compiled graph (no recompilation needed on wake).
    pub async fn hibernate(&mut self) -> Result<(), SchedulingError>;

    /// Transition any → TERMINATED. Releases everything.
    pub fn terminate(&mut self);

    /// Only callable in ACTIVE state.
    pub async fn infer(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, InferenceError>;
}
```

### Hibernate Cost Model

| Operation | Approximate Cost |
|-----------|-----------------|
| hibernate (7B model, fp16) | ~100ms (page-out 14GB to host, async) |
| wake (7B model, fp16) | ~200ms (page-in 14GB from host) |
| wake (from disk/NVMe) | ~2-4s (14GB at 5-7 GB/s) |
| recompile from scratch | ~10-60s (depends on model + optimization level) |

Wake is 50-300x cheaper than recompile. Hibernation is always preferable to termination
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
    /// Hibernate this session to free resources.
    Hibernate,
    /// Wake a hibernated session.
    Wake,
    /// Preempt this session to make room for another.
    Preempt { in_favor_of: SessionId },
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
    /// Weight for transition overhead (wake/hibernate).
    pub transition_weight: f64,
}

impl SchedulingCostModel for LatencySchedulingCost {
    fn evaluate(&self, session: &SessionInfo, action: ScheduleAction, ctx: &SystemContext) -> f64 {
        match action {
            ScheduleAction::Execute => {
                self.compute_weight * session.estimated_latency_ms as f64
                    + self.queue_weight * ctx.queue_depth_for(session.id) as f64
            }
            ScheduleAction::Hibernate => {
                self.transition_weight * session.hibernate_cost_ms as f64
            }
            ScheduleAction::Wake => {
                self.transition_weight * session.wake_cost_ms as f64
            }
            ScheduleAction::Preempt { .. } => {
                self.transition_weight * session.hibernate_cost_ms as f64 * 1.5
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
            // ... other actions
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
            ScheduleAction::Preempt { in_favor_of } => {
                // Preempt is cheap if the victim is in decode (interruptible)
                // and the beneficiary is a prefill (latency-sensitive)
                let victim_interruptibility = session.decode_progress_ratio();
                let beneficiary_urgency = ctx.sla_slack_for(in_favor_of);
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
    pub hibernated_sessions: usize,

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
/// The scheduling policy decides which sessions to activate, hibernate, or preempt.
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
    /// Resource pressure — need to free resources
    ResourcePressure { device: DeviceId, pressure: f32 },
    /// Thermal management
    ThermalManagement { device: DeviceId, state: ThermalState },
    /// SLA deadline approaching
    SlaUrgency { request_id: RequestId, slack_ms: i64 },
    /// Idle timeout — no requests for this session in a while
    IdleTimeout { idle_duration: Duration },
    /// Manual override from operator
    ManualOverride,
}

pub struct SessionInfo {
    pub id: SessionId,
    pub state: SessionState,
    pub model_id: ModelId,
    pub plan_variant: PlanVariant,

    // Resource footprint
    pub weight_size_bytes: usize,
    pub workspace_size_bytes: usize,
    pub device_affinity: DeviceId,

    // Performance characteristics (from compilation / profiling)
    // These estimates are populated during session compilation:
    // - estimated_latency_ms: from PlacementCostModel (ORT2.md §6) applied to the compiled plan
    // - hibernate_cost_ms: weight_size_bytes / host_bandwidth (from pager stats)
    // - wake_cost_ms: weight_size_bytes / device_bandwidth (from pager stats)
    pub estimated_latency_ms: u64,
    pub hibernate_cost_ms: u64,
    pub wake_cost_ms: u64,

    // Runtime state
    pub last_active: Instant,
    pub total_inferences: u64,
    pub current_request: Option<RequestId>,
}
```

### Default Policy: Priority Queue with Backpressure

```rust
pub struct DefaultSchedulingPolicy {
    /// Max concurrent active sessions (resource budget).
    pub max_active: usize,
    /// Idle timeout before auto-hibernate.
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

        // 1. Thermal emergency: hibernate sessions on overheating devices
        for session in sessions.iter().filter(|s| s.state == SessionState::Active) {
            let device_ctx = ctx.device(session.device_affinity);
            if device_ctx.thermal_state >= self.thermal_preempt_threshold {
                decisions.push(ScheduleDecision {
                    session_id: session.id,
                    action: ScheduleAction::Hibernate,
                    reason: ScheduleReason::ThermalManagement {
                        device: session.device_affinity,
                        state: device_ctx.thermal_state,
                    },
                    priority: 0, // highest priority
                });
            }
        }

        // 2. Idle timeout: hibernate sessions with no recent activity
        for session in sessions.iter().filter(|s| s.state == SessionState::Active) {
            if session.last_active.elapsed() > self.idle_timeout {
                decisions.push(ScheduleDecision {
                    session_id: session.id,
                    action: ScheduleAction::Hibernate,
                    reason: ScheduleReason::IdleTimeout {
                        idle_duration: session.last_active.elapsed(),
                    },
                    priority: 10,
                });
            }
        }

        // 3. Capacity management: if over max_active, hibernate lowest-value sessions
        let active_count = sessions.iter().filter(|s| s.state == SessionState::Active).count();
        if active_count > self.max_active {
            let mut active: Vec<_> = sessions.iter()
                .filter(|s| s.state == SessionState::Active)
                .collect();
            // Sort by cost of keeping active (highest cost = first to hibernate)
            active.sort_by(|a, b| {
                let cost_a = cost_model.evaluate(a, ScheduleAction::Execute, ctx);
                let cost_b = cost_model.evaluate(b, ScheduleAction::Execute, ctx);
                cost_b.partial_cmp(&cost_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            for session in active.iter().take(active_count - self.max_active) {
                decisions.push(ScheduleDecision {
                    session_id: session.id,
                    action: ScheduleAction::Hibernate,
                    reason: ScheduleReason::ResourcePressure {
                        device: session.device_affinity,
                        pressure: ctx.device(session.device_affinity).compute_utilization,
                    },
                    priority: 20,
                });
            }
        }

        // 4. Wake sessions needed for pending requests
        // (handled by request router, not periodic scheduling)

        decisions.sort_by_key(|d| d.priority);
        decisions
    }
}
```

---

## 7. Interaction with Paged Memory

Session hibernate/wake uses nxrt's PagerSchedulerAPI (defined in ORT2.md §33). The key methods are `deprioritize_session`, `prioritize_session`, and `await_critical_resident`.

Session hibernation leverages nxrt's paged memory system (ORT2.md §8) rather than
implementing its own offload mechanism.

```
Session ACTIVE:
  weights: pages resident in device memory (GPU VRAM)
  workspace: allocated from arena allocator
  compiled graph: in host memory (always resident)

Session HIBERNATED:
  weights: pages evicted (host memory or disk, managed by pager)
  workspace: freed back to arena
  compiled graph: in host memory (always resident, cheap)

Session wake():
  1. Allocate workspace from arena
  2. Request pager to page-in weight pages (async, overlapped with compute if possible)
  3. Once all pages resident → state = ACTIVE
```

**Key insight:** We don't need a separate hibernation data path. The paged memory system
already knows how to evict and restore pages. Hibernate is just "evict all my pages" and
wake is "page them back in." The scheduler becomes a *hint provider* to the pager:
"this session is going to sleep, feel free to evict its pages for higher-priority sessions."

```rust
impl Session {
    pub async fn hibernate(&mut self) -> Result<(), SchedulingError> {
        // Release workspace (immediate)
        self.arena.release_workspace();

        // Hint to pager: deprioritize all our weight pages
        self.pager.deprioritize_all(self.weight_page_set).await;

        // Pages will be evicted lazily when memory pressure demands it,
        // or eagerly if another session needs the space.
        self.state = SessionState::Hibernated;
        Ok(())
    }

    pub async fn wake(&mut self) -> Result<(), SchedulingError> {
        // Allocate workspace
        self.arena.allocate_workspace(self.workspace_size)?;

        // Request page-in for all weight pages (async)
        self.pager.prioritize_all(self.weight_page_set).await?;

        // Wait for critical pages (first few layers) before marking active
        self.pager.await_resident(self.critical_pages).await?;

        self.state = SessionState::Active;
        Ok(())
    }
}
```

### Progressive Wake (Optimization)

For large models, don't wait for all pages before starting inference:

1. Page in layers 0–N (critical path for first token)
2. Mark session ACTIVE, begin inference
3. Continue paging remaining layers in background
4. If inference reaches a not-yet-paged layer → block on that page (rare with good prefetch)

This reduces wake latency from "page entire model" to "page first few layers."

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
    │ [ACTIVE] │   │ [ACTIVE] │   │[HIBERNATED]│
    └──────────┘   └──────────┘   └──────────┘
```

### Server → Scheduler Communication

```rust
/// The server-facing API for the scheduler.
pub trait Scheduler {
    /// Register a new session with the scheduler.
    fn register_session(&mut self, session: SessionHandle, info: SessionInfo);

    /// Request: "I need to run inference on this model. Which session should I use?"
    /// Scheduler may wake a hibernated session, or reject if overloaded.
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
    /// Update resource budget. Scheduler will hibernate/terminate sessions to fit.
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

Plan switching = hibernate current session + wake alternative session:

```rust
// Scheduler decides GPU is overheating, switch model_A from gpu-plan to cpu-plan
scheduler.hibernate(model_a_gpu_session).await?;
scheduler.wake(model_a_cpu_session).await?;
// Next request for model_A routes to cpu session
```

No special "plan switching" mechanism needed — it's just session lifecycle management.
The scheduler's cost model determines when switching is worthwhile (transition cost vs
continued degraded performance).

---

## 11. Prior Art & Differentiation

| System | What It Has | What It Lacks |
|--------|-------------|---------------|
| **vLLM** | KV cache swap/recompute preemption | General session hibernate; pluggable cost; thermal awareness |
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
3. **Session hibernation via paged memory** — general-purpose compiled session
   suspend/resume without recompilation. Goes beyond vLLM's KV-cache-only swap.
4. **Server-level intelligence, session-level simplicity** — clean separation that keeps
   nxrt sessions deterministic and debuggable while enabling sophisticated scheduling.

---

## 12. Open Questions

1. **Feedback loop frequency:** How often should `CostModel::observe()` be called? Every
   inference? Every N inferences? On significant state change only?

2. **Learned cost models:** Can we ship a cost model that learns from historical data
   (e.g., simple regression on latency vs utilization)? What's the cold-start story?

3. **Plan variant budget:** How many pre-compiled plans per model is reasonable? Storage
   and compilation time cost. Is 2-3 enough for most cases?

4. **Progressive wake ordering:** How to determine "critical pages" for early wake? First
   N layers? Or profiling-guided (which layers are hit first)?

5. **Cross-session weight sharing:** If two sessions (same model, different plans) share
   weights, can the pager deduplicate? Saves memory but complicates page ownership.

6. **Scheduler tick frequency:** How often should the scheduler's periodic `tick()` run?
   Too fast = overhead. Too slow = missed thermal events. Adaptive frequency?

7. **Multi-GPU scheduling:** When model spans multiple GPUs (tensor parallel), does each
   GPU-slice count as a separate session? Or one session with multi-device plan?

8. **Migration vs restart:** If a model needs to move from GPU A to GPU B (e.g., GPU A
   overheating), is it cheaper to hibernate+wake-on-B, or recompile-for-B?

9. **SLA specification format:** How do users express SLA requirements? Simple deadline_ms?
   Or richer (p99 < 100ms, throughput > 10 req/s)?

10. **Scheduler fairness:** With multiple tenants, how to prevent one tenant's sessions from
    starving another? Priority classes? Weighted fair scheduling?

---

## Appendix A: Signal Collection Architecture

```
┌──────────────────────────────────────────────────────┐
│                 DeviceMonitor                          │
│                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ NVML Plugin │  │ sysfs Plugin │  │ IOKit Plugin│  │
│  │ (NVIDIA GPU)│  │ (Linux CPU)  │  │ (macOS)    │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │
│         └────────────────┼─────────────────┘         │
│                          ▼                            │
│              SystemContext (unified)                   │
│                          │                            │
└──────────────────────────┼────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       CostModel    SchedulingPolicy   Alerts
       .evaluate()  .schedule()        (thermal critical, OOM)
```
