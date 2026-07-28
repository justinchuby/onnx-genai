//! Engine-side budgeted adapter pool — the control plane over the ep-api data
//! plane (design `docs/NATIVE_LORA_DESIGN.md` §J.2).
//!
//! The ep-api [`LoraWeightPool`] is the *data plane*: aligned pages, adapter-
//! major layout, read-only hot-path accessors. It deliberately carries its own
//! self-contained byte accounting rather than the scheduler's [`ByteBudget`],
//! because ep-api is a leaf crate every execution provider depends on — importing
//! the higher-layer scheduler into it would invert the dependency graph and force
//! every EP consumer to build the scheduler + KV + metadata stack.
//!
//! [`BudgetedLoraPool`] is the missing *control plane* that literally reuses the
//! shared cross-session [`ByteBudget`] (design §J.2 "reuse ByteBudget"): it makes
//! the budget authoritative over adapter residency. The inner pool is created
//! with an effectively unbounded ceiling so it never self-evicts; every admission
//! first reserves the exact resident byte cost from the shared budget (failing
//! loud, cross-session, if the machine-wide ceiling is hit) and every eviction
//! releases it. Because the same [`ByteBudget`] handle can be shared with the KV
//! subsystem, adapter memory and KV memory account against one device-wide
//! ceiling.

use std::collections::HashMap;
use std::sync::Arc;

use onnx_genai_scheduler::byte_budget::{ByteBudget, ByteBudgetError};
use onnx_runtime_ep_api::{
    AdapterId, LoraFactorInput, LoraModuleId, LoraPoolError, LoraPoolId, LoraPoolRegistry,
    LoraWeightPool,
};

/// A failure admitting an adapter factor pair through the budgeted pool.
#[derive(Debug, thiserror::Error)]
pub enum BudgetedLoraPoolError {
    /// The shared cross-session byte budget rejected the reservation.
    #[error("adapter pool admission rejected by the shared byte budget: {0}")]
    Budget(#[from] ByteBudgetError),
    /// The data-plane pool rejected the factor pair (shape/rank/capacity).
    #[error("adapter pool rejected the factor pair: {0}")]
    Pool(#[from] LoraPoolError),
}

/// The control-plane wrapper: an ep-api [`LoraWeightPool`] whose residency is
/// governed by a shared [`ByteBudget`].
pub struct BudgetedLoraPool {
    pool: LoraWeightPool,
    budget: ByteBudget,
    /// Bytes reserved from the budget per resident key, so eviction releases the
    /// exact amount admission charged.
    reserved: HashMap<(AdapterId, LoraModuleId), u64>,
}

impl BudgetedLoraPool {
    /// Create a pool governed by the shared cross-session `budget`. The inner
    /// data-plane pool is given an effectively unbounded ceiling so the budget is
    /// the sole gate; share the same `budget` handle with the KV subsystem to
    /// account adapter and KV memory against one device-wide ceiling.
    pub fn new(budget: ByteBudget) -> Self {
        Self {
            pool: LoraWeightPool::with_capacity_bytes(u64::MAX),
            budget,
            reserved: HashMap::new(),
        }
    }

    /// Bytes currently reserved from the shared budget by this pool's pages.
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved.values().copied().sum()
    }

    /// Number of resident `(adapter, module)` pairs.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Whether the pool holds no pages.
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Admit one `(adapter, module)` factor pair, reserving its exact resident
    /// byte cost from the shared budget first. On budget rejection nothing is
    /// admitted; on a data-plane rejection the reservation is rolled back, so the
    /// budget never leaks bytes for a page that did not land.
    pub fn admit(
        &mut self,
        adapter: AdapterId,
        module: LoraModuleId,
        a_t: LoraFactorInput<'_>,
        b_t: LoraFactorInput<'_>,
        scale: f32,
    ) -> Result<(), BudgetedLoraPoolError> {
        let cost = LoraWeightPool::page_pair_resident_bytes(a_t.bytes.len(), b_t.bytes.len());

        // A re-admission of the same key releases its old reservation first so the
        // budget nets only the delta (a hot-swap of the same slot).
        if let Some(previous) = self.reserved.remove(&(adapter, module)) {
            self.budget.release(previous);
        }

        self.budget.try_reserve(cost)?;
        match self.pool.admit(adapter, module, a_t, b_t, scale) {
            Ok(()) => {
                self.reserved.insert((adapter, module), cost);
                Ok(())
            }
            Err(error) => {
                // Roll the reservation back so a shape/rank rejection never leaks.
                self.budget.release(cost);
                Err(error.into())
            }
        }
    }

    /// Evict a specific pair, releasing its reserved bytes back to the shared
    /// budget. Returns whether it was present.
    pub fn evict(&mut self, adapter: AdapterId, module: LoraModuleId) -> bool {
        let present = self.pool.evict(adapter, module);
        if let Some(bytes) = self.reserved.remove(&(adapter, module)) {
            self.budget.release(bytes);
        }
        present
    }

    /// Freeze the populated data-plane pool into a shared handle and register it
    /// in the process [`LoraPoolRegistry`], returning the `pool_id` to bake into
    /// the emitted `GroupedLoraDelta` ops. Consumes `self`, handing ownership of
    /// the pages to the registry; the reservations stay live for the lifetime of
    /// the returned `Arc` (release them by unregistering and dropping it).
    pub fn register(self) -> (LoraPoolId, Arc<LoraWeightPool>) {
        let pool = Arc::new(self.pool);
        let id = LoraPoolRegistry::global().register(Arc::clone(&pool));
        (id, pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::DataType;

    fn factor(rows: usize, cols: usize, bytes: &[u8]) -> LoraFactorInput<'_> {
        LoraFactorInput {
            dtype: DataType::Float32,
            rows,
            cols,
            bytes,
        }
    }

    #[test]
    fn admit_reserves_and_evict_releases_budget() {
        let budget = ByteBudget::new(4096);
        let mut pool = BudgetedLoraPool::new(budget.clone());

        // A_t [2,1] = 8 bytes -> one 64-byte page; B_t [1,3] = 12 bytes -> one
        // 64-byte page. Resident cost = 128 bytes.
        let a = vec![0u8; 8];
        let b = vec![0u8; 12];
        pool.admit(
            AdapterId(0),
            LoraModuleId(0),
            factor(2, 1, &a),
            factor(1, 3, &b),
            1.0,
        )
        .expect("admit fits");
        assert_eq!(pool.reserved_bytes(), 128);
        assert_eq!(budget.snapshot().used, 128);

        assert!(pool.evict(AdapterId(0), LoraModuleId(0)));
        assert_eq!(pool.reserved_bytes(), 0);
        assert_eq!(budget.snapshot().used, 0);
    }

    #[test]
    fn over_budget_admission_fails_loud_and_leaks_nothing() {
        // Ceiling smaller than one page pair (128 bytes).
        let budget = ByteBudget::new(64);
        let mut pool = BudgetedLoraPool::new(budget.clone());
        let a = vec![0u8; 8];
        let b = vec![0u8; 12];
        let error = pool
            .admit(
                AdapterId(0),
                LoraModuleId(0),
                factor(2, 1, &a),
                factor(1, 3, &b),
                1.0,
            )
            .expect_err("must exceed the byte budget");
        assert!(matches!(error, BudgetedLoraPoolError::Budget(_)));
        assert!(pool.is_empty());
        assert_eq!(budget.snapshot().used, 0, "rejected admission leaks no bytes");
    }

    #[test]
    fn shape_rejection_rolls_back_reservation() {
        let budget = ByteBudget::new(4096);
        let mut pool = BudgetedLoraPool::new(budget.clone());
        // Rank disagreement: A_t cols (1) != B_t rows (2).
        let a = vec![0u8; 8];
        let b = vec![0u8; 24];
        let error = pool
            .admit(
                AdapterId(0),
                LoraModuleId(0),
                factor(2, 1, &a),
                factor(2, 3, &b),
                1.0,
            )
            .expect_err("rank mismatch must fail");
        assert!(matches!(error, BudgetedLoraPoolError::Pool(_)));
        assert_eq!(budget.snapshot().used, 0, "data-plane rejection releases the reservation");
    }
}
