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
//! releases it. Remaining reservations are owned by the populated data-plane
//! pool and returned when its final `Arc` is dropped.

use std::collections::HashMap;
use std::sync::Arc;

use onnx_genai_scheduler::byte_budget::{ByteBudget, ByteBudgetError};
use onnx_runtime_ep_api::{
    AdapterId, LoraFactorInput, LoraModuleId, LoraPoolError, LoraPoolRegistration,
    LoraPoolRegistry, LoraWeightPool,
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

struct BudgetReservation {
    budget: ByteBudget,
    reserved_bytes: u64,
}

impl BudgetReservation {
    fn new(budget: ByteBudget) -> Self {
        Self { budget, reserved_bytes: 0 }
    }

    fn reserve(&mut self, bytes: u64) -> Result<(), ByteBudgetError> {
        self.budget.try_reserve(bytes)?;
        self.reserved_bytes += bytes;
        Ok(())
    }

    fn release(&mut self, bytes: u64) {
        self.reserved_bytes = self
            .reserved_bytes
            .checked_sub(bytes)
            .expect("LoRA budget release must not exceed its reservation");
        self.budget.release(bytes);
    }
}

impl Drop for BudgetReservation {
    fn drop(&mut self) {
        let reserved_bytes = std::mem::take(&mut self.reserved_bytes);
        if reserved_bytes != 0 {
            self.budget.release(reserved_bytes);
        }
    }
}

/// The control-plane wrapper: an ep-api [`LoraWeightPool`] whose residency is
/// governed by a shared [`ByteBudget`].
pub struct BudgetedLoraPool {
    pool: LoraWeightPool,
    reservation: BudgetReservation,
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
            reservation: BudgetReservation::new(budget),
            reserved: HashMap::new(),
        }
    }

    /// Bytes currently reserved from the shared budget by this pool's pages.
    pub fn reserved_bytes(&self) -> u64 {
        self.reservation.reserved_bytes
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

        let previous = self.reserved.get(&(adapter, module)).copied().unwrap_or(0);
        let additional = cost.saturating_sub(previous);
        if additional != 0 {
            self.reservation.reserve(additional)?;
        }

        match self.pool.admit(adapter, module, a_t, b_t, scale) {
            Ok(()) => {
                if previous > cost {
                    self.reservation.release(previous - cost);
                }
                self.reserved.insert((adapter, module), cost);
                Ok(())
            }
            Err(error) => {
                if additional != 0 {
                    self.reservation.release(additional);
                }
                Err(error.into())
            }
        }
    }

    /// Evict a specific pair, releasing its reserved bytes back to the shared
    /// budget. Returns whether it was present.
    pub fn evict(&mut self, adapter: AdapterId, module: LoraModuleId) -> bool {
        let present = self.pool.evict(adapter, module);
        if let Some(bytes) = self.reserved.remove(&(adapter, module)) {
            self.reservation.release(bytes);
        }
        present
    }

    /// Freeze the populated data-plane pool into a shared handle and register it
    /// in the process [`LoraPoolRegistry`], returning the `pool_id` to bake into
    /// the emitted `GroupedLoraDelta` ops. The registration unregisters on drop;
    /// reservations remain attached until every pool `Arc` is gone.
    pub fn register(self) -> (LoraPoolRegistration, Arc<LoraWeightPool>) {
        let Self { pool, reservation, reserved: _ } = self;
        let pool = Arc::new(pool.with_residency_owner(reservation));
        let registration = LoraPoolRegistry::global().register_owned(Arc::clone(&pool));
        (registration, pool)
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

    #[test]
    fn registered_pool_drop_restores_exact_prior_budget_availability() {
        let budget = ByteBudget::new(4096);
        budget.try_reserve(256).expect("baseline reservation");
        let available_before = budget.available();
        let mut pool = BudgetedLoraPool::new(budget.clone());
        let a = vec![0u8; 8];
        let b = vec![0u8; 12];
        for module in 0..2 {
            pool.admit(
                AdapterId(0),
                LoraModuleId(module),
                factor(2, 1, &a),
                factor(1, 3, &b),
                1.0,
            )
            .expect("admit fits");
        }

        let (registration, resident_pool) = pool.register();
        let pool_id = registration.pool_id();
        assert_eq!(budget.available(), available_before - 256);
        assert!(LoraPoolRegistry::global().get(pool_id).is_some());
        drop(registration);
        assert!(LoraPoolRegistry::global().get(pool_id).is_none());
        assert_eq!(budget.available(), available_before - 256);
        drop(resident_pool);
        assert_eq!(budget.available(), available_before);
        assert_eq!(budget.used(), 256);
    }

    #[test]
    fn eviction_then_pool_drop_does_not_double_release_budget() {
        let budget = ByteBudget::new(4096);
        budget.try_reserve(256).expect("baseline reservation");
        let available_before = budget.available();
        {
            let mut pool = BudgetedLoraPool::new(budget.clone());
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
            assert!(pool.evict(AdapterId(0), LoraModuleId(0)));
        }
        assert_eq!(budget.available(), available_before);
        assert_eq!(budget.used(), 256);
    }
}
