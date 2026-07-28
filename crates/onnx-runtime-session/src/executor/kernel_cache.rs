use super::*;

use std::sync::atomic::{AtomicU64, Ordering};

/// Global counters for kernel pre-binding path reachability.
/// Incremented on the pre-bound fast path; read by tests to prove the path fires.
#[cfg(test)]
pub(crate) static PREBIND_FAST_PATH_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static PREBIND_FALLBACK_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Cache key for a compiled kernel (§11.1). Keyed by the concrete node and its
/// **resolved** (concrete) input shapes: attributes are fixed per node, so this
/// is correct, and the shape component makes it *shape-keyed* — a re-run with
/// the same resolved shapes hits, a different shape (e.g. a new batch/seq)
/// misses and re-compiles. This preserves Chew's guarantee: a kernel is never
/// reused for a shape it was not compiled for.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) struct KernelKey {
    pub(super) node: u32,
    pub(super) shapes: Vec<Vec<usize>>,
}

impl KernelKey {
    /// Check whether the cached key matches the current input shapes **without
    /// allocating**. The caller's `input_shapes` is a `&[Vec<usize>]` (the reused
    /// scratch), so this comparison is a flat slice-of-slices equality.
    #[inline]
    pub(super) fn matches_shapes(&self, input_shapes: &[Vec<usize>]) -> bool {
        self.shapes.len() == input_shapes.len()
            && self
                .shapes
                .iter()
                .zip(input_shapes.iter())
                .all(|(a, b)| a.as_slice() == b.as_slice())
    }
}

/// Observable kernel-cache statistics (§11.1) — enough to prove reuse in tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Distinct compiled entries currently held.
    pub entries: usize,
    /// Lookups served from an existing entry.
    pub hits: u64,
    /// Lookups that compiled a new kernel.
    pub misses: u64,
    /// Lookups served via the pre-bound fast path (zero-alloc).
    pub prebind_hits: u64,
}

/// Shape-keyed kernel cache (§11.1). Owns the compiled kernels for the session.
#[derive(Default)]
pub(crate) struct KernelCache {
    pub(super) entries: HashMap<KernelKey, Box<dyn onnx_runtime_ep_api::Kernel>>,
    pub(super) hits: u64,
    pub(super) misses: u64,
    pub(super) prebind_hits: AtomicU64,
}

impl KernelCache {
    pub(super) fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            prebind_hits: self.prebind_hits.load(Ordering::Relaxed),
        }
    }

    /// Zero-allocation kernel lookup via a pre-stored binding key.
    ///
    /// Returns `Some` when `binding` shapes match the current `input_shapes`
    /// and the compiled kernel is present in the cache. Returns `None` on any
    /// mismatch — caller falls through to `get_or_create`. This is the
    /// **pre-bound fast path**: during steady-state decode (fixed shapes), it
    /// replaces the per-token HashMap-key allocation with a single
    /// pointer chase + slice comparison.
    #[inline]
    pub(super) fn get_prebound<'a>(
        &'a self,
        binding: &KernelKey,
        input_shapes: &[Vec<usize>],
    ) -> Option<&'a dyn onnx_runtime_ep_api::Kernel> {
        if !binding.matches_shapes(input_shapes) {
            return None;
        }
        let kernel = self.entries.get(binding)?.as_ref();
        self.prebind_hits.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        PREBIND_FAST_PATH_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(kernel)
    }

    /// Return the cached kernel for `(node, resolved_input_shapes)`, verifying
    /// EP support and compiling+inserting it on a miss. Also returns the
    /// [`KernelKey`] so the caller can store it as a pre-binding for future
    /// zero-alloc lookups.
    // Each argument is an independent part of the kernel-cache key or the EP
    // contract; bundling them into a context struct is tracked separately
    // (Dallas #5, kernel-dispatch decomposition).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn get_or_create(
        &mut self,
        node_id: NodeId,
        node: &Node,
        input_shapes: &[Vec<usize>],
        input_dtypes: &[DataType],
        constant_inputs: &[bool],
        opset: u64,
        ep: &dyn ExecutionProvider,
    ) -> Result<(&dyn onnx_runtime_ep_api::Kernel, KernelKey)> {
        let key = KernelKey {
            node: node_id.0,
            shapes: input_shapes.to_vec(),
        };
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            // Verify the EP claims this op at these concrete shapes/layouts
            // before compiling — same gate the static path used at build.
            let shape_dims: Vec<Shape> = input_shapes
                .iter()
                .map(|s| s.iter().map(|&d| Dim::Static(d)).collect())
                .collect();
            let layouts = vec![TensorLayout::contiguous(); input_shapes.len()];
            if let KernelMatch::Unsupported { reason } =
                ep.supports_op(node, opset, &shape_dims, input_dtypes, &layouts)
            {
                return Err(SessionError::unsupported_op(
                    node,
                    node_id,
                    opset,
                    ep.name(),
                    reason,
                ));
            }
            let mut kernel = match ep.get_kernel(node, input_shapes, opset) {
                Ok(kernel) => kernel,
                Err(EpError::NoEpForOp {
                    domain,
                    op_type,
                    opset,
                }) => {
                    // Opset-aware claims should make this unreachable. Preserve
                    // the actionable diagnostic if an EP's claim drifts.
                    return Err(SessionError::unsupported_op(
                        node,
                        node_id,
                        opset,
                        ep.name(),
                        format!(
                            "no handler for {domain}::{op_type} at opset {opset} — add a claim+handler"
                        ),
                    ));
                }
                Err(error) => return Err(error.into()),
            };
            kernel.set_constant_inputs(constant_inputs);
            self.entries.insert(key.clone(), kernel);
            self.misses += 1;
        }
        #[cfg(test)]
        PREBIND_FALLBACK_TEST_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let kernel_ref = self.entries.get(&key).expect("just inserted").as_ref();
        Ok((kernel_ref, key))
    }
}
