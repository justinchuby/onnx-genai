//! The extensible, opset-aware operator registry.
//!
//! Inference rules are keyed by `(domain, op_type)` and, within a key, by the
//! opset version at which they were introduced. Registration is *range-based*:
//! a rule registered at version `N` applies to every opset `>= N` until a later
//! registration supersedes it — mirroring how ONNX operator schemas evolve.
//! Unregistered operators are not an error; their outputs are simply left
//! unresolved (permissive behaviour).

use std::collections::HashMap;

use onnx_runtime_ir::{Node, normalize_domain};

use crate::context::{InferenceContext, MergePolicy, NodeIo, SymbolInterner};
use crate::error::ShapeInferError;

/// An operator inference rule: reads inputs from the [`InferenceContext`] and
/// sets its outputs' types (and, where applicable, shape-data).
pub type InferenceFn = fn(&mut InferenceContext) -> Result<(), ShapeInferError>;

/// A registry mapping `(domain, op_type, opset)` to an [`InferenceFn`].
#[derive(Default)]
pub struct InferenceRegistry {
    /// `(domain, op)` → ascending list of `(min_opset, rule)`.
    handlers: HashMap<(String, String), Vec<(u64, InferenceFn)>>,
}

impl InferenceRegistry {
    /// An empty registry (no rules).
    pub fn empty() -> Self {
        Self::default()
    }

    /// A registry populated with every built-in rule.
    pub fn default_registry() -> Self {
        let mut reg = Self::empty();
        crate::handlers::register_all(&mut reg);
        reg
    }

    /// Register `rule` for `(domain, op)` applying from opset `min_opset`
    /// upward. A later registration at a higher `min_opset` supersedes this one
    /// for those versions.
    pub fn register(&mut self, domain: &str, op: &str, min_opset: u64, rule: InferenceFn) {
        let key = (normalize_domain(domain).to_string(), op.to_string());
        let entry = self.handlers.entry(key).or_default();
        match entry.binary_search_by_key(&min_opset, |(v, _)| *v) {
            Ok(idx) => entry[idx] = (min_opset, rule), // replace same-version rule
            Err(idx) => entry.insert(idx, (min_opset, rule)),
        }
    }

    /// Look up the rule for `(domain, op)` effective at opset `version`: the
    /// registration with the greatest `min_opset <= version`.
    pub fn get(&self, domain: &str, op: &str, version: u64) -> Option<InferenceFn> {
        let key = (normalize_domain(domain).to_string(), op.to_string());
        let entry = self.handlers.get(&key)?;
        let mut chosen = None;
        for &(min_opset, rule) in entry {
            if min_opset <= version {
                chosen = Some(rule);
            } else {
                break;
            }
        }
        chosen
    }

    /// Infer a single node's outputs.
    ///
    /// Returns one [`NodeIo`] per output slot. An unregistered op (or one whose
    /// rule declines to resolve an output) yields empty [`NodeIo`]s — the
    /// permissive "leave it unknown" outcome, never an error.
    pub fn infer_node(
        &self,
        node: &Node,
        opset_imports: &HashMap<String, u64>,
        inputs: Vec<NodeIo>,
        policy: MergePolicy,
        interner: &mut SymbolInterner,
    ) -> Result<Vec<NodeIo>, ShapeInferError> {
        let version = {
            // Loaded IR is canonical (`normalize_domain` applied at load), so the
            // default domain is `""` for both node domains and opset-import keys.
            if node.domain.is_empty() {
                opset_imports.get("").copied().unwrap_or(1)
            } else {
                opset_imports.get(&node.domain).copied().unwrap_or(1)
            }
        };
        let Some(rule) = self.get(&node.domain, &node.op_type, version) else {
            return Ok(vec![NodeIo::default(); node.outputs.len()]);
        };
        let mut ctx = InferenceContext::new(node, inputs, opset_imports, policy, interner);
        rule(&mut ctx)?;
        Ok(ctx.into_outputs())
    }
}
