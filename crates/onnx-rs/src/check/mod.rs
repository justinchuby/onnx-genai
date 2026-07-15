//! Extensible model checker / validator (ONNX_RS §8).
//!
//! The architecture mirrors §8.1: a [`ValidationRule`] trait, a [`Severity`] /
//! [`Violation`] model, and an [`OnnxChecker`] registry that runs an ordered set
//! of rules and can be extended with custom ones (§8.3). Three built-in rules
//! ship in this first wave (see [`rules`]); the full ONNX spec rule set and the
//! schema-driven op-level rules are deferred to later waves.
//!
//! Note: the runtime [`onnx_runtime_loader`] already rejects many malformed
//! models *at load time* (dangling refs, duplicate producers, missing opset
//! imports, cycles, …). This checker is complementary: it runs against an
//! already-built [`Model`] so tools can validate graphs they *constructed in
//! memory* and so organisations can layer their own rules on top.

mod rules;

pub use rules::{DuplicateValueNameRule, GraphAcyclicRule, MissingOpsetImportRule};

use crate::model::Model;

/// Severity of a [`Violation`] (ONNX_RS §8.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// Model is invalid per the ONNX spec.
    Error,
    /// Likely a mistake but technically valid.
    Warning,
    /// Suggestion / best practice.
    Info,
}

/// Where a [`Violation`] was found (ONNX_RS §8.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ViolationLocation {
    /// The model as a whole.
    Model,
    /// A named graph (top-level or subgraph).
    Graph { graph_name: String },
    /// A specific node within a graph.
    Node {
        graph_name: String,
        node_name: String,
    },
    /// A specific value / edge.
    Value { value_name: String },
}

/// A single problem found by a [`ValidationRule`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    /// The id of the rule that produced this violation.
    pub rule_id: String,
    /// How severe the problem is.
    pub severity: Severity,
    /// Human-readable description.
    pub message: String,
    /// Where in the model the problem is.
    pub location: ViolationLocation,
}

/// Shared context handed to every rule during a check.
///
/// A placeholder for now; later waves will thread the op-schema registry
/// (ONNX_RS §7) and version info through here so op-level rules can look up
/// operator definitions.
#[derive(Clone, Debug, Default)]
pub struct ValidationContext {
    // FOLLOW-UP (ONNX_RS §7): carry the OpSchema registry so op-level rules
    // (input/output arity, attribute types, type constraints) can be added.
    _private: (),
}

/// A rule that checks one aspect of a model (ONNX_RS §8.1).
pub trait ValidationRule: Send + Sync {
    /// Unique identifier for enable/disable (e.g. `"ir.opset_import_present"`).
    fn id(&self) -> &str;

    /// Severity of the violations this rule produces.
    fn severity(&self) -> Severity;

    /// Run the check and return any violations found.
    fn check(&self, model: &Model, ctx: &ValidationContext) -> Vec<Violation>;
}

/// The outcome of running a checker (ONNX_RS §8.2).
#[derive(Clone, Debug, Default)]
pub struct ValidationResult {
    /// All violations found, in rule order.
    pub violations: Vec<Violation>,
    /// Count of `Error`-severity violations.
    pub errors: usize,
    /// Count of `Warning`-severity violations.
    pub warnings: usize,
}

impl ValidationResult {
    /// True when there are no `Error`-severity violations.
    pub fn is_valid(&self) -> bool {
        self.errors == 0
    }
}

/// The standard, extensible checker (ONNX_RS §8.2).
///
/// Construct with [`OnnxChecker::new`] for the built-in rule set, or
/// [`OnnxChecker::empty`] to build a custom set from scratch. Add organisation-
/// specific rules with [`OnnxChecker::add_rule`] (§8.3) and turn individual
/// rules off with [`OnnxChecker::disable_rule`].
pub struct OnnxChecker {
    rules: Vec<Box<dyn ValidationRule>>,
    disabled: std::collections::HashSet<String>,
    context: ValidationContext,
}

impl OnnxChecker {
    /// A checker with no rules registered.
    pub fn empty() -> Self {
        Self {
            rules: Vec::new(),
            disabled: std::collections::HashSet::new(),
            context: ValidationContext::default(),
        }
    }

    /// A checker pre-loaded with the built-in rules for this wave.
    pub fn new() -> Self {
        let mut checker = Self::empty();
        checker.add_rule(MissingOpsetImportRule);
        checker.add_rule(DuplicateValueNameRule);
        checker.add_rule(GraphAcyclicRule);
        // FOLLOW-UP (ONNX_RS §8.2): add the remaining structural, type, and
        // schema-driven op-level rules from the design.
        checker
    }

    /// Register a custom rule (ONNX_RS §8.3).
    pub fn add_rule<R: ValidationRule + 'static>(&mut self, rule: R) {
        self.rules.push(Box::new(rule));
    }

    /// Disable a rule by id; disabled rules are skipped during [`Self::check`].
    pub fn disable_rule(&mut self, rule_id: &str) {
        self.disabled.insert(rule_id.to_string());
    }

    /// Ids of all currently-registered rules, in run order.
    pub fn rule_ids(&self) -> Vec<&str> {
        self.rules.iter().map(|r| r.id()).collect()
    }

    /// Run all enabled rules and aggregate the result.
    pub fn check(&self, model: &Model) -> ValidationResult {
        let mut result = ValidationResult::default();
        for rule in &self.rules {
            if self.disabled.contains(rule.id()) {
                continue;
            }
            for violation in rule.check(model, &self.context) {
                match violation.severity {
                    Severity::Error => result.errors += 1,
                    Severity::Warning => result.warnings += 1,
                    Severity::Info => {}
                }
                result.violations.push(violation);
            }
        }
        result
    }
}

impl Default for OnnxChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysFails;
    impl ValidationRule for AlwaysFails {
        fn id(&self) -> &str {
            "test.always_fails"
        }
        fn severity(&self) -> Severity {
            Severity::Error
        }
        fn check(&self, _model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
            vec![Violation {
                rule_id: self.id().to_string(),
                severity: Severity::Error,
                message: "boom".to_string(),
                location: ViolationLocation::Model,
            }]
        }
    }

    fn empty_model() -> Model {
        let mut g = onnx_runtime_ir::Graph::new();
        g.opset_imports.insert(String::new(), 21);
        Model::new(g)
    }

    #[test]
    fn empty_checker_reports_no_violations() {
        let result = OnnxChecker::empty().check(&empty_model());
        assert!(result.is_valid());
        assert!(result.violations.is_empty());
    }

    #[test]
    fn custom_rule_runs_and_counts_errors() {
        let mut checker = OnnxChecker::empty();
        checker.add_rule(AlwaysFails);
        let result = checker.check(&empty_model());
        assert_eq!(result.errors, 1);
        assert!(!result.is_valid());
    }

    #[test]
    fn disabled_rule_is_skipped() {
        let mut checker = OnnxChecker::empty();
        checker.add_rule(AlwaysFails);
        checker.disable_rule("test.always_fails");
        let result = checker.check(&empty_model());
        assert!(result.is_valid());
    }

    #[test]
    fn default_checker_has_builtin_rules() {
        let checker = OnnxChecker::new();
        let ids = checker.rule_ids();
        assert!(ids.contains(&"ir.opset_import_present"));
        assert!(ids.contains(&"structure.duplicate_value_name"));
        assert!(ids.contains(&"structure.graph_acyclic"));
    }
}
