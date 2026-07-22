//! Router configuration model (see `docs/DESIGN.md` §34.11).
//!
//! The config is deliberately generalizable: routing policy is an enum with
//! per-policy knobs, health thresholds are explicit, and session-map
//! persistence is optional. It carries **no** model-specific settings — the
//! router is model-agnostic.

use serde::Deserialize;

/// Top-level router configuration, parsed from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct RouterConfig {
    /// Address the router's own HTTP server binds to (used by R3).
    pub listen: String,
    /// Inference nodes to route across. Must be non-empty.
    pub nodes: Vec<NodeConfig>,
    /// Routing policy + knobs.
    #[serde(default)]
    pub routing: RoutingConfig,
    /// Health-check thresholds.
    #[serde(default)]
    pub health: HealthConfig,
    /// Optional session-map persistence.
    #[serde(default)]
    pub session_map: SessionMapConfig,
}

/// A single inference node entry.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeConfig {
    /// Upstream address (`host:port`).
    pub address: String,
    /// Opaque node name/id.
    pub name: String,
}

/// Routing policy + tuning knobs.
#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    /// Which routing strategy to use.
    #[serde(default)]
    pub policy: RoutingPolicy,
    /// How often (ms) the background poller refreshes node status.
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,
    /// KV usage at/above which a node is considered overloaded and affinity
    /// breaks (triggering migration). Must be in `0.0..=1.0`.
    #[serde(default = "default_overload_threshold")]
    pub overload_threshold: f32,
    /// Whether to co-locate sessions that share a system-prompt prefix.
    #[serde(default = "default_true")]
    pub prefix_colocate: bool,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        RoutingConfig {
            policy: RoutingPolicy::default(),
            poll_interval_ms: default_poll_interval_ms(),
            overload_threshold: default_overload_threshold(),
            prefix_colocate: default_true(),
        }
    }
}

/// Routing strategy (see `docs/DESIGN.md` §34.5).
///
/// Deserializes from YAML in a user-friendly shape: unit variants are plain
/// snake_case strings (`policy: affinity_then_load`), and
/// [`RoutingPolicy::Weighted`] is a single-key map:
///
/// ```yaml
/// policy:
///   weighted:
///     affinity_weight: 0.5
///     kv_weight: 0.3
///     queue_weight: 0.2
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RoutingPolicy {
    /// Prefer session affinity, fall back to least-loaded.
    #[default]
    AffinityThenLoad,
    /// Prefer prefix sharing, fall back to least-loaded.
    PrefixThenLoad,
    /// Always route to least KV usage.
    LeastKvUsage,
    /// Weighted score across affinity, KV usage, and queue depth.
    Weighted(WeightConfig),
}

impl<'de> Deserialize<'de> for RoutingPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Accept either a bare string (unit variants) or a `{ weighted: {..} }`
        // map, sidestepping serde_yaml's YAML-tag enum representation.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Named(String),
            Weighted { weighted: WeightConfig },
        }
        match Raw::deserialize(deserializer)? {
            Raw::Weighted { weighted } => Ok(RoutingPolicy::Weighted(weighted)),
            Raw::Named(name) => match name.as_str() {
                "affinity_then_load" => Ok(RoutingPolicy::AffinityThenLoad),
                "prefix_then_load" => Ok(RoutingPolicy::PrefixThenLoad),
                "least_kv_usage" => Ok(RoutingPolicy::LeastKvUsage),
                "weighted" => Ok(RoutingPolicy::Weighted(WeightConfig::default())),
                other => Err(serde::de::Error::custom(format!(
                    "unknown routing policy `{other}` (expected one of: \
                     affinity_then_load, prefix_then_load, least_kv_usage, weighted)"
                ))),
            },
        }
    }
}

/// Weights for [`RoutingPolicy::Weighted`]. Defaults mirror the design's
/// `affinity × 0.5 + kv_usage × 0.3 + queue_depth × 0.2`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WeightConfig {
    #[serde(default = "default_affinity_weight")]
    pub affinity_weight: f32,
    #[serde(default = "default_kv_weight")]
    pub kv_weight: f32,
    #[serde(default = "default_queue_weight")]
    pub queue_weight: f32,
}

impl Default for WeightConfig {
    fn default() -> Self {
        WeightConfig {
            affinity_weight: default_affinity_weight(),
            kv_weight: default_kv_weight(),
            queue_weight: default_queue_weight(),
        }
    }
}

/// Health-check thresholds.
#[derive(Debug, Clone, Deserialize)]
pub struct HealthConfig {
    /// How often (ms) to run a health sweep.
    #[serde(default = "default_check_interval_ms")]
    pub check_interval_ms: u64,
    /// Mark a node unhealthy after this many consecutive missed polls.
    #[serde(default = "default_unhealthy_after_misses")]
    pub unhealthy_after_misses: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        HealthConfig {
            check_interval_ms: default_check_interval_ms(),
            unhealthy_after_misses: default_unhealthy_after_misses(),
        }
    }
}

/// Optional persistence of the session→node affinity table so it survives a
/// router restart.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct SessionMapConfig {
    /// Whether to persist the session map at all.
    #[serde(default)]
    pub persist: bool,
    /// Where to persist it (JSON). Required when `persist` is true.
    #[serde(default)]
    pub persist_path: Option<String>,
}

/// Errors produced while loading or validating configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The YAML text could not be parsed.
    #[error("failed to parse router config YAML: {0}")]
    Parse(#[from] serde_yaml::Error),
    /// The config file could not be read.
    #[error("failed to read router config at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// A semantic validation rule was violated.
    #[error("invalid router config: {0}")]
    Invalid(String),
}

impl RouterConfig {
    /// Parse a [`RouterConfig`] from a YAML string and validate it.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, ConfigError> {
        let cfg: RouterConfig = serde_yaml::from_str(yaml)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read and parse a [`RouterConfig`] from a YAML file, then validate it.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_yaml_str(&text)
    }

    /// Validate semantic invariants that serde cannot express.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.nodes.is_empty() {
            return Err(ConfigError::Invalid("`nodes` must not be empty".into()));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if node.address.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "node[{i}] has an empty `address`"
                )));
            }
            if node.name.trim().is_empty() {
                return Err(ConfigError::Invalid(format!(
                    "node[{i}] has an empty `name`"
                )));
            }
        }
        // Reject duplicate node names — ids must be unique to key affinity.
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                if self.nodes[i].name == self.nodes[j].name {
                    return Err(ConfigError::Invalid(format!(
                        "duplicate node name `{}`",
                        self.nodes[i].name
                    )));
                }
            }
        }
        check_unit_interval(
            "routing.overload_threshold",
            self.routing.overload_threshold,
        )?;
        if let RoutingPolicy::Weighted(w) = &self.routing.policy {
            check_unit_interval("routing.weighted.affinity_weight", w.affinity_weight)?;
            check_unit_interval("routing.weighted.kv_weight", w.kv_weight)?;
            check_unit_interval("routing.weighted.queue_weight", w.queue_weight)?;
        }
        if self.health.unhealthy_after_misses == 0 {
            return Err(ConfigError::Invalid(
                "`health.unhealthy_after_misses` must be >= 1".into(),
            ));
        }
        if self.session_map.persist && self.session_map.persist_path.is_none() {
            return Err(ConfigError::Invalid(
                "`session_map.persist_path` is required when persist = true".into(),
            ));
        }
        Ok(())
    }
}

fn check_unit_interval(field: &str, value: f32) -> Result<(), ConfigError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(ConfigError::Invalid(format!(
            "`{field}` must be within 0.0..=1.0 (got {value})"
        )));
    }
    Ok(())
}

fn default_true() -> bool {
    true
}
fn default_poll_interval_ms() -> u64 {
    1000
}
fn default_overload_threshold() -> f32 {
    0.95
}
fn default_check_interval_ms() -> u64 {
    5000
}
fn default_unhealthy_after_misses() -> u32 {
    3
}
fn default_affinity_weight() -> f32 {
    0.5
}
fn default_kv_weight() -> f32 {
    0.3
}
fn default_queue_weight() -> f32 {
    0.2
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
  - address: "10.0.0.2:8000"
    name: "gpu-1"
  - address: "10.0.0.3:8000"
    name: "gpu-2"
routing:
  policy: affinity_then_load
  poll_interval_ms: 1000
  overload_threshold: 0.95
  prefix_colocate: true
health:
  check_interval_ms: 5000
  unhealthy_after_misses: 3
session_map:
  persist: false
  persist_path: "/var/lib/onnx-genai-router/sessions.json"
"#;

    #[test]
    fn parses_valid_config() {
        let cfg = RouterConfig::from_yaml_str(VALID).expect("valid config");
        assert_eq!(cfg.listen, "0.0.0.0:8080");
        assert_eq!(cfg.nodes.len(), 3);
        assert_eq!(cfg.nodes[2].name, "gpu-2");
        assert_eq!(cfg.routing.policy, RoutingPolicy::AffinityThenLoad);
        assert_eq!(cfg.routing.poll_interval_ms, 1000);
        assert!((cfg.routing.overload_threshold - 0.95).abs() < 1e-6);
        assert!(cfg.routing.prefix_colocate);
        assert_eq!(cfg.health.unhealthy_after_misses, 3);
        assert!(!cfg.session_map.persist);
    }

    #[test]
    fn applies_defaults_for_omitted_sections() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
"#;
        let cfg = RouterConfig::from_yaml_str(yaml).unwrap();
        assert_eq!(cfg.routing.policy, RoutingPolicy::AffinityThenLoad);
        assert_eq!(cfg.routing.poll_interval_ms, 1000);
        assert!((cfg.routing.overload_threshold - 0.95).abs() < 1e-6);
        assert_eq!(cfg.health.check_interval_ms, 5000);
        assert_eq!(cfg.health.unhealthy_after_misses, 3);
        assert!(!cfg.session_map.persist);
    }

    #[test]
    fn parses_weighted_policy() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
routing:
  policy:
    weighted:
      affinity_weight: 0.6
      kv_weight: 0.3
      queue_weight: 0.1
"#;
        let cfg = RouterConfig::from_yaml_str(yaml).unwrap();
        match cfg.routing.policy {
            RoutingPolicy::Weighted(w) => {
                assert!((w.affinity_weight - 0.6).abs() < 1e-6);
                assert!((w.queue_weight - 0.1).abs() < 1e-6);
            }
            other => panic!("expected weighted, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_nodes() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes: []
"#;
        let err = RouterConfig::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
    }

    #[test]
    fn rejects_out_of_range_threshold() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
routing:
  overload_threshold: 1.5
"#;
        let err = RouterConfig::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
    }

    #[test]
    fn rejects_duplicate_node_names() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
  - address: "10.0.0.2:8000"
    name: "gpu-0"
"#;
        let err = RouterConfig::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
    }

    #[test]
    fn rejects_persist_without_path() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
session_map:
  persist: true
"#;
        let err = RouterConfig::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
    }

    #[test]
    fn rejects_zero_unhealthy_after_misses() {
        let yaml = r#"
listen: "0.0.0.0:8080"
nodes:
  - address: "10.0.0.1:8000"
    name: "gpu-0"
health:
  unhealthy_after_misses: 0
"#;
        let err = RouterConfig::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(_)));
    }
}
