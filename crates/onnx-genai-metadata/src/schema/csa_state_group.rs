//! Model-IO contract for a CompressedSparseAttention (CSA/HCA) state group.
//!
//! DeepSeek-V4 threads compressed KV attention state as graph `past_* ->
//! present_*` port pairs, not as a growing dense KV cache and not as a
//! fixed-shape recurrent replace tensor. The compressed records, their
//! compression carry, and — for the query-selective ratio-4 variant — the
//! learned index key and index carry are each a distinct state edge whose
//! logical length is a backend-owned cursor that cannot be inferred from the
//! token count.
//!
//! This module declares that contract in metadata terms: which ratio and cache
//! format the group was built against, and which graph ports carry each state
//! edge. It is intentionally property-based and role-typed — it never names a
//! model, a layer, or a tensor spelling — so a runtime discovers the group from
//! the declared roles and refuses, with a typed reason, anything it cannot
//! honor *before* it allocates a byte. The concrete dtype/shape check against a
//! real graph is a runtime concern (the engine validates the declared ports
//! against the session's typed IO); this type owns the declaration and the
//! property-level validity of the group itself.

use super::*;

/// Official compression ratio of a CompressedSparseAttention state group.
///
/// Only the two property-compatible official ratios are expressible. Ratio-4 is
/// query-selective CSA (compressor + a learned FP4 index); ratio-128 is HCA,
/// temporal compression with no index. Any other ratio is refused before
/// allocation rather than silently coerced to one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CsaCompressionRatio {
    /// Query-selective compressed sparse attention (compressor + learned index).
    Ratio4,
    /// Hierarchical compressed attention: temporal compression, no index.
    Ratio128,
}

impl CsaCompressionRatio {
    /// Whether this ratio threads learned-index state edges.
    ///
    /// Ratio-4 selects keys with a learned FP4 index and therefore threads
    /// `index_key`/`index_carry`; ratio-128 compresses temporally and threads
    /// neither.
    pub fn has_index_edges(self) -> bool {
        matches!(self, Self::Ratio4)
    }
}

impl std::fmt::Display for CsaCompressionRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ratio4 => f.write_str("ratio4"),
            Self::Ratio128 => f.write_str("ratio128"),
        }
    }
}

/// Element/cache format of the compressed KV records a CSA group carries.
///
/// `fp4_e2m1_block32` is the learned *index* format, never a KV cache format,
/// so declaring it for the compressed KV cache is refused for both ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CsaCacheFormat {
    /// Uncompressed float32 records.
    F32,
    /// FP8 e4m3 records in blocks of 64.
    Fp8E4m3Block64,
    /// FP4 e2m1 records in blocks of 32 — the learned-index format only.
    Fp4E2m1Block32,
}

impl std::fmt::Display for CsaCacheFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::Fp8E4m3Block64 => f.write_str("fp8_e4m3_block64"),
            Self::Fp4E2m1Block32 => f.write_str("fp4_e2m1_block32"),
        }
    }
}

/// Which `present_* -> past_*` state edge of a CSA group a port pair carries.
///
/// A role rather than a tensor name because the four edges are shape- and
/// dtype-overlapping (two `uint8` record buffers, two `float32` carries) and so
/// are indistinguishable by structure; only the producer knows which port is
/// which, so it declares the role and the runtime never guesses from spelling.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CsaStateRole {
    /// Compressed KV records accumulated by the compressor.
    CompressedKv,
    /// Fixed-size compression carry (the compressor's running state).
    CompressionCarry,
    /// Learned index keys (ratio-4 only).
    IndexKey,
    /// Fixed-size index carry (ratio-4 only).
    IndexCarry,
}

impl CsaStateRole {
    /// All four roles in a stable canonical order.
    pub const ALL: [CsaStateRole; 4] = [
        CsaStateRole::CompressedKv,
        CsaStateRole::CompressionCarry,
        CsaStateRole::IndexKey,
        CsaStateRole::IndexCarry,
    ];

    /// Whether this role is a learned-index edge (present only for ratio-4).
    pub fn is_index_edge(self) -> bool {
        matches!(self, Self::IndexKey | Self::IndexCarry)
    }
}

impl std::fmt::Display for CsaStateRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompressedKv => f.write_str("compressed_kv"),
            Self::CompressionCarry => f.write_str("compression_carry"),
            Self::IndexKey => f.write_str("index_key"),
            Self::IndexCarry => f.write_str("index_carry"),
        }
    }
}

/// The recurrence discipline a CSA group threads.
///
/// Only `standard` per-step compression is supported. Multi-token prediction
/// adds a recurrence this slice does not model, so a group that declares it is
/// refused before allocation rather than approximated — an invented MTP
/// recurrence would silently corrupt state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CsaRecurrence {
    /// One compression step per decoded position.
    #[default]
    Standard,
    /// Multi-token prediction recurrence — declared, unsupported, refused.
    MultiTokenPrediction,
}

/// One declared `present_* -> past_*` state edge of a CSA group.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CsaStateEdge {
    /// Which state edge of the group this pair carries.
    pub role: CsaStateRole,
    /// Graph input port that receives the carried state for this step.
    #[schemars(length(min = 1))]
    pub past_port: String,
    /// Graph output port that produces the next-step state.
    #[schemars(length(min = 1))]
    pub present_port: String,
}

/// Declared model-IO contract for one CompressedSparseAttention state group.
///
/// A package emits one of these per CSA/HCA layer group whose compressed state
/// the decode loop must thread. It names the ratio, the KV cache format, and
/// the role-typed present/past edges; the runtime validates it against the real
/// graph and refuses, with a typed reason, before allocating.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CsaStateGroupAbi {
    /// Official compression ratio this group was built against.
    pub ratio: CsaCompressionRatio,
    /// Element/cache format of the compressed KV records.
    pub cache_format: CsaCacheFormat,
    /// Recurrence discipline; anything but `standard` is refused before alloc.
    #[serde(default)]
    pub recurrence: CsaRecurrence,
    /// The role-typed `present_* -> past_*` state edges of this group.
    #[schemars(length(min = 1))]
    pub edges: Vec<CsaStateEdge>,
}

/// A typed reason a CSA state group cannot be honored.
///
/// Every variant is raised *before* any device allocation, so a refusal never
/// leaves partially reserved state behind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsaStateGroupRefusal {
    /// The group declares an MTP (or other non-standard) recurrence this slice
    /// does not model; it must not be approximated.
    UnknownMtpRecurrence,
    /// `fp4_e2m1_block32` is the learned-index format and is never a legal KV
    /// cache format.
    Fp4NotKvCacheFormat,
    /// A required state edge for this ratio is absent.
    MissingStateEdge(CsaStateRole),
    /// The same state role is declared by more than one edge.
    DuplicateStateEdge(CsaStateRole),
    /// A learned-index edge was declared for a ratio that has no index.
    UnexpectedIndexEdge(CsaStateRole),
    /// A declared edge names an empty graph port.
    EmptyPort(CsaStateRole),
    /// Two edges name the same graph port, so a rebind would collide.
    PortCollision(String),
}

impl std::fmt::Display for CsaStateGroupRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownMtpRecurrence => f.write_str(
                "CSA state group declares a multi-token-prediction recurrence this slice does not \
                 model; refusing rather than inventing an MTP recurrence",
            ),
            Self::Fp4NotKvCacheFormat => f.write_str(
                "CSA cache_format 'fp4_e2m1_block32' is the learned-index format, not a KV cache \
                 format; declare 'f32' or 'fp8_e4m3_block64'",
            ),
            Self::MissingStateEdge(role) => {
                write!(
                    f,
                    "CSA state group is missing the required '{role}' state edge"
                )
            }
            Self::DuplicateStateEdge(role) => {
                write!(
                    f,
                    "CSA state group declares the '{role}' state edge more than once"
                )
            }
            Self::UnexpectedIndexEdge(role) => write!(
                f,
                "CSA state group declares the learned-index edge '{role}' for a ratio that carries \
                 no index; only ratio-4 threads index_key/index_carry"
            ),
            Self::EmptyPort(role) => {
                write!(f, "CSA state edge '{role}' declares an empty graph port")
            }
            Self::PortCollision(port) => write!(
                f,
                "CSA state group binds graph port '{port}' to more than one edge; every edge needs \
                 a distinct port"
            ),
        }
    }
}

impl std::error::Error for CsaStateGroupRefusal {}

impl CsaStateGroupAbi {
    /// The state roles this ratio requires, in canonical order.
    pub fn required_roles(&self) -> &'static [CsaStateRole] {
        if self.ratio.has_index_edges() {
            &[
                CsaStateRole::CompressedKv,
                CsaStateRole::CompressionCarry,
                CsaStateRole::IndexKey,
                CsaStateRole::IndexCarry,
            ]
        } else {
            &[CsaStateRole::CompressedKv, CsaStateRole::CompressionCarry]
        }
    }

    /// Look up a declared edge by role.
    pub fn edge(&self, role: CsaStateRole) -> Option<&CsaStateEdge> {
        self.edges.iter().find(|edge| edge.role == role)
    }

    /// Validate the property-level consistency of this group before any device
    /// allocation. Ratio/format/recurrence/edge-set validity only; the concrete
    /// dtype/shape check against a real graph is a runtime concern.
    pub fn validate(&self) -> Result<(), CsaStateGroupRefusal> {
        if matches!(self.recurrence, CsaRecurrence::MultiTokenPrediction) {
            return Err(CsaStateGroupRefusal::UnknownMtpRecurrence);
        }
        if matches!(self.cache_format, CsaCacheFormat::Fp4E2m1Block32) {
            return Err(CsaStateGroupRefusal::Fp4NotKvCacheFormat);
        }

        // No duplicate roles and no port collisions.
        let mut seen_roles: Vec<CsaStateRole> = Vec::with_capacity(self.edges.len());
        let mut seen_ports: Vec<&str> = Vec::with_capacity(self.edges.len() * 2);
        for edge in &self.edges {
            if edge.past_port.is_empty() || edge.present_port.is_empty() {
                return Err(CsaStateGroupRefusal::EmptyPort(edge.role));
            }
            if seen_roles.contains(&edge.role) {
                return Err(CsaStateGroupRefusal::DuplicateStateEdge(edge.role));
            }
            seen_roles.push(edge.role);
            for port in [edge.past_port.as_str(), edge.present_port.as_str()] {
                if seen_ports.contains(&port) {
                    return Err(CsaStateGroupRefusal::PortCollision(port.to_string()));
                }
                seen_ports.push(port);
            }
        }

        // Index edges are legal only for the ratio that carries an index.
        if !self.ratio.has_index_edges() {
            for role in [CsaStateRole::IndexKey, CsaStateRole::IndexCarry] {
                if seen_roles.contains(&role) {
                    return Err(CsaStateGroupRefusal::UnexpectedIndexEdge(role));
                }
            }
        }

        // Every required role is present.
        for role in self.required_roles() {
            if !seen_roles.contains(role) {
                return Err(CsaStateGroupRefusal::MissingStateEdge(*role));
            }
        }

        Ok(())
    }

    /// The validated `present_* -> past_*` edges in canonical role order, for a
    /// runtime to lower into its stable-address state threading.
    ///
    /// Returns `(role, past_port, present_port)` tuples so the caller threads
    /// each edge by rebinding the present output onto the past input for the
    /// next step, exactly as the compressed-state cursor discipline requires,
    /// without conflating these edges with token-counted KV cache.
    pub fn present_past_edges(
        &self,
    ) -> Result<Vec<(CsaStateRole, String, String)>, CsaStateGroupRefusal> {
        self.validate()?;
        let mut edges = Vec::with_capacity(self.required_roles().len());
        for role in self.required_roles() {
            let edge = self
                .edge(*role)
                .ok_or(CsaStateGroupRefusal::MissingStateEdge(*role))?;
            edges.push((*role, edge.past_port.clone(), edge.present_port.clone()));
        }
        Ok(edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(role: CsaStateRole, past: &str, present: &str) -> CsaStateEdge {
        CsaStateEdge {
            role,
            past_port: past.to_string(),
            present_port: present.to_string(),
        }
    }

    fn ratio128_group() -> CsaStateGroupAbi {
        CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio128,
            cache_format: CsaCacheFormat::Fp8E4m3Block64,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(
                    CsaStateRole::CompressedKv,
                    "past_compressed_kv",
                    "present_compressed_kv",
                ),
                edge(
                    CsaStateRole::CompressionCarry,
                    "past_compression_carry",
                    "present_compression_carry",
                ),
            ],
        }
    }

    fn ratio4_group() -> CsaStateGroupAbi {
        CsaStateGroupAbi {
            ratio: CsaCompressionRatio::Ratio4,
            cache_format: CsaCacheFormat::Fp8E4m3Block64,
            recurrence: CsaRecurrence::Standard,
            edges: vec![
                edge(
                    CsaStateRole::CompressedKv,
                    "past_compressed_kv",
                    "present_compressed_kv",
                ),
                edge(
                    CsaStateRole::CompressionCarry,
                    "past_compression_carry",
                    "present_compression_carry",
                ),
                edge(
                    CsaStateRole::IndexKey,
                    "past_index_key",
                    "present_index_key",
                ),
                edge(
                    CsaStateRole::IndexCarry,
                    "past_index_carry",
                    "present_index_carry",
                ),
            ],
        }
    }

    #[test]
    fn ratio128_requires_exactly_two_edges() {
        let group = ratio128_group();
        group.validate().unwrap();
        assert_eq!(
            group.required_roles(),
            &[CsaStateRole::CompressedKv, CsaStateRole::CompressionCarry]
        );
        let edges = group.present_past_edges().unwrap();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].1, "past_compressed_kv");
        assert_eq!(edges[0].2, "present_compressed_kv");
    }

    #[test]
    fn ratio4_requires_four_edges_including_index() {
        let group = ratio4_group();
        group.validate().unwrap();
        assert_eq!(group.required_roles().len(), 4);
        let edges = group.present_past_edges().unwrap();
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[2].0, CsaStateRole::IndexKey);
        assert_eq!(edges[3].0, CsaStateRole::IndexCarry);
    }

    #[test]
    fn ratio128_rejects_index_edges() {
        let mut group = ratio128_group();
        group.edges.push(edge(
            CsaStateRole::IndexKey,
            "past_index_key",
            "present_index_key",
        ));
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::UnexpectedIndexEdge(
                CsaStateRole::IndexKey
            ))
        );
    }

    #[test]
    fn ratio4_missing_index_edge_is_refused() {
        let mut group = ratio4_group();
        group.edges.retain(|e| e.role != CsaStateRole::IndexCarry);
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::MissingStateEdge(
                CsaStateRole::IndexCarry
            ))
        );
    }

    #[test]
    fn missing_compressed_kv_is_refused() {
        let mut group = ratio128_group();
        group.edges.retain(|e| e.role != CsaStateRole::CompressedKv);
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::MissingStateEdge(
                CsaStateRole::CompressedKv
            ))
        );
    }

    #[test]
    fn duplicate_role_is_refused() {
        let mut group = ratio128_group();
        group.edges.push(edge(
            CsaStateRole::CompressedKv,
            "past_compressed_kv_dup",
            "present_compressed_kv_dup",
        ));
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::DuplicateStateEdge(
                CsaStateRole::CompressedKv
            ))
        );
    }

    #[test]
    fn port_collision_is_refused() {
        let mut group = ratio128_group();
        // Second edge reuses the first edge's present port.
        group.edges[1].present_port = "present_compressed_kv".to_string();
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::PortCollision(
                "present_compressed_kv".to_string()
            ))
        );
    }

    #[test]
    fn empty_port_is_refused() {
        let mut group = ratio128_group();
        group.edges[0].past_port = String::new();
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::EmptyPort(CsaStateRole::CompressedKv))
        );
    }

    #[test]
    fn multi_token_prediction_is_refused() {
        let mut group = ratio128_group();
        group.recurrence = CsaRecurrence::MultiTokenPrediction;
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::UnknownMtpRecurrence)
        );
    }

    #[test]
    fn fp4_kv_cache_format_is_refused() {
        let mut group = ratio128_group();
        group.cache_format = CsaCacheFormat::Fp4E2m1Block32;
        assert_eq!(
            group.validate(),
            Err(CsaStateGroupRefusal::Fp4NotKvCacheFormat)
        );
        let mut group4 = ratio4_group();
        group4.cache_format = CsaCacheFormat::Fp4E2m1Block32;
        assert_eq!(
            group4.validate(),
            Err(CsaStateGroupRefusal::Fp4NotKvCacheFormat)
        );
    }

    #[test]
    fn present_past_edges_refuses_before_returning() {
        let mut group = ratio4_group();
        group.recurrence = CsaRecurrence::MultiTokenPrediction;
        assert!(group.present_past_edges().is_err());
    }

    #[test]
    fn round_trips_through_json() {
        for group in [ratio128_group(), ratio4_group()] {
            let json = serde_json::to_string(&group).unwrap();
            let back: CsaStateGroupAbi = serde_json::from_str(&json).unwrap();
            assert_eq!(group, back);
            back.validate().unwrap();
        }
    }

    #[test]
    fn recurrence_defaults_to_standard_when_absent() {
        let json = r#"{
            "ratio": "ratio128",
            "cache_format": "f32",
            "edges": [
                {"role": "compressed_kv", "past_port": "pk", "present_port": "xk"},
                {"role": "compression_carry", "past_port": "pc", "present_port": "xc"}
            ]
        }"#;
        let group: CsaStateGroupAbi = serde_json::from_str(json).unwrap();
        assert_eq!(group.recurrence, CsaRecurrence::Standard);
        group.validate().unwrap();
    }
}
