//! HuggingFace PEFT LoRA adapter format support.

use onnx_runtime_ir::{DataType, TensorData};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

const CONFIG_FILE: &str = "adapter_config.json";
const WEIGHTS_FILE: &str = "adapter_model.safetensors";

/// A PEFT adapter decoded into the contiguous tensor orientation used by ONNX `MatMul`.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedAdapter {
    pub name: String,
    pub target_modules: Vec<String>,
    pub modules: BTreeMap<String, LoadedAdapterModule>,
}

/// One paired LoRA A/B factor for a model layer and semantic target module.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedAdapterModule {
    pub module_key: String,
    pub module_name: String,
    pub layer_index: usize,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
    pub fan_in_fan_out: bool,
    pub source_layout: PeftFactorLayout,
    /// Contiguous `[K, rank]` data, transposed from PEFT storage when necessary.
    pub a_transposed: TensorData,
    /// Contiguous `[rank, N]` data, transposed from PEFT storage when necessary.
    pub b_transposed: TensorData,
}

/// Factor orientation found in the PEFT weight file before loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeftFactorLayout {
    /// PEFT `lora_A[rank, K]` and `lora_B[N, rank]`.
    Standard,
    /// Already-oriented `lora_A[K, rank]` and `lora_B[rank, N]`.
    MatMulReady,
}

/// Errors returned while reading or validating a PEFT adapter.
#[derive(Debug, thiserror::Error)]
pub enum AdapterLoadError {
    #[error("failed to read PEFT adapter file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse PEFT configuration {path}: {source}")]
    Configuration {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to parse safetensors weights {path}: {source}")]
    Safetensors {
        path: PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },
    #[error("invalid PEFT configuration: {0}")]
    InvalidConfiguration(String),
    #[error("malformed LoRA tensor key {key:?}: {reason}")]
    MalformedKey { key: String, reason: String },
    #[error("LoRA tensor {key:?} targets module {module_name:?}, which is not in target_modules")]
    UntargetedModule { key: String, module_name: String },
    #[error(
        "LoRA tensor {key:?} has unsupported dtype {dtype:?}; only fp16 and fp32 are supported"
    )]
    UnsupportedDtype { key: String, dtype: Dtype },
    #[error("duplicate LoRA factor {factor} for module {module_key:?}")]
    DuplicateFactor {
        module_key: String,
        factor: &'static str,
    },
    #[error("module {module_key:?} is missing its lora_{factor}.weight tensor")]
    MissingFactor {
        module_key: String,
        factor: &'static str,
    },
    #[error(
        "LoRA tensor {key:?} has shape {actual:?}; expected a rank-2 {expected} tensor with rank {rank}"
    )]
    InvalidShape {
        key: String,
        actual: Vec<usize>,
        expected: &'static str,
        rank: usize,
    },
    #[error("LoRA factors for module {module_key:?} have different dtypes: A is {a:?}, B is {b:?}")]
    DtypeMismatch {
        module_key: String,
        a: DataType,
        b: DataType,
    },
    #[error("tensor geometry for {key:?} overflows the platform address space")]
    GeometryOverflow { key: String },
}

#[derive(Debug, Deserialize)]
struct AdapterConfiguration {
    r: usize,
    lora_alpha: f32,
    target_modules: Vec<String>,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    rank_pattern: HashMap<String, usize>,
    #[serde(default)]
    alpha_pattern: HashMap<String, f32>,
}

#[derive(Clone, Copy, Debug)]
enum Factor {
    A,
    B,
}

impl Factor {
    fn name(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }
}

#[derive(Debug)]
struct ParsedKey {
    module_key: String,
    module_name: String,
    layer_index: usize,
    factor: Factor,
}

#[derive(Debug)]
struct PendingModule {
    module_name: String,
    layer_index: usize,
    rank: usize,
    alpha: f32,
    a: Option<(String, TensorData)>,
    b: Option<(String, TensorData)>,
}

/// Load a standard HuggingFace PEFT LoRA adapter directory.
pub fn load_peft_adapter(directory: impl AsRef<Path>) -> Result<LoadedAdapter, AdapterLoadError> {
    let directory = directory.as_ref();
    let configuration_path = directory.join(CONFIG_FILE);
    let weights_path = directory.join(WEIGHTS_FILE);

    let configuration_bytes =
        fs::read(&configuration_path).map_err(|source| AdapterLoadError::Read {
            path: configuration_path.clone(),
            source,
        })?;
    let configuration: AdapterConfiguration = serde_json::from_slice(&configuration_bytes)
        .map_err(|source| AdapterLoadError::Configuration {
            path: configuration_path,
            source,
        })?;
    validate_configuration(&configuration)?;

    let weights = fs::read(&weights_path).map_err(|source| AdapterLoadError::Read {
        path: weights_path.clone(),
        source,
    })?;
    let tensors =
        SafeTensors::deserialize(&weights).map_err(|source| AdapterLoadError::Safetensors {
            path: weights_path,
            source,
        })?;

    let mut pending = BTreeMap::<String, PendingModule>::new();
    for key in tensors.names() {
        let Some(parsed) = parse_tensor_key(key)? else {
            continue;
        };
        if !is_targeted(&parsed.module_name, &configuration.target_modules) {
            return Err(AdapterLoadError::UntargetedModule {
                key: key.to_owned(),
                module_name: parsed.module_name,
            });
        }

        let rank = pattern_value(
            &configuration.rank_pattern,
            &parsed.module_key,
            &parsed.module_name,
        )
        .unwrap_or(configuration.r);
        let alpha = pattern_value(
            &configuration.alpha_pattern,
            &parsed.module_key,
            &parsed.module_name,
        )
        .unwrap_or(configuration.lora_alpha);
        validate_rank_alpha(&parsed.module_key, rank, alpha)?;

        let view = tensors
            .tensor(key)
            .map_err(|source| AdapterLoadError::Safetensors {
                path: directory.join(WEIGHTS_FILE),
                source,
            })?;
        let tensor = read_tensor(key, &view)?;
        let entry = pending
            .entry(parsed.module_key.clone())
            .or_insert_with(|| PendingModule {
                module_name: parsed.module_name.clone(),
                layer_index: parsed.layer_index,
                rank,
                alpha,
                a: None,
                b: None,
            });
        if entry.rank != rank || entry.alpha != alpha {
            return Err(AdapterLoadError::InvalidConfiguration(format!(
                "LoRA factors for module {:?} resolve to different rank or alpha values",
                parsed.module_key
            )));
        }
        let slot = match parsed.factor {
            Factor::A => &mut entry.a,
            Factor::B => &mut entry.b,
        };
        if slot.replace((key.to_owned(), tensor)).is_some() {
            return Err(AdapterLoadError::DuplicateFactor {
                module_key: parsed.module_key,
                factor: parsed.factor.name(),
            });
        }
    }

    if pending.is_empty() {
        return Err(AdapterLoadError::InvalidConfiguration(
            "adapter_model.safetensors contains no LoRA A/B tensors".to_owned(),
        ));
    }

    let mut modules = BTreeMap::new();
    for (module_key, pending) in pending {
        let (a_key, a) = pending.a.ok_or_else(|| AdapterLoadError::MissingFactor {
            module_key: module_key.clone(),
            factor: "A",
        })?;
        let (b_key, b) = pending.b.ok_or_else(|| AdapterLoadError::MissingFactor {
            module_key: module_key.clone(),
            factor: "B",
        })?;
        let (a_transposed, b_transposed, source_layout) = orient_factors(
            &module_key,
            &a_key,
            a,
            &b_key,
            b,
            pending.rank,
            configuration.fan_in_fan_out,
        )?;
        modules.insert(
            module_key.clone(),
            LoadedAdapterModule {
                module_key,
                module_name: pending.module_name,
                layer_index: pending.layer_index,
                rank: pending.rank,
                alpha: pending.alpha,
                scale: pending.alpha / pending.rank as f32,
                fan_in_fan_out: configuration.fan_in_fan_out,
                source_layout,
                a_transposed,
                b_transposed,
            },
        );
    }

    Ok(LoadedAdapter {
        name: directory
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("adapter")
            .to_owned(),
        target_modules: configuration.target_modules,
        modules,
    })
}

fn validate_configuration(configuration: &AdapterConfiguration) -> Result<(), AdapterLoadError> {
    if configuration.target_modules.is_empty() {
        return Err(AdapterLoadError::InvalidConfiguration(
            "target_modules must not be empty".to_owned(),
        ));
    }
    validate_rank_alpha("default", configuration.r, configuration.lora_alpha)?;
    for (pattern, &rank) in &configuration.rank_pattern {
        validate_rank_alpha(pattern, rank, configuration.lora_alpha)?;
    }
    for (pattern, &alpha) in &configuration.alpha_pattern {
        validate_rank_alpha(pattern, configuration.r, alpha)?;
    }
    Ok(())
}

fn validate_rank_alpha(module_key: &str, rank: usize, alpha: f32) -> Result<(), AdapterLoadError> {
    if rank == 0 {
        return Err(AdapterLoadError::InvalidConfiguration(format!(
            "rank for module pattern {module_key:?} must be greater than zero"
        )));
    }
    if !alpha.is_finite() {
        return Err(AdapterLoadError::InvalidConfiguration(format!(
            "alpha for module pattern {module_key:?} must be finite"
        )));
    }
    Ok(())
}

fn pattern_value<T: Copy>(
    patterns: &HashMap<String, T>,
    module_key: &str,
    module_name: &str,
) -> Option<T> {
    patterns
        .iter()
        .filter(|(pattern, _)| {
            suffix_matches(module_key, pattern) || suffix_matches(module_name, pattern)
        })
        .max_by_key(|(pattern, _)| pattern.len())
        .map(|(_, value)| *value)
}

fn suffix_matches(module: &str, pattern: &str) -> bool {
    module == pattern
        || module
            .strip_suffix(pattern)
            .is_some_and(|prefix| prefix.ends_with('.'))
}

fn is_targeted(module_name: &str, targets: &[String]) -> bool {
    targets
        .iter()
        .any(|target| suffix_matches(module_name, target))
}

fn parse_tensor_key(key: &str) -> Result<Option<ParsedKey>, AdapterLoadError> {
    let (module_key, factor) = if let Some(prefix) = key.strip_suffix(".lora_A.weight") {
        (prefix, Factor::A)
    } else if let Some(prefix) = key.strip_suffix(".lora_B.weight") {
        (prefix, Factor::B)
    } else {
        return Ok(None);
    };

    let components: Vec<_> = module_key.split('.').collect();
    let (layers_position, layer_index) = components
        .windows(2)
        .enumerate()
        .find_map(|(position, pair)| {
            (pair[0] == "layers")
                .then(|| pair[1].parse::<usize>().ok().map(|index| (position, index)))
                .flatten()
        })
        .ok_or_else(|| AdapterLoadError::MalformedKey {
            key: key.to_owned(),
            reason: "expected a `layers.<non-negative integer>` path segment".to_owned(),
        })?;
    let module_name = components[(layers_position + 2)..].join(".");
    if module_name.is_empty() {
        return Err(AdapterLoadError::MalformedKey {
            key: key.to_owned(),
            reason: "the semantic module name after the layer index is empty".to_owned(),
        });
    }

    Ok(Some(ParsedKey {
        module_key: module_key.to_owned(),
        module_name,
        layer_index,
        factor,
    }))
}

fn read_tensor(
    key: &str,
    view: &safetensors::tensor::TensorView<'_>,
) -> Result<TensorData, AdapterLoadError> {
    let dtype = match view.dtype() {
        Dtype::F16 => DataType::Float16,
        Dtype::F32 => DataType::Float32,
        dtype => {
            return Err(AdapterLoadError::UnsupportedDtype {
                key: key.to_owned(),
                dtype,
            });
        }
    };
    Ok(TensorData::from_raw(
        dtype,
        view.shape().to_vec(),
        view.data().to_vec(),
    ))
}

fn orient_factors(
    module_key: &str,
    a_key: &str,
    a: TensorData,
    b_key: &str,
    b: TensorData,
    rank: usize,
    fan_in_fan_out: bool,
) -> Result<(TensorData, TensorData, PeftFactorLayout), AdapterLoadError> {
    if a.dtype != b.dtype {
        return Err(AdapterLoadError::DtypeMismatch {
            module_key: module_key.to_owned(),
            a: a.dtype,
            b: b.dtype,
        });
    }
    if a.dims.len() != 2 {
        return Err(AdapterLoadError::InvalidShape {
            key: a_key.to_owned(),
            actual: a.dims,
            expected: "rank-2 [rank, K] or [K, rank]",
            rank,
        });
    }
    if b.dims.len() != 2 {
        return Err(AdapterLoadError::InvalidShape {
            key: b_key.to_owned(),
            actual: b.dims,
            expected: "rank-2 [N, rank] or [rank, N]",
            rank,
        });
    }

    let standard_layout = a.dims[0] == rank && b.dims[1] == rank;
    let matmul_ready_layout = a.dims[1] == rank && b.dims[0] == rank;
    if standard_layout {
        Ok((
            transpose_rank_two(a_key, a)?,
            transpose_rank_two(b_key, b)?,
            PeftFactorLayout::Standard,
        ))
    } else if fan_in_fan_out && matmul_ready_layout {
        Ok((a, b, PeftFactorLayout::MatMulReady))
    } else {
        Err(AdapterLoadError::InvalidShape {
            key: format!("{module_key}: {a_key} / {b_key}"),
            actual: vec![a.dims[0], a.dims[1], b.dims[0], b.dims[1]],
            expected: if fan_in_fan_out {
                "paired [rank, K]/[N, rank] or [K, rank]/[rank, N]"
            } else {
                "paired [rank, K]/[N, rank]"
            },
            rank,
        })
    }
}

fn transpose_rank_two(key: &str, tensor: TensorData) -> Result<TensorData, AdapterLoadError> {
    let rows = tensor.dims[0];
    let columns = tensor.dims[1];
    let element_size = tensor.dtype.byte_size();
    let element_count =
        rows.checked_mul(columns)
            .ok_or_else(|| AdapterLoadError::GeometryOverflow {
                key: key.to_owned(),
            })?;
    let byte_count = element_count.checked_mul(element_size).ok_or_else(|| {
        AdapterLoadError::GeometryOverflow {
            key: key.to_owned(),
        }
    })?;
    if tensor.data.len() != byte_count {
        return Err(AdapterLoadError::InvalidShape {
            key: key.to_owned(),
            actual: tensor.dims,
            expected: "densely packed tensor data matching its declared shape",
            rank: 2,
        });
    }

    let mut transposed = vec![0_u8; byte_count];
    for row in 0..rows {
        for column in 0..columns {
            let source = (row * columns + column) * element_size;
            let destination = (column * rows + row) * element_size;
            transposed[destination..destination + element_size]
                .copy_from_slice(&tensor.data[source..source + element_size]);
        }
    }
    Ok(TensorData::from_raw(
        tensor.dtype,
        vec![columns, rows],
        transposed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize_to_file};
    use std::collections::HashMap;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn tensor_f32_values(tensor: &TensorData) -> Vec<f32> {
        tensor
            .data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn write_adapter(
        configuration: &str,
        tensors: Vec<(&str, Vec<usize>, Vec<u8>)>,
    ) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join(CONFIG_FILE), configuration).unwrap();
        let views: HashMap<_, _> = tensors
            .iter()
            .map(|(name, shape, data)| {
                (
                    (*name).to_owned(),
                    TensorView::new(Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        serialize_to_file(&views, None, &directory.path().join(WEIGHTS_FILE)).unwrap();
        directory
    }

    #[test]
    fn lora_loads_peft_factors_with_overrides_and_transposes() {
        let directory = write_adapter(
            r#"{
                "r": 1,
                "lora_alpha": 2.0,
                "target_modules": ["q_proj"],
                "fan_in_fan_out": false,
                "rank_pattern": {"layers.3.self_attn.q_proj": 2},
                "alpha_pattern": {"self_attn.q_proj": 6.0}
            }"#,
            vec![
                (
                    "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
                    vec![2, 3],
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ),
                (
                    "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight",
                    vec![4, 2],
                    f32_bytes(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]),
                ),
                (
                    "base_model.model.unrelated.weight",
                    vec![1],
                    f32_bytes(&[99.0]),
                ),
            ],
        );

        let adapter = load_peft_adapter(directory.path()).unwrap();
        assert_eq!(adapter.modules.len(), 1);
        let module = adapter.modules.values().next().unwrap();
        assert_eq!(module.layer_index, 3);
        assert_eq!(module.module_name, "self_attn.q_proj");
        assert_eq!(module.rank, 2);
        assert_eq!(module.alpha, 6.0);
        assert_eq!(module.scale, 3.0);
        assert_eq!(module.source_layout, PeftFactorLayout::Standard);
        assert_eq!(module.a_transposed.dims, [3, 2]);
        assert_eq!(module.b_transposed.dims, [2, 4]);
        assert_eq!(
            tensor_f32_values(&module.a_transposed),
            [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert_eq!(
            tensor_f32_values(&module.b_transposed),
            [10.0, 20.0, 30.0, 40.0, 11.0, 21.0, 31.0, 41.0]
        );
    }

    #[test]
    fn lora_keeps_fan_in_fan_out_factors_in_matmul_orientation() {
        let directory = write_adapter(
            r#"{
                "r": 2,
                "lora_alpha": 4.0,
                "target_modules": ["down_proj"],
                "fan_in_fan_out": true
            }"#,
            vec![
                (
                    "base_model.model.model.layers.1.mlp.down_proj.lora_A.weight",
                    vec![3, 2],
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ),
                (
                    "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight",
                    vec![2, 4],
                    f32_bytes(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
                ),
            ],
        );

        let adapter = load_peft_adapter(directory.path()).unwrap();
        let module = adapter.modules.values().next().unwrap();
        assert_eq!(module.source_layout, PeftFactorLayout::MatMulReady);
        assert_eq!(module.a_transposed.dims, [3, 2]);
        assert_eq!(module.b_transposed.dims, [2, 4]);
        assert_eq!(
            tensor_f32_values(&module.a_transposed),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn lora_transposes_standard_fp16_factors_with_fan_in_fan_out() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join(CONFIG_FILE),
            r#"{
                "r": 1,
                "lora_alpha": 2.0,
                "target_modules": ["c_attn"],
                "fan_in_fan_out": true
            }"#,
        )
        .unwrap();
        let a_bytes: Vec<u8> = [1.0_f32, 2.0]
            .into_iter()
            .flat_map(|value| half::f16::from_f32(value).to_le_bytes())
            .collect();
        let b_bytes: Vec<u8> = [3.0_f32, 4.0, 5.0]
            .into_iter()
            .flat_map(|value| half::f16::from_f32(value).to_le_bytes())
            .collect();
        let mut views = HashMap::new();
        views.insert(
            "base_model.model.transformer.layers.0.c_attn.lora_A.weight".to_owned(),
            TensorView::new(Dtype::F16, vec![1, 2], &a_bytes).unwrap(),
        );
        views.insert(
            "base_model.model.transformer.layers.0.c_attn.lora_B.weight".to_owned(),
            TensorView::new(Dtype::F16, vec![3, 1], &b_bytes).unwrap(),
        );
        serialize_to_file(&views, None, &directory.path().join(WEIGHTS_FILE)).unwrap();

        let adapter = load_peft_adapter(directory.path()).unwrap();
        let module = adapter.modules.values().next().unwrap();
        assert_eq!(module.source_layout, PeftFactorLayout::Standard);
        assert_eq!(module.a_transposed.dtype, DataType::Float16);
        assert_eq!(module.a_transposed.dims, [2, 1]);
        assert_eq!(module.b_transposed.dims, [1, 3]);
        assert_eq!(module.a_transposed.data, a_bytes);
        assert_eq!(module.b_transposed.data, b_bytes);
    }

    #[test]
    fn lora_reports_missing_factor_as_typed_error() {
        let directory = write_adapter(
            r#"{
                "r": 1,
                "lora_alpha": 1.0,
                "target_modules": ["q_proj"],
                "fan_in_fan_out": false
            }"#,
            vec![(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                vec![1, 2],
                f32_bytes(&[1.0, 2.0]),
            )],
        );

        let error = load_peft_adapter(directory.path()).unwrap_err();
        assert!(matches!(
            error,
            AdapterLoadError::MissingFactor { factor: "B", .. }
        ));
    }

    #[test]
    fn lora_reports_bad_rank_shape_as_typed_error() {
        let directory = write_adapter(
            r#"{
                "r": 2,
                "lora_alpha": 2.0,
                "target_modules": ["q_proj"],
                "fan_in_fan_out": false
            }"#,
            vec![
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    vec![1, 2],
                    f32_bytes(&[1.0, 2.0]),
                ),
                (
                    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    vec![3, 2],
                    f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ),
            ],
        );

        let error = load_peft_adapter(directory.path()).unwrap_err();
        assert!(matches!(error, AdapterLoadError::InvalidShape { .. }));
    }
}
