//! Runtime-selection flags shared by every command-line entry point.
//!
//! # Why these live here rather than next to each command
//!
//! `generate`, `run`, `transcribe` and `serve` all have to answer the same four
//! questions before a model can load: which decode backend, which device, how
//! much VRAM, how much host RAM. They used to answer them in two different
//! places with two different spellings — the generation commands took
//! `--backend`/`--device`, while `serve` took neither and instead inferred the
//! backend from a `--native-device` flag that existed nowhere else (now
//! `--device`, shared by every subcommand). The result
//! was that a command line learned from `run` was rejected by `serve`, and the
//! flag that did work was undiscoverable from the other subcommands' help.
//!
//! Defining the flags once removes the possibility of that drift: a flag added
//! here appears on every command, with the same name, the same value grammar
//! and the same help text.
//!
//! This crate is the shared home because it is the lowest layer that both the
//! unified CLI and the standalone server binary already depend on, and it
//! already carries `clap` with the `env` feature.

use std::num::NonZeroUsize;

use clap::Args;
use onnx_genai_engine::{EngineConfig, EngineDecodeBackend, ResourceLimit, parse_resource_limit};

#[cfg(feature = "native-backend")]
use onnx_genai_engine::NativeDecodeDevice;

/// Backend, device and memory-ceiling flags.
#[derive(Debug, Args, Default, Clone)]
pub struct EngineArgs {
    /// Decoder backend for text generation.
    #[arg(
        long,
        value_name = "auto|ort|native",
        env = "ONNX_GENAI_BACKEND",
        value_parser = parse_decode_backend,
        default_value = "auto"
    )]
    pub backend: EngineDecodeBackend,

    /// Memory ceiling the engine may use for weights and KV cache: a byte count
    /// (`8GiB`), a fraction of detected capacity (`0.9`), or `auto`.
    ///
    /// An explicit byte value is authoritative — the runtime's device-capacity
    /// probe is still provisional, so this is how you tell it what is really
    /// available. Raising it enlarges the KV cache, and therefore the context
    /// that fits.
    #[arg(long, value_name = "LIMIT", env = "ONNX_GENAI_VRAM_LIMIT", value_parser = parse_limit)]
    pub vram_limit: Option<ResourceLimit>,

    /// Host RAM ceiling for the warm offload tier, in the same format.
    #[arg(long, value_name = "LIMIT", env = "ONNX_GENAI_HOST_RAM_LIMIT", value_parser = parse_limit)]
    pub host_ram_limit: Option<ResourceLimit>,

    /// Device the native decode backend runs on: `cpu`, `cuda`, `cuda:N`, or
    /// `auto`.
    ///
    /// `auto` (the default) takes the device from the model's declared execution
    /// providers. Most exported models declare none, which resolves to the CPU —
    /// so on a machine with a GPU, `--backend native` alone will still run on the
    /// CPU unless you say `--device cuda` (#1064). Ignored by the ORT backend,
    /// which selects providers from the model's own session options.
    #[arg(
        long,
        value_name = "auto|cpu|cuda[:N]",
        env = "ONNX_GENAI_DEVICE",
        value_parser = parse_device
    )]
    pub device: Option<DeviceChoice>,
}

/// A `--device` value. `Auto` is distinct from an absent flag only in intent;
/// both defer to the model's declared providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceChoice {
    /// Take the device from the model's declared execution providers.
    Auto,
    /// Decode on the CPU.
    Cpu,
    /// Decode on CUDA, optionally on a named device index.
    Cuda(Option<u32>),
}

/// Parse a `--device` value.
///
/// Refuses anything else rather than silently falling back: a user who asks for
/// a device they cannot have should be told, not quietly given the CPU. That
/// silent fallback is exactly what #1064 documents.
pub fn parse_device(input: &str) -> Result<DeviceChoice, String> {
    let raw = input.trim();
    let lowered = raw.to_ascii_lowercase();
    match lowered.as_str() {
        "auto" => Ok(DeviceChoice::Auto),
        "cpu" => Ok(DeviceChoice::Cpu),
        "cuda" | "gpu" => Ok(DeviceChoice::Cuda(None)),
        _ => match lowered.strip_prefix("cuda:") {
            Some(index) => index
                .parse::<u32>()
                .map(|index| DeviceChoice::Cuda(Some(index)))
                .map_err(|_| {
                    format!("'{raw}' is not a valid device: expected a CUDA index, as in 'cuda:0'")
                }),
            None => Err(format!(
                "'{raw}' is not a valid device: expected 'auto', 'cpu', 'cuda', or 'cuda:N'"
            )),
        },
    }
}

/// Parse a `--backend` value.
pub fn parse_decode_backend(name: &str) -> Result<EngineDecodeBackend, String> {
    match name {
        "auto" => Ok(EngineDecodeBackend::Auto),
        "ort" => Ok(EngineDecodeBackend::Ort),
        "native" => Ok(EngineDecodeBackend::Native),
        other => Err(format!(
            "What: {other:?} is not a decode backend. \
             Why: the choices are fixed by the engine, not by the model. \
             How: use auto, ort, or native."
        )),
    }
}

/// The name `--backend` would have been given for `backend`.
pub fn decode_backend_name(backend: EngineDecodeBackend) -> &'static str {
    match backend {
        EngineDecodeBackend::Auto => "auto",
        EngineDecodeBackend::Ort => "ort",
        EngineDecodeBackend::Native => "native",
    }
}

/// Parse a `--vram-limit` / `--host-ram-limit` value.
pub fn parse_limit(input: &str) -> Result<ResourceLimit, String> {
    parse_resource_limit(input).map_err(|error| {
        format!(
            "What: the memory limit {input:?} was rejected. \
             Why: {error}. \
             How: pass a byte count such as 8GiB, a fraction such as 0.9, or auto."
        )
    })
}

impl EngineArgs {
    /// Fold these flags into an [`EngineConfig`].
    pub fn to_config(&self) -> EngineConfig {
        let mut config = EngineConfig {
            decode_backend: self.backend,
            ..EngineConfig::default()
        };
        if let Some(limit) = self.vram_limit {
            config.limits.vram_limit = limit;
        }
        if let Some(limit) = self.host_ram_limit {
            config.limits.host_ram_limit = limit;
        }
        self.apply_device(&mut config);
        config
    }

    /// Naming a device implies the native backend.
    ///
    /// `--device cuda:0` is only meaningful to the native decoder, so asking for
    /// one and leaving `--backend` at `auto` used to run on ORT and ignore the
    /// device silently. `serve` already worked this way — naming a device
    /// selected the backend as a side effect — and that behaviour is the one
    /// users expect, so it now applies everywhere.
    pub fn resolved_backend(&self) -> EngineDecodeBackend {
        match (self.backend, self.device) {
            (EngineDecodeBackend::Auto, Some(DeviceChoice::Cpu | DeviceChoice::Cuda(_))) => {
                EngineDecodeBackend::Native
            }
            (backend, _) => backend,
        }
    }

    #[cfg(feature = "native-backend")]
    fn apply_device(&self, config: &mut EngineConfig) {
        config.decode_backend = self.resolved_backend();
        match self.device {
            None | Some(DeviceChoice::Auto) => {}
            Some(DeviceChoice::Cpu) => {
                config.native_device = Some(NativeDecodeDevice::Cpu);
            }
            Some(DeviceChoice::Cuda(index)) => {
                config.native_device = Some(NativeDecodeDevice::Cuda { index });
            }
        }
    }

    #[cfg(not(feature = "native-backend"))]
    fn apply_device(&self, _config: &mut EngineConfig) {}
}

/// CPU resource controls.
#[derive(Debug, Args, Default, Clone)]
pub struct CpuArgs {
    /// Cap native CPU decode to N worker cores. Overrides
    /// ONNX_GENAI_CPU_DECODE_THREADS; when neither is set, automatic sizing is
    /// unchanged. Setting N now also bounds prefill/MLAS: the global Rayon pool
    /// is built with N workers (not all logical CPUs) and, on Linux, the process
    /// is pinned to N CPUs (packed on one NUMA node where possible), so
    /// `--cpu-cores N` alone makes the engine coexist with other programs -- no
    /// external `taskset` needed. An explicit ONNX_GENAI_CPU_DECODE_AFFINITY
    /// still wins over the automatic pinning.
    #[arg(long, value_name = "N", env = "ONNX_GENAI_CPU_CORES")]
    pub cpu_cores: Option<NonZeroUsize>,
}

impl CpuArgs {
    /// Install the requested CPU budget process-wide.
    pub fn apply(&self) -> Result<(), String> {
        #[cfg(feature = "native-backend")]
        {
            onnx_genai_engine::set_cpu_decode_thread_budget(self.cpu_cores.map(NonZeroUsize::get))?;
        }
        #[cfg(not(feature = "native-backend"))]
        {
            let _ = self.cpu_cores;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--device` exists because `--backend native` alone silently ran on the CPU
    /// on a GPU machine: device selection came only from the model's declared
    /// execution providers, and typical exports declare none (#1064). Measured
    /// 0 MiB of GPU memory across a whole native run before this flag existed.
    #[test]
    fn a_device_can_be_asked_for_explicitly_and_nonsense_is_refused() {
        assert_eq!(parse_device("auto"), Ok(DeviceChoice::Auto));
        assert_eq!(parse_device("cpu"), Ok(DeviceChoice::Cpu));
        assert_eq!(parse_device("cuda"), Ok(DeviceChoice::Cuda(None)));
        assert_eq!(parse_device("gpu"), Ok(DeviceChoice::Cuda(None)));
        assert_eq!(
            parse_device("CUDA:1"),
            Ok(DeviceChoice::Cuda(Some(1))),
            "device names are case-insensitive"
        );

        // Refused rather than quietly resolved to the CPU: a silent fallback is
        // the behaviour this flag exists to end.
        let error = parse_device("cuda:x").expect_err("not a device index");
        assert!(error.contains("cuda:0"), "{error}");
        let error = parse_device("tpu").expect_err("not a device");
        assert!(
            error.contains("'auto', 'cpu', 'cuda', or 'cuda:N'"),
            "{error}"
        );
    }

    #[test]
    fn decode_backends_are_named_by_the_engine_not_guessed() {
        assert_eq!(parse_decode_backend("auto"), Ok(EngineDecodeBackend::Auto));
        assert_eq!(parse_decode_backend("ort"), Ok(EngineDecodeBackend::Ort));
        assert_eq!(
            parse_decode_backend("native"),
            Ok(EngineDecodeBackend::Native)
        );
        let error = parse_decode_backend("cuda").expect_err("not a backend");
        assert!(error.contains("auto, ort, or native"), "{error}");
        assert_eq!(decode_backend_name(EngineDecodeBackend::Native), "native");
    }

    /// The flag has to reach `EngineConfig`, not merely parse. Absent and `auto`
    /// both leave the engine's own resolution untouched.
    #[cfg(feature = "native-backend")]
    #[test]
    fn the_device_flag_reaches_the_engine_config() {
        assert!(
            EngineArgs::default().to_config().native_device.is_none(),
            "an absent --device must not override the model's declared providers"
        );

        let auto = EngineArgs {
            device: Some(DeviceChoice::Auto),
            ..EngineArgs::default()
        };
        assert!(auto.to_config().native_device.is_none());

        let cuda = EngineArgs {
            device: Some(DeviceChoice::Cuda(Some(1))),
            ..EngineArgs::default()
        };
        assert_eq!(
            cuda.to_config().native_device,
            Some(NativeDecodeDevice::Cuda { index: Some(1) })
        );

        let cpu = EngineArgs {
            device: Some(DeviceChoice::Cpu),
            ..EngineArgs::default()
        };
        assert_eq!(cpu.to_config().native_device, Some(NativeDecodeDevice::Cpu));
    }

    #[test]
    fn naming_a_device_selects_the_native_backend() {
        let args = EngineArgs {
            device: Some(DeviceChoice::Cuda(Some(6))),
            ..EngineArgs::default()
        };
        assert_eq!(args.resolved_backend(), EngineDecodeBackend::Native);
    }

    #[test]
    fn an_explicit_backend_is_not_overridden_by_a_device() {
        let args = EngineArgs {
            backend: EngineDecodeBackend::Ort,
            device: Some(DeviceChoice::Cuda(None)),
            ..EngineArgs::default()
        };
        assert_eq!(args.resolved_backend(), EngineDecodeBackend::Ort);
    }

    #[test]
    fn no_device_leaves_the_backend_alone() {
        assert_eq!(
            EngineArgs::default().resolved_backend(),
            EngineDecodeBackend::Auto
        );
    }
}
