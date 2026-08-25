//! Artifact layout of the package a converted workflow runs against.
//!
//! The importer never creates ONNX graphs. It names the components a canonical
//! diffusion workflow invokes and the artifact each one loads, which is exactly
//! the layout Mobius's diffusion exporter already writes and the checked
//! `tests/fixtures/onnx_genai_workflows/diffusion*` packages already use.
//!
//! Every path is overridable so a package with a different on-disk layout can
//! still be imported without the importer guessing. What is *not* overridable
//! is the component ABI: port names and contracts are fixed, because they are
//! what makes the emitted workflow executable by the generic runtime.

/// Where each canonical component's ONNX artifact lives inside the package.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentLayout {
    /// Primary text encoder, without its file extension.
    pub text_encoder: String,
    /// Second text encoder of an SDXL dual-encoder package.
    pub text_encoder_2: String,
    /// Denoiser (UNet or DiT).
    pub denoiser: String,
    /// VAE decoder.
    pub vae_decoder: String,
    /// VAE encoder, used by image-to-image and inpainting.
    pub vae_encoder: String,
    /// ControlNet residual producer.
    pub controlnet: String,
    /// Directory holding the schedule/solver policy graphs.
    pub policies: String,
    /// File extension every artifact path ends in, without a leading dot.
    ///
    /// Checked-in fixtures are ONNX protobuf **TextFormat** (`onnx.textproto`)
    /// so a package under review is readable in a diff. The runtime accepts
    /// either encoding, so this is a packaging choice rather than a semantic
    /// one, and it belongs to the layout instead of the lowering.
    pub extension: String,
}

impl Default for ComponentLayout {
    fn default() -> Self {
        Self {
            text_encoder: "text_encoder/model".to_owned(),
            text_encoder_2: "text_encoder_2/model".to_owned(),
            denoiser: "denoiser/model".to_owned(),
            vae_decoder: "vae_decoder/model".to_owned(),
            vae_encoder: "vae_encoder/model".to_owned(),
            controlnet: "controlnet/model".to_owned(),
            policies: "policies".to_owned(),
            extension: "onnx".to_owned(),
        }
    }
}

impl ComponentLayout {
    /// Layout whose artifacts are ONNX protobuf TextFormat documents.
    pub fn textproto() -> Self {
        Self {
            extension: "onnx.textproto".to_owned(),
            ..Self::default()
        }
    }

    /// Full artifact path of one component stem.
    pub fn artifact(&self, stem: &str) -> String {
        format!("{stem}.{}", self.extension.trim_start_matches('.'))
    }

    /// Full artifact path of one policy graph inside the package.
    pub fn policy(&self, name: &str) -> String {
        self.artifact(&format!("{}/{name}", self.policies.trim_end_matches('/')))
    }
}
