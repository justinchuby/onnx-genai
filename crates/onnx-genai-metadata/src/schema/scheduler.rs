use super::*;

/// Diffusion scheduler configuration for an iterative strategy.
///
/// The runtime treats the denoiser's loop-carried output as a noise prediction
/// (or, for `flow_matching`, as a vector field and, for `masked_diffusion`, as
/// token logits) and applies one scheduler step per iteration. Supported
/// `kind`s: `ddpm`, `ddim`, `euler`, `euler_ancestral`, `dpmpp_2m`,
/// `flow_matching`, and `masked_diffusion`.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, JsonSchema)]
pub struct SchedulerSpec {
    /// Scheduler algorithm: `"ddpm"`, `"ddim"`, `"euler"`,
    /// `"euler_ancestral"`, `"dpmpp_2m"`, `"flow_matching"`, or
    /// `"masked_diffusion"`.
    pub kind: String,

    /// Training timesteps the noise schedule was defined over (default 1000).
    #[schemars(range(min = 2))]
    pub num_train_timesteps: Option<usize>,

    /// Linear beta-schedule start (default 0.00085).
    #[schemars(range(min = 0.0))]
    pub beta_start: Option<f32>,

    /// Linear beta-schedule end (default 0.012).
    #[schemars(range(min = 0.0))]
    pub beta_end: Option<f32>,

    /// Beta schedule shape: `"linear"` (default) or `"scaled_linear"` (Stable
    /// Diffusion).
    pub beta_schedule: Option<String>,

    /// Model output parameterization: `"epsilon"` (default, noise prediction),
    /// `"v_prediction"` (velocity; SD 2.x, SDXL refiner, many fine-tunes), or
    /// `"sample"`/`"x0"` (the model predicts the clean sample directly). All
    /// built-in diffusion schedulers (`ddpm`, `ddim`, `euler`,
    /// `euler_ancestral`, `dpmpp_2m`) support every parameterization.
    /// `flow_matching` instead consumes the model's velocity/vector-field output
    /// directly and accepts an omitted value or `"flow"`/`"velocity"`.
    pub prediction_type: Option<String>,

    /// Static timestep shift for `flow_matching` (default `1.0`). The base
    /// rectified-flow sigma `s` is transformed to
    /// `shift * s / (1 + (shift - 1) * s)`.
    #[schemars(range(min = 0.0))]
    pub shift: Option<f32>,

    /// Mask token id for a `masked_diffusion` (language-diffusion) scheduler:
    /// each step commits the highest-confidence still-masked positions.
    pub mask_token_id: Option<i64>,

    /// Sampling temperature for a `masked_diffusion` scheduler. `0` (default)
    /// selects each masked position's argmax token deterministically; a positive
    /// value applies Gumbel noise (`logits.exp() / (-log u)^temperature`) before
    /// the argmax, matching LLaDA's `add_gumbel_noise`. Confidence used for
    /// remasking is always the clean-softmax probability of the chosen token.
    pub temperature: Option<f32>,

    /// Semi-autoregressive block length for a `masked_diffusion` scheduler, in
    /// tokens. When set (and smaller than the masked generation region), each
    /// step only commits tokens inside the current left-to-right block, matching
    /// LLaDA's semi-autoregressive remasking. Defaults to a single block
    /// spanning the whole masked region.
    pub block_length: Option<usize>,

    /// Unmasking strategy for a `masked_diffusion` scheduler:
    ///   * `"low_confidence"` (default) — LLaDA: each step commits the
    ///     highest-confidence still-masked positions (confidence-ranked). Best
    ///     for LLaDA checkpoints, but greedy/confidence-ranked decoding of other
    ///     masked-diffusion LMs (e.g. MDLM) collapses into repetitive text.
    ///   * `"random"` — MDLM-style ancestral: each still-masked position unmasks
    ///     independently with the schedule probability `1/(steps_remaining)`,
    ///     sampling its token from the model's categorical distribution (use
    ///     `temperature: 1.0` for a true categorical sample). This per-position
    ///     stochastic unmasking avoids the degenerate loops that confidence
    ///     ranking produces. The mask token is never emitted.
    pub remasking: Option<String>,

    /// Use the Karras (arXiv:2206.00364, rho=7) sigma spacing instead of the
    /// default linspace spacing. Applies to sigma-space schedulers (`euler`,
    /// `dpmpp_2m`); the most popular ComfyUI scheduler for those samplers.
    pub use_karras_sigmas: Option<bool>,

    /// Use the exponential sigma spacing (`exp(linspace(log σ_max, log σ_min))`)
    /// instead of linspace. Applies to `euler`/`dpmpp_2m`. Mutually exclusive
    /// with `use_karras_sigmas` (Karras takes precedence).
    pub use_exponential_sigmas: Option<bool>,
}

/// Pipeline execution strategy family.
///
/// Known values are enumerated while future strings remain valid.
#[derive(Debug, Clone, Default, PartialEq, Eq, JsonSchema)]
#[schemars(
    with = "String",
    transform = schema_helpers::pipeline_strategy_kind
)]
pub enum PipelineStrategyKind {
    /// Token-by-token autoregressive generation.
    #[default]
    Autoregressive,
    /// Repeated denoising or another bounded iterative loop.
    Iterative,
    /// One invocation with no runtime-managed loop.
    SinglePass,
    /// Ordered composition of nested strategies.
    Composite,
    /// Dual, hierarchically-nested autoregressive loops (multi-decoder TTS).
    ///
    /// An outer decoder (talker) AR loop where each outer step drives an inner
    /// decoder (code_predictor) AR loop; see [`PipelineStrategy::outer`],
    /// [`PipelineStrategy::inner`], and [`PipelineStrategy::num_code_groups`].
    NestedAutoregressive,
    /// Future strategy family not recognized by this runtime version.
    Other(String),
}

impl<'de> Deserialize<'de> for PipelineStrategyKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "autoregressive" => Self::Autoregressive,
            "iterative" | "diffusion_steps" | "diffusion-steps" => Self::Iterative,
            "single_pass" | "single-pass" => Self::SinglePass,
            "composite" => Self::Composite,
            "nested_autoregressive" | "nested-autoregressive" => Self::NestedAutoregressive,
            _ => Self::Other(value),
        })
    }
}
