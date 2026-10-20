use serde::Deserialize;

use crate::error::{Error, Result};

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_rope_partial_factor() -> f64 {
    0.5
}

fn default_norm_eps() -> f64 {
    1e-6
}

fn default_gate() -> String {
    "sdpa_sigmoid".to_owned()
}

fn default_minimum_scale() -> f64 {
    1e-6
}

fn default_true() -> bool {
    true
}

fn default_quantile_width_factor() -> f64 {
    2.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Time,
    Variate,
    TimeVariate,
    VariateTime,
}

impl LayerKind {
    /// Attention axes run before the shared MLP; `true` means the time axis.
    pub fn axes(self) -> &'static [bool] {
        match self {
            Self::Time => &[true],
            Self::Variate => &[false],
            Self::TimeVariate => &[true, false],
            Self::VariateTime => &[false, true],
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "time" => Ok(Self::Time),
            "variate" => Ok(Self::Variate),
            "time_variate" => Ok(Self::TimeVariate),
            "variate_time" => Ok(Self::VariateTime),
            other => Err(Error(format!(
                "unknown layer kind {other:?}; expected time, variate, time_variate or variate_time"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    None,
    SdpaSigmoid,
}

impl Gate {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "sdpa_sigmoid" => Ok(Self::SdpaSigmoid),
            other => Err(Error(format!(
                "unknown attention gate {other:?}; expected none or sdpa_sigmoid"
            ))),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<String>,
    pub quantile_knots: Vec<f64>,
    pub block_size: usize,
    pub context_length: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_partial_factor")]
    pub rope_partial_factor: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub attn_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_gate")]
    pub time_attn_gate: String,
    #[serde(default = "default_gate")]
    pub variate_attn_gate: String,
    #[serde(default = "default_minimum_scale")]
    pub minimum_scale: f64,
    #[serde(default = "default_true")]
    pub use_asinh: bool,
    #[serde(default)]
    pub stabilize_with_global: bool,
    #[serde(default = "default_true")]
    pub covariate_enabled: bool,
    #[serde(default = "default_true")]
    pub split_context_horizon_attn: bool,
    #[serde(default = "default_true")]
    pub attend_missing_patches: bool,
    #[serde(default = "default_quantile_width_factor")]
    pub quantile_width_factor: f64,
}

impl Config {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&text)?;
        config.validate()?;
        Ok(config)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if !self.d_model.is_multiple_of(self.num_heads) {
            return Err(Error(format!(
                "d_model={} not divisible by num_heads={}",
                self.d_model, self.num_heads
            )));
        }
        if self.head_dim() == 0 || self.rope_dim() == 0 || !self.rope_dim().is_multiple_of(2) {
            return Err(Error(format!(
                "rope_partial_factor={} gives an invalid rope_dim={}",
                self.rope_partial_factor,
                self.rope_dim()
            )));
        }
        if self.patch_size == 0 {
            return Err(Error("patch_size must be positive".into()));
        }
        if self.block_size == 0 {
            return Err(Error("block_size must be positive".into()));
        }
        if self.num_quantiles() == 0 || !self.quantile_knots.contains(&0.5) {
            return Err(Error("quantile_knots must contain 0.5".into()));
        }
        if !self.quantile_knots.windows(2).all(|w| w[0] < w[1]) {
            return Err(Error("quantile_knots must be increasing".into()));
        }
        for kind in &self.block_schedule {
            LayerKind::parse(kind)?;
        }
        Gate::parse(&self.time_attn_gate)?;
        Gate::parse(&self.variate_attn_gate)?;
        if self.stabilize_with_global {
            return Err(Error(
                "the advancing refine frame needs stabilize_with_global=false".into(),
            ));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    pub fn rope_dim(&self) -> usize {
        (self.head_dim() as f64 * self.rope_partial_factor) as usize
    }

    pub fn num_quantiles(&self) -> usize {
        self.quantile_knots.len()
    }

    pub fn median_index(&self) -> usize {
        self.quantile_knots
            .iter()
            .position(|&q| q == 0.5)
            .expect("validated config contains 0.5")
    }

    pub fn layer_kinds(&self) -> Result<Vec<LayerKind>> {
        self.block_schedule
            .iter()
            .map(|kind| LayerKind::parse(kind))
            .collect()
    }

    pub fn time_gate(&self) -> Result<Gate> {
        Gate::parse(&self.time_attn_gate)
    }

    pub fn variate_gate(&self) -> Result<Gate> {
        Gate::parse(&self.variate_attn_gate)
    }

    pub fn supports_known_covariates(&self) -> bool {
        if !self.covariate_enabled {
            return false;
        }
        let ops: Vec<bool> = self
            .block_schedule
            .iter()
            .filter_map(|kind| LayerKind::parse(kind).ok())
            .flat_map(|kind| kind.axes().iter().copied())
            .collect();
        ops.iter().enumerate().any(|(i, &is_time)| {
            let rest = &ops[i + 1..];
            is_time && rest.iter().any(|&x| x) && !rest.iter().all(|&x| x)
        })
    }
}
