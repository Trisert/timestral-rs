//! CPU inference for the Timestral family of time-series foundation models.

pub mod attention;
pub mod config;
pub mod covariate;
pub mod error;
pub mod forecaster;
pub mod model;
pub mod ops;
pub mod safetensors;

pub use config::Config;
pub use error::{Error, Result};
pub use forecaster::{Array3, Array3F64, Forecast, ForecastOptions, Forecaster, Input};
pub use model::Model;
