use std::path::Path;

use serde_json::Value;
use timestral::{Array3, Array3F64, ForecastOptions, Forecaster, Input};

fn flat_f32(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            if v.is_null() {
                f32::NAN
            } else {
                v.as_f64().unwrap() as f32
            }
        })
        .collect()
}

fn flat_f64(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            if v.is_null() {
                f64::NAN
            } else {
                v.as_f64().unwrap()
            }
        })
        .collect()
}

fn shape3(value: &Value) -> [usize; 3] {
    let a = value.as_array().unwrap();
    match a.len() {
        1 => [1, 1, a[0].as_u64().unwrap() as usize],
        2 => [
            a[0].as_u64().unwrap() as usize,
            1,
            a[1].as_u64().unwrap() as usize,
        ],
        _ => [
            a[0].as_u64().unwrap() as usize,
            a[1].as_u64().unwrap() as usize,
            a[2].as_u64().unwrap() as usize,
        ],
    }
}

fn optional_array3(case: &Value, key: &str) -> Option<Array3> {
    let value = case.get(key)?;
    if value.is_null() {
        return None;
    }
    let shape = shape3(case.get(format!("{key}_shape")).unwrap());
    Some(Array3::new(flat_f32(value), shape).unwrap())
}

fn optional_array3_f64(case: &Value, key: &str) -> Option<Array3F64> {
    let value = case.get(key)?;
    if value.is_null() {
        return None;
    }
    let shape = shape3(case.get(format!("{key}_shape")).unwrap());
    Some(Array3F64::new(flat_f64(value), shape).unwrap())
}

#[test]
fn matches_pytorch_reference_5m() {
    let model_dir =
        std::env::var("TIMESTRAL_MODEL").unwrap_or_else(|_| "models/Timestral-5M".into());
    run_parity(&model_dir, Path::new("fixtures/reference_5m.json"));
}

#[test]
fn matches_pytorch_reference_28m() {
    let model_dir =
        std::env::var("TIMESTRAL_MODEL_28M").unwrap_or_else(|_| "models/Timestral-28M".into());
    run_parity(&model_dir, Path::new("fixtures/reference_28m.json"));
}

fn run_parity(model_dir: &str, fixture_path: &Path) {
    if !Path::new(model_dir).join("model.safetensors").exists() {
        eprintln!("skipping parity test: model not found at {model_dir}");
        return;
    }
    let forecaster = Forecaster::load(model_dir).expect("load model");
    let payload: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture_path).unwrap()).unwrap();
    let cases = payload.get("cases").unwrap().as_array().unwrap();
    let mut failed = 0;
    for case in cases {
        let name = case.get("name").unwrap().as_str().unwrap();
        let context = Array3::new(
            flat_f32(case.get("context").unwrap()),
            shape3(case.get("context_shape").unwrap()),
        )
        .unwrap();
        let mut input = Input::new(context);
        input.observation_mask = case.get("observation_mask").and_then(|v| {
            if v.is_null() {
                None
            } else {
                Some(flat_f32(v).into_iter().map(|x| x != 0.0).collect())
            }
        });
        input.past_covariates = optional_array3(case, "past_covariates");
        input.known_covariates = optional_array3(case, "known_covariates");
        input.categorical_past_covariates =
            optional_array3_f64(case, "categorical_past_covariates");
        input.categorical_known_covariates =
            optional_array3_f64(case, "categorical_known_covariates");
        let prediction_length = case.get("prediction_length").unwrap().as_u64().unwrap() as usize;
        let options = ForecastOptions {
            symmetric_averaging: case.get("symmetric_averaging").unwrap().as_bool().unwrap(),
            clip_nonnegative: case.get("clip_nonnegative").unwrap().as_bool().unwrap(),
        };
        let forecast = match forecaster.forecast(&input, prediction_length, options) {
            Ok(forecast) => forecast,
            Err(error) => {
                eprintln!("{name}: ERROR {error}");
                failed += 1;
                continue;
            }
        };
        let expected = flat_f32(case.get("quantiles").unwrap());
        assert_eq!(
            forecast.quantiles.len(),
            expected.len(),
            "{name}: output length"
        );
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = 0usize;
        let mut max_scale = 0.0f32;
        for (i, (&got, &want)) in forecast.quantiles.iter().zip(&expected).enumerate() {
            let diff = (got - want).abs();
            if diff > max_abs {
                max_abs = diff;
                worst = i;
            }
            max_rel = max_rel.max(diff / want.abs().max(1e-6));
            max_scale = max_scale.max(want.abs());
        }
        let mono = forecast
            .quantiles
            .chunks_exact(forecast.shape[3])
            .all(|row| row.windows(2).all(|w| w[0] <= w[1]));
        let ok = max_abs <= 2e-3 + 2e-5 * max_scale && mono;
        println!(
            "{name}: max_abs={max_abs:.6e} max_rel={max_rel:.3e} scale={max_scale:.3e} monotone={mono} {}",
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            let q = forecast.shape[3];
            let step = worst / q;
            println!(
                "  worst idx {worst} (step {step}, q {}) got={} want={}",
                worst % q,
                forecast.quantiles[worst],
                expected[worst]
            );
            failed += 1;
        }
    }
    assert_eq!(failed, 0, "{failed} parity cases failed");
}
