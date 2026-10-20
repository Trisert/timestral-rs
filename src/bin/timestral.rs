use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value};
use timestral::{Array3, Array3F64, Error, ForecastOptions, Forecaster, Input, Result};

const BOOL_FLAGS: &[&str] = &["no-sym", "no-clip", "json", "help"];

struct Args {
    command: String,
    values: HashMap<String, String>,
    flags: HashSet<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut argv = std::env::args().skip(1);
        let command = argv.next().unwrap_or_else(|| "help".to_string());
        let rest: Vec<String> = argv.collect();
        let mut values = HashMap::new();
        let mut flags = HashSet::new();
        let mut i = 0;
        while i < rest.len() {
            let arg = &rest[i];
            let Some(body) = arg.strip_prefix("--") else {
                return Err(Error(format!("unexpected argument '{arg}'")));
            };
            if let Some((key, value)) = body.split_once('=') {
                values.insert(key.to_string(), value.to_string());
            } else if BOOL_FLAGS.contains(&body) {
                flags.insert(body.to_string());
            } else {
                i += 1;
                let value = rest
                    .get(i)
                    .ok_or_else(|| Error(format!("missing value for --{body}")))?;
                values.insert(body.to_string(), value.clone());
            }
            i += 1;
        }
        Ok(Self {
            command,
            values,
            flags,
        })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn flag(&self, key: &str) -> bool {
        self.flags.contains(key)
    }

    fn number<T: std::str::FromStr>(&self, key: &str, default: T) -> Result<T> {
        match self.get(key) {
            None => Ok(default),
            Some(text) => text
                .parse()
                .map_err(|_| Error(format!("invalid value '{text}' for --{key}"))),
        }
    }

    fn required(&self, key: &str) -> Result<&str> {
        self.get(key)
            .ok_or_else(|| Error(format!("missing required --{key}")))
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    match args.command.as_str() {
        "forecast" => forecast_command(&args),
        "bench" => bench_command(&args),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(Error(format!("unknown command '{other}'"))),
    }
}

fn print_help() {
    println!(
        "\
timestral - CPU inference for Timestral time-series models

USAGE:
  timestral forecast --input FILE [options]
  timestral bench [options]

FORECAST OPTIONS:
  --model DIR        Model directory (default: models/Timestral-5M)
  --input FILE       JSON input; accepts a fixture case or {{\"cases\": [...]}}
  --case NAME        Case to select when the input contains several cases
  --horizon N        Prediction length (default: input prediction_length)
  --output FILE      Write JSON output to FILE instead of stdout
  --threads N        Rayon worker threads (default: available cores)
  --no-sym           Disable symmetric averaging
  --no-clip          Disable non-negative clipping

BENCH OPTIONS:
  --model DIR        Model directory (default: models/Timestral-5M)
  --context N        Context length (default: 2016)
  --horizon N        Prediction length (default: 96)
  --batch N          Number of series (default: 1)
  --variates N       Variates per series (default: 1)
  --iters N          Timed iterations (default: 30)
  --warmup N         Warmup iterations (default: 3)
  --threads N        Rayon worker threads (default: available cores)
  --no-sym           Disable symmetric averaging
  --no-clip          Disable non-negative clipping
  --json             Print machine-readable results"
    );
}

fn configure_threads(args: &Args) -> Result<()> {
    if let Some(text) = args.get("threads") {
        let threads: usize = text
            .parse()
            .map_err(|_| Error(format!("invalid --threads value '{text}'")))?;
        if threads == 0 {
            return Err(Error("--threads must be >= 1".into()));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| Error(format!("failed to configure threads: {error}")))?;
    }
    Ok(())
}

fn forecast_command(args: &Args) -> Result<()> {
    configure_threads(args)?;
    let model_dir = args.get("model").unwrap_or("models/Timestral-5M");
    let input_path = PathBuf::from(args.required("input")?);
    let text = std::fs::read_to_string(&input_path)
        .map_err(|error| Error(format!("cannot read {}: {error}", input_path.display())))?;
    let document: Value = serde_json::from_str(&text)
        .map_err(|error| Error(format!("invalid JSON in {}: {error}", input_path.display())))?;
    let case = select_case(&document, args.get("case"))?;
    let input = parse_input(case)?;
    let horizon = match args.get("horizon") {
        Some(text) => text
            .parse()
            .map_err(|_| Error(format!("invalid --horizon value '{text}'")))?,
        None => case
            .get("prediction_length")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| Error("missing --horizon and input has no prediction_length".into()))?,
    };
    let options = ForecastOptions {
        symmetric_averaging: case
            .get("symmetric_averaging")
            .and_then(Value::as_bool)
            .unwrap_or(true)
            && !args.flag("no-sym"),
        clip_nonnegative: case
            .get("clip_nonnegative")
            .and_then(Value::as_bool)
            .unwrap_or(true)
            && !args.flag("no-clip"),
    };

    let started = Instant::now();
    let forecaster = Forecaster::load(model_dir)?;
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    let started = Instant::now();
    let forecast = forecaster.forecast(&input, horizon, options)?;
    let forecast_ms = started.elapsed().as_secs_f64() * 1e3;

    let output = json!({
        "model": model_dir,
        "case": case.get("name").and_then(Value::as_str),
        "load_ms": load_ms,
        "forecast_ms": forecast_ms,
        "shape": forecast.shape,
        "levels": forecast.levels,
        "median_index": forecast.median_index,
        "median": forecast.median(),
        "quantiles": forecast.quantiles,
    });
    let rendered = match args.get("output") {
        Some(path) => {
            let text = serde_json::to_string_pretty(&output)
                .map_err(|error| Error(format!("cannot encode output: {error}")))?;
            std::fs::write(path, text)
                .map_err(|error| Error(format!("cannot write {path}: {error}")))?;
            format!(
                "wrote {} (load {:.1} ms, forecast {:.1} ms)",
                path, load_ms, forecast_ms
            )
        }
        None => serde_json::to_string(&output)
            .map_err(|error| Error(format!("cannot encode output: {error}")))?,
    };
    println!("{rendered}");
    Ok(())
}

fn bench_command(args: &Args) -> Result<()> {
    configure_threads(args)?;
    let model_dir = args.get("model").unwrap_or("models/Timestral-5M");
    let context = args.number("context", 2016usize)?;
    let horizon = args.number("horizon", 96usize)?;
    let batch = args.number("batch", 1usize)?;
    let variates = args.number("variates", 1usize)?;
    let iters = args.number("iters", 30usize)?;
    let warmup = args.number("warmup", 3usize)?;
    if context == 0 || horizon == 0 || batch == 0 || variates == 0 || iters == 0 {
        return Err(Error(
            "context, horizon, batch, variates and iters must be >= 1".into(),
        ));
    }
    let options = ForecastOptions {
        symmetric_averaging: !args.flag("no-sym"),
        clip_nonnegative: !args.flag("no-clip"),
    };
    let input = Input::new(synthetic(batch, variates, context));

    let started = Instant::now();
    let forecaster = Forecaster::load(model_dir)?;
    let load_ms = started.elapsed().as_secs_f64() * 1e3;

    for _ in 0..warmup {
        let forecast = forecaster.forecast(&input, horizon, options)?;
        std::hint::black_box(&forecast);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        let forecast = forecaster.forecast(&input, horizon, options)?;
        std::hint::black_box(&forecast);
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = percentile(&samples, 0.5);
    let p90 = percentile(&samples, 0.9);
    let threads = rayon::current_num_threads();
    let config = &forecaster.model.config;

    if args.flag("json") {
        let output = json!({
            "model": model_dir,
            "d_model": config.d_model,
            "layers": config.block_schedule.len(),
            "context": context,
            "horizon": horizon,
            "batch": batch,
            "variates": variates,
            "symmetric_averaging": options.symmetric_averaging,
            "clip_nonnegative": options.clip_nonnegative,
            "threads": threads,
            "load_ms": load_ms,
            "warmup": warmup,
            "iters": iters,
            "min_ms": samples.first().copied().unwrap_or(0.0),
            "mean_ms": mean,
            "median_ms": median,
            "p90_ms": p90,
            "max_ms": samples.last().copied().unwrap_or(0.0),
        });
        println!(
            "{}",
            serde_json::to_string(&output)
                .map_err(|error| Error(format!("cannot encode output: {error}")))?
        );
    } else {
        println!(
            "model:    {model_dir} (d_model={}, layers={}, quantiles={})",
            config.d_model,
            config.block_schedule.len(),
            config.num_quantiles()
        );
        println!(
            "input:    batch={batch} variates={variates} context={context} horizon={horizon} sym={} threads={threads}",
            options.symmetric_averaging
        );
        println!("load:     {load_ms:.1} ms");
        println!(
            "latency:  min {:.2} ms | mean {:.2} ms | median {:.2} ms | p90 {:.2} ms | max {:.2} ms",
            samples.first().copied().unwrap_or(0.0),
            mean,
            median,
            p90,
            samples.last().copied().unwrap_or(0.0)
        );
        println!("runs:     {warmup} warmup + {iters} timed");
    }
    Ok(())
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn synthetic(batch: usize, variates: usize, length: usize) -> Array3 {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let noise = |state: &mut u64| {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0) as f32
    };
    let mut data = Vec::with_capacity(batch * variates * length);
    for b in 0..batch {
        for v in 0..variates {
            let phase = 0.7 * (b * variates + v) as f32;
            let scale = 1.0 + 0.5 * v as f32;
            for t in 0..length {
                let seasonal =
                    10.0 * scale * (std::f32::consts::TAU * t as f32 / 96.0 + phase).sin();
                let trend = 0.01 * t as f32;
                data.push(seasonal + trend + 2.0 * noise(&mut state));
            }
        }
    }
    Array3::new(data, [batch, variates, length]).expect("synthetic shape is valid")
}

fn select_case<'a>(document: &'a Value, name: Option<&str>) -> Result<&'a Value> {
    if let Some(cases) = document.get("cases").and_then(Value::as_array) {
        if let Some(name) = name {
            return cases
                .iter()
                .find(|case| case.get("name").and_then(Value::as_str) == Some(name))
                .ok_or_else(|| Error(format!("no case named '{name}'")));
        }
        return cases
            .first()
            .ok_or_else(|| Error("input has an empty cases array".into()));
    }
    Ok(document)
}

fn parse_input(case: &Value) -> Result<Input> {
    let context = numeric_array(case, "context", ShapeKind::Context)?;
    let observation_mask = parse_mask(case.get("observation_mask"), context.data.len())?;
    let past_covariates = optional_numeric_array(case, "past_covariates", ShapeKind::Covariate)?;
    let known_covariates = optional_numeric_array(case, "known_covariates", ShapeKind::Covariate)?;
    let categorical_past_covariates =
        optional_numeric_array_f64(case, "categorical_past_covariates", ShapeKind::Covariate)?;
    let categorical_known_covariates =
        optional_numeric_array_f64(case, "categorical_known_covariates", ShapeKind::Covariate)?;
    Ok(Input {
        context,
        observation_mask,
        past_covariates,
        known_covariates,
        categorical_past_covariates,
        categorical_known_covariates,
    })
}

enum ShapeKind {
    Context,
    Covariate,
}

fn numeric_array(case: &Value, key: &str, kind: ShapeKind) -> Result<Array3> {
    let value = case
        .get(key)
        .ok_or_else(|| Error(format!("input is missing '{key}'")))?;
    let mut data = Vec::new();
    flatten_f32(value, &mut data)?;
    let shape = explicit_shape(case, key)
        .or_else(|| infer_shape(value))
        .map(|shape| normalize_shape(shape, &kind))
        .ok_or_else(|| Error(format!("cannot infer shape of '{key}'")))?;
    Array3::new(data, shape3(shape)?)
}

fn optional_numeric_array(case: &Value, key: &str, kind: ShapeKind) -> Result<Option<Array3>> {
    if case.get(key).is_none_or(Value::is_null) {
        return Ok(None);
    }
    numeric_array(case, key, kind).map(Some)
}

fn optional_numeric_array_f64(
    case: &Value,
    key: &str,
    kind: ShapeKind,
) -> Result<Option<Array3F64>> {
    if case.get(key).is_none_or(Value::is_null) {
        return Ok(None);
    }
    let value = case.get(key).unwrap();
    let mut data = Vec::new();
    flatten_f64(value, &mut data)?;
    let shape = explicit_shape(case, key)
        .or_else(|| infer_shape(value))
        .map(|shape| normalize_shape(shape, &kind))
        .ok_or_else(|| Error(format!("cannot infer shape of '{key}'")))?;
    Array3F64::new(data, shape3(shape)?).map(Some)
}

fn explicit_shape(case: &Value, key: &str) -> Option<Vec<usize>> {
    case.get(format!("{key}_shape"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .map(|value| value as usize)
                .collect()
        })
}

fn normalize_shape(shape: Vec<usize>, kind: &ShapeKind) -> Vec<usize> {
    match kind {
        ShapeKind::Context => match shape.len() {
            1 => vec![1, 1, shape[0]],
            2 => vec![shape[0], 1, shape[1]],
            _ => shape,
        },
        ShapeKind::Covariate => match shape.len() {
            1 => vec![1, 1, shape[0]],
            2 => vec![1, shape[0], shape[1]],
            _ => shape,
        },
    }
}

fn shape3(shape: Vec<usize>) -> Result<[usize; 3]> {
    if shape.len() != 3 {
        return Err(Error(format!(
            "expected a 3-dimensional shape, got {shape:?}"
        )));
    }
    Ok([shape[0], shape[1], shape[2]])
}

fn infer_shape(value: &Value) -> Option<Vec<usize>> {
    let array = value.as_array()?;
    if let Some(first) = array.first() {
        if first.is_array() {
            let mut shape = vec![array.len()];
            shape.extend(infer_shape(first)?);
            return Some(shape);
        }
    } else {
        return None;
    }
    Some(vec![array.len()])
}

fn flatten_f32(value: &Value, out: &mut Vec<f32>) -> Result<()> {
    match value {
        Value::Array(values) => {
            for value in values {
                flatten_f32(value, out)?;
            }
            Ok(())
        }
        Value::Null => {
            out.push(f32::NAN);
            Ok(())
        }
        other => {
            let number = other
                .as_f64()
                .ok_or_else(|| Error("array contains a non-numeric value".into()))?;
            out.push(number as f32);
            Ok(())
        }
    }
}

fn flatten_f64(value: &Value, out: &mut Vec<f64>) -> Result<()> {
    match value {
        Value::Array(values) => {
            for value in values {
                flatten_f64(value, out)?;
            }
            Ok(())
        }
        Value::Null => {
            out.push(f64::NAN);
            Ok(())
        }
        other => {
            let number = other
                .as_f64()
                .ok_or_else(|| Error("array contains a non-numeric value".into()))?;
            out.push(number);
            Ok(())
        }
    }
}

fn parse_mask(value: Option<&Value>, expected: usize) -> Result<Option<Vec<bool>>> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let mut flat = Vec::new();
    flatten_mask(value, &mut flat)?;
    if flat.len() != expected {
        return Err(Error(format!(
            "observation_mask has {} values, expected {expected}",
            flat.len()
        )));
    }
    Ok(Some(flat))
}

fn flatten_mask(value: &Value, out: &mut Vec<bool>) -> Result<()> {
    match value {
        Value::Array(values) => {
            for value in values {
                flatten_mask(value, out)?;
            }
            Ok(())
        }
        Value::Null => {
            out.push(false);
            Ok(())
        }
        Value::Bool(value) => {
            out.push(*value);
            Ok(())
        }
        other => {
            let number = other
                .as_f64()
                .ok_or_else(|| Error("observation_mask contains a non-numeric value".into()))?;
            out.push(number != 0.0);
            Ok(())
        }
    }
}
