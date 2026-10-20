use std::path::Path;

use crate::covariate::target_encode;
use crate::error::{Error, Result};
use crate::model::{refine_block, Model, ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET};

#[derive(Clone, Debug)]
pub struct Array3 {
    pub shape: [usize; 3],
    pub data: Vec<f32>,
}

impl Array3 {
    pub fn new(data: Vec<f32>, shape: [usize; 3]) -> Result<Self> {
        if data.len() != shape[0] * shape[1] * shape[2] {
            return Err(Error(format!(
                "array shape {shape:?} does not match {} values",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }
}

#[derive(Clone, Debug)]
pub struct Array3F64 {
    pub shape: [usize; 3],
    pub data: Vec<f64>,
}

impl Array3F64 {
    pub fn new(data: Vec<f64>, shape: [usize; 3]) -> Result<Self> {
        if data.len() != shape[0] * shape[1] * shape[2] {
            return Err(Error(format!(
                "array shape {shape:?} does not match {} values",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    /// `(series, variates, time)`; NaN marks a missing value.
    pub context: Array3,
    /// Optional boolean mask shaped like `context`; false marks an additional missing value.
    pub observation_mask: Option<Vec<bool>>,
    /// `(series, channels, context length)`.
    pub past_covariates: Option<Array3>,
    /// `(series, channels, context length + prediction length)`.
    pub known_covariates: Option<Array3>,
    /// Category codes over the context, NaN = missing.
    pub categorical_past_covariates: Option<Array3F64>,
    /// Category codes over context and horizon, NaN = missing.
    pub categorical_known_covariates: Option<Array3F64>,
}

impl Input {
    pub fn new(context: Array3) -> Self {
        Self {
            context,
            observation_mask: None,
            past_covariates: None,
            known_covariates: None,
            categorical_past_covariates: None,
            categorical_known_covariates: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ForecastOptions {
    pub symmetric_averaging: bool,
    pub clip_nonnegative: bool,
}

impl Default for ForecastOptions {
    fn default() -> Self {
        Self {
            symmetric_averaging: true,
            clip_nonnegative: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Forecast {
    /// `(series, variates, horizon, quantiles)`, non-decreasing along the quantile axis.
    pub quantiles: Vec<f32>,
    pub shape: [usize; 4],
    pub levels: Vec<f64>,
    pub median_index: usize,
}

impl Forecast {
    pub fn median(&self) -> Vec<f32> {
        let [b, v, h, q] = self.shape;
        let mut out = Vec::with_capacity(b * v * h);
        for i in 0..b * v * h {
            out.push(self.quantiles[i * q + self.median_index]);
        }
        out
    }

    pub fn quantile(&self, level: f64) -> Result<Vec<f32>> {
        let index = self
            .levels
            .iter()
            .position(|&value| value == level)
            .ok_or_else(|| Error(format!("level {level} is not one of the model quantiles")))?;
        let [b, v, h, q] = self.shape;
        let mut out = Vec::with_capacity(b * v * h);
        for i in 0..b * v * h {
            out.push(self.quantiles[i * q + index]);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct CovData {
    batch: usize,
    channels: usize,
    len: usize,
    /// `(batch, channels, len)` with NaN retained.
    data: Vec<f32>,
}

impl CovData {
    fn negated(&self) -> Self {
        Self {
            batch: self.batch,
            channels: self.channels,
            len: self.len,
            data: self.data.iter().map(|&v| -v).collect(),
        }
    }
}

fn concat_channels(first: Option<CovData>, second: Option<CovData>) -> Result<Option<CovData>> {
    match (first, second) {
        (None, None) => Ok(None),
        (Some(a), None) => Ok(Some(a)),
        (None, Some(b)) => Ok(Some(b)),
        (Some(a), Some(b)) => {
            if a.batch != b.batch || a.len != b.len {
                return Err(Error("covariate shapes do not align".into()));
            }
            let channels = a.channels + b.channels;
            let mut data = vec![0.0f32; a.batch * channels * a.len];
            for batch in 0..a.batch {
                for channel in 0..a.channels {
                    let src = (batch * a.channels + channel) * a.len;
                    let dst = (batch * channels + channel) * a.len;
                    data[dst..dst + a.len].copy_from_slice(&a.data[src..src + a.len]);
                }
                for channel in 0..b.channels {
                    let src = (batch * b.channels + channel) * b.len;
                    let dst = (batch * channels + a.channels + channel) * b.len;
                    data[dst..dst + b.len].copy_from_slice(&b.data[src..src + b.len]);
                }
            }
            Ok(Some(CovData {
                batch: a.batch,
                channels,
                len: a.len,
                data,
            }))
        }
    }
}

fn numeric_covariates(
    array: &Option<Array3>,
    batch: usize,
    len: usize,
    name: &str,
    supported: bool,
) -> Result<Option<CovData>> {
    let Some(array) = array else {
        return Ok(None);
    };
    if !supported {
        return Err(Error(format!(
            "this checkpoint was not trained with covariates; got {name}"
        )));
    }
    if array.shape[0] != batch || array.shape[2] != len {
        return Err(Error(format!(
            "{name} must be ({batch}, C, {len}), got {:?}",
            array.shape
        )));
    }
    if array.shape[1] == 0 {
        return Ok(None);
    }
    Ok(Some(CovData {
        batch,
        channels: array.shape[1],
        len,
        data: array.data.clone(),
    }))
}

fn ffill_row(inputs: &[f32], mask: &[bool]) -> Vec<f32> {
    let mut out = vec![0.0f32; inputs.len()];
    let mut last = 0.0f32;
    for i in 0..inputs.len() {
        if mask[i] {
            last = inputs[i];
        }
        out[i] = last;
    }
    out
}

fn encoding_target(
    inputs: &[f32],
    mask: &[bool],
    batch: usize,
    t: usize,
    max_context: usize,
) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &inputs[b * t..(b + 1) * t];
        let row_mask = &mask[b * t..(b + 1) * t];
        let filled = ffill_row(row, row_mask);
        let mut started = false;
        let mut target = vec![f64::NAN; t];
        for i in 0..t {
            started |= row_mask[i];
            let in_window = i + max_context >= t;
            if started && in_window {
                target[i] = filled[i] as f64;
            }
        }
        out.push(target);
    }
    out
}

fn encode_categorical(
    codes: &Array3F64,
    target: Option<&[Vec<f64>]>,
    batch: usize,
    t: usize,
) -> Result<CovData> {
    let channels = codes.shape[1];
    let len = codes.shape[2];
    if codes.shape[0] != batch || len < t {
        return Err(Error("categorical covariate shape mismatch".into()));
    }
    let mut data = vec![f32::NAN; batch * channels * len];
    for b in 0..batch {
        for c in 0..channels {
            let base = (b * channels + c) * len;
            let past = &codes.data[base..base + t];
            let future = if len > t {
                Some(&codes.data[base + t..base + len])
            } else {
                None
            };
            if let Some(targets) = target {
                let (past_encoded, future_encoded) = target_encode(&targets[b], past, future, 1.0);
                let dst = (b * channels + c) * len;
                for i in 0..t {
                    data[dst + i] = if targets[b][i].is_nan() && past[i].is_nan() {
                        f32::NAN
                    } else {
                        past_encoded[i]
                    };
                }
                if let Some(future_encoded) = future_encoded {
                    data[dst + t..dst + len].copy_from_slice(&future_encoded);
                }
            } else {
                let dst = (b * channels + c) * len;
                for i in 0..len {
                    data[dst + i] = codes.data[base + i] as f32;
                }
            }
        }
    }
    Ok(CovData {
        batch,
        channels,
        len,
        data,
    })
}

fn categorical_covariates<'a>(
    array: &'a Option<Array3F64>,
    batch: usize,
    len: usize,
    name: &str,
    supported: bool,
) -> Result<Option<&'a Array3F64>> {
    let Some(array) = array else {
        return Ok(None);
    };
    if !supported {
        return Err(Error(format!(
            "this checkpoint was not trained with covariates; got {name}"
        )));
    }
    if array.shape[0] != batch || array.shape[2] != len {
        return Err(Error(format!(
            "{name} must be ({batch}, C, {len}), got {:?}",
            array.shape
        )));
    }
    if array.data.iter().any(|value| value.is_infinite()) {
        return Err(Error(format!(
            "{name} must hold finite codes or NaN; got an infinite value"
        )));
    }
    if array.shape[1] == 0 {
        return Ok(None);
    }
    Ok(Some(array))
}

struct AssembledRows {
    data: Vec<f32>,
    mask: Vec<bool>,
    roles: Option<Vec<u8>>,
    known_future: Option<Vec<f32>>,
    known_future_mask: Option<Vec<bool>>,
    batch: usize,
    channels: usize,
    t: usize,
    n_targets: usize,
}

pub struct Forecaster {
    pub model: Model,
    pub max_context: usize,
    pub per_shot_horizon_patches: usize,
    pub quantile_real_cap_k: f64,
    pub extreme_value_threshold: f32,
}

impl Forecaster {
    pub fn new(model: Model) -> Self {
        let max_context = model.config.context_length;
        let per_shot_horizon_patches = 32.min(model.config.block_size);
        Self {
            model,
            max_context,
            per_shot_horizon_patches,
            quantile_real_cap_k: 1e4,
            extreme_value_threshold: 1e30,
        }
    }

    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(Model::load(dir)?))
    }

    pub fn forecast(
        &self,
        input: &Input,
        prediction_length: usize,
        options: ForecastOptions,
    ) -> Result<Forecast> {
        if prediction_length < 1 {
            return Err(Error(format!(
                "prediction_length must be >= 1, got {prediction_length}"
            )));
        }
        let [batch, variates, t] = input.context.shape;
        if batch == 0 || variates == 0 || t == 0 {
            return Err(Error("context must be non-empty".into()));
        }
        let mut context = input.context.data.clone();
        let mut mask = vec![false; context.len()];
        if let Some(user_mask) = &input.observation_mask {
            if user_mask.len() != context.len() {
                return Err(Error("observation_mask shape mismatch".into()));
            }
        }
        for i in 0..context.len() {
            let finite = context[i].is_finite();
            let user = input.observation_mask.as_ref().is_none_or(|m| m[i]);
            mask[i] = finite && user;
            if !mask[i] {
                context[i] = 0.0;
            }
        }

        let supports_covariates = self.model.config.covariate_enabled;
        let mut past = numeric_covariates(
            &input.past_covariates,
            batch,
            t,
            "past_covariates",
            supports_covariates,
        )?;
        let mut known = numeric_covariates(
            &input.known_covariates,
            batch,
            t + prediction_length,
            "known_covariates",
            supports_covariates,
        )?;

        let cat_past = categorical_covariates(
            &input.categorical_past_covariates,
            batch,
            t,
            "categorical_past_covariates",
            supports_covariates,
        )?;
        let cat_known = categorical_covariates(
            &input.categorical_known_covariates,
            batch,
            t + prediction_length,
            "categorical_known_covariates",
            supports_covariates,
        )?;

        if cat_past.is_some() || cat_known.is_some() {
            let target = if variates == 1 {
                Some(encoding_target(&context, &mask, batch, t, self.max_context))
            } else {
                None
            };
            let encoded_past = cat_past
                .map(|codes| encode_categorical(codes, target.as_deref(), batch, t))
                .transpose()?;
            let encoded_known = cat_known
                .map(|codes| encode_categorical(codes, target.as_deref(), batch, t))
                .transpose()?;
            past = concat_channels(past, encoded_past)?;
            known = concat_channels(known, encoded_known)?;
        }

        if known.is_some() && !self.model.config.supports_known_covariates() {
            return Err(Error(
                "this checkpoint cannot condition on known-future covariates".into(),
            ));
        }

        let raw = self.forecast_raw(
            &context,
            &mask,
            [batch, variates, t],
            prediction_length,
            past.as_ref(),
            known.as_ref(),
        )?;
        let q = self.model.config.num_quantiles();
        let steps = prediction_length;
        let mut quantiles = raw;
        if options.symmetric_averaging {
            let levels = &self.model.config.quantile_knots;
            let symmetric = levels
                .iter()
                .zip(levels.iter().rev())
                .all(|(&lo, &hi)| (lo + hi - 1.0).abs() < 1e-9);
            if !symmetric {
                return Err(Error(
                    "symmetric averaging needs quantile levels symmetric about 0.5".into(),
                ));
            }
            let negated_context: Vec<f32> = context.iter().map(|&v| -v).collect();
            let negated_past = past.as_ref().map(CovData::negated);
            let negated_known = known.as_ref().map(CovData::negated);
            let mirrored = self.forecast_raw(
                &negated_context,
                &mask,
                [batch, variates, t],
                prediction_length,
                negated_past.as_ref(),
                negated_known.as_ref(),
            )?;
            for i in 0..batch * variates * steps {
                for j in 0..q {
                    let a = quantiles[i * q + j];
                    let b = mirrored[i * q + (q - 1 - j)];
                    quantiles[i * q + j] = 0.5 * (a - b);
                }
            }
        }
        if options.clip_nonnegative {
            for b in 0..batch {
                for v in 0..variates {
                    let base = (b * variates + v) * t;
                    let mut any = false;
                    let mut nonnegative = true;
                    for i in 0..t {
                        if mask[base + i] {
                            any = true;
                            if context[base + i] < 0.0 {
                                nonnegative = false;
                            }
                        }
                    }
                    if any && nonnegative {
                        let row = &mut quantiles
                            [(b * variates + v) * steps * q..(b * variates + v + 1) * steps * q];
                        for value in row.iter_mut() {
                            *value = value.max(0.0);
                        }
                    }
                }
            }
        }
        Ok(Forecast {
            quantiles,
            shape: [batch, variates, steps, q],
            levels: self.model.config.quantile_knots.clone(),
            median_index: self.model.config.median_index(),
        })
    }

    fn forecast_raw(
        &self,
        context: &[f32],
        mask: &[bool],
        shape: [usize; 3],
        prediction_length: usize,
        past: Option<&CovData>,
        known: Option<&CovData>,
    ) -> Result<Vec<f32>> {
        let [batch, variates, _t] = shape;
        let rows = self.assemble_rows(context, mask, shape, prediction_length, past, known)?;
        let all_rows = self.block_ar_decode(&rows, prediction_length)?;
        let q = self.model.config.num_quantiles();
        let n_targets = batch * variates;
        let t_pad = rows.t;
        let steps = prediction_length;
        let mut quantiles = vec![0.0f32; n_targets * steps * q];
        for b in 0..batch {
            for v in 0..variates {
                let src_row = b * rows.channels + v;
                let dst_row = b * variates + v;
                quantiles[dst_row * steps * q..(dst_row + 1) * steps * q]
                    .copy_from_slice(&all_rows[src_row * steps * q..(src_row + 1) * steps * q]);
            }
        }
        let mut target_data = vec![0.0f32; n_targets * t_pad];
        let mut target_mask = vec![false; n_targets * t_pad];
        for b in 0..batch {
            for v in 0..variates {
                let src = (b * rows.channels + v) * t_pad;
                let dst = (b * variates + v) * t_pad;
                target_data[dst..dst + t_pad].copy_from_slice(&rows.data[src..src + t_pad]);
                target_mask[dst..dst + t_pad].copy_from_slice(&rows.mask[src..src + t_pad]);
            }
        }
        self.cap_quantiles(
            &mut quantiles,
            &target_data,
            &target_mask,
            n_targets,
            prediction_length,
            t_pad,
            q,
        );
        if quantiles.iter().any(|value| value.is_nan()) {
            return Err(Error(format!(
                "forecast produced {} NaN values; the input is degenerate",
                quantiles.iter().filter(|value| value.is_nan()).count()
            )));
        }
        Ok(quantiles)
    }

    fn assemble_rows(
        &self,
        context: &[f32],
        mask: &[bool],
        shape: [usize; 3],
        prediction_length: usize,
        past: Option<&CovData>,
        known: Option<&CovData>,
    ) -> Result<AssembledRows> {
        let [batch, variates, t] = shape;
        let ps = self.model.config.patch_size;
        let horizon_padded = prediction_length.div_ceil(ps) * ps;
        let past_channels = past.map_or(0, |c| c.channels);
        let known_channels = known.map_or(0, |c| c.channels);
        let channels = variates + past_channels + known_channels;
        let mut raw = vec![0.0f32; batch * channels * t];
        let mut raw_mask = vec![false; batch * channels * t];
        for b in 0..batch {
            for v in 0..variates {
                let src = (b * variates + v) * t;
                let dst = (b * channels + v) * t;
                let filled = ffill_row(&context[src..src + t], &mask[src..src + t]);
                raw[dst..dst + t].copy_from_slice(&filled);
                raw_mask[dst..dst + t].copy_from_slice(&mask[src..src + t]);
            }
            if let Some(past) = past {
                for c in 0..past_channels {
                    let src = (b * past_channels + c) * past.len;
                    let dst = (b * channels + variates + c) * t;
                    for i in 0..t {
                        let value = past.data[src + i];
                        raw[dst + i] = if value.is_nan() { 0.0 } else { value };
                        raw_mask[dst + i] = !value.is_nan();
                    }
                }
            }
            if let Some(known) = known {
                for c in 0..known_channels {
                    let src = (b * known_channels + c) * known.len;
                    let dst = (b * channels + variates + past_channels + c) * t;
                    for i in 0..t {
                        let value = known.data[src + i];
                        raw[dst + i] = if value.is_nan() { 0.0 } else { value };
                        raw_mask[dst + i] = !value.is_nan();
                    }
                }
            }
        }

        let keep = t.min(self.max_context);
        let mut t_keep = keep;
        let pad = (ps - t_keep % ps) % ps;
        t_keep += pad;
        let mut data = vec![0.0f32; batch * channels * t_keep];
        let mut data_mask = vec![false; batch * channels * t_keep];
        for b in 0..batch {
            for c in 0..channels {
                let src = (b * channels + c) * t + (t - keep);
                let dst = (b * channels + c) * t_keep + pad;
                data[dst..dst + keep].copy_from_slice(&raw[src..src + keep]);
                data_mask[dst..dst + keep].copy_from_slice(&raw_mask[src..src + keep]);
            }
        }

        let mut role_template = Vec::with_capacity(channels);
        role_template.extend(std::iter::repeat_n(ROLE_TARGET, variates));
        role_template.extend(std::iter::repeat_n(ROLE_PAST_COV, past_channels));
        role_template.extend(std::iter::repeat_n(ROLE_KNOWN_COV, known_channels));
        let roles = if past_channels == 0 && known_channels == 0 {
            None
        } else {
            let mut roles = Vec::with_capacity(batch * channels);
            for _ in 0..batch {
                roles.extend_from_slice(&role_template);
            }
            Some(roles)
        };

        let (known_future, known_future_mask) = if known_channels > 0 {
            let known = known.unwrap();
            if known.len != t + prediction_length {
                return Err(Error(
                    "known covariates must cover the context plus the horizon".into(),
                ));
            }
            let mut values = vec![0.0f32; batch * channels * horizon_padded];
            let mut future_mask = vec![false; batch * channels * horizon_padded];
            for b in 0..batch {
                for c in 0..known_channels {
                    let src = (b * known_channels + c) * known.len + t;
                    let dst = (b * channels + variates + past_channels + c) * horizon_padded;
                    for i in 0..prediction_length {
                        let value = known.data[src + i];
                        values[dst + i] = if value.is_nan() { 0.0 } else { value };
                        future_mask[dst + i] = !value.is_nan();
                    }
                }
            }
            (Some(values), Some(future_mask))
        } else {
            (None, None)
        };

        Ok(AssembledRows {
            data,
            mask: data_mask,
            roles,
            known_future,
            known_future_mask,
            batch,
            channels,
            t: t_keep,
            n_targets: variates,
        })
    }

    fn block_ar_decode(&self, rows: &AssembledRows, prediction_length: usize) -> Result<Vec<f32>> {
        let ps = self.model.config.patch_size;
        let q = self.model.config.num_quantiles();
        let total_rows = rows.batch * rows.channels;
        let mut context = rows.data.clone();
        let mut mask = rows.mask.clone();
        let mut group = Vec::with_capacity(total_rows);
        for b in 0..rows.batch {
            for _ in 0..rows.channels {
                group.push(b);
            }
        }
        let target_rows: Vec<bool> = (0..total_rows)
            .map(|row| row % rows.channels < rows.n_targets)
            .collect();
        let known_rows: Vec<bool> = (0..total_rows)
            .map(|row| {
                rows.roles
                    .as_ref()
                    .is_some_and(|roles| roles[row] == ROLE_KNOWN_COV)
            })
            .collect();
        let past_rows: Vec<bool> = (0..total_rows)
            .map(|row| {
                rows.roles
                    .as_ref()
                    .is_some_and(|roles| roles[row] == ROLE_PAST_COV)
            })
            .collect();

        let blocks = prediction_length.div_ceil(ps);
        let total_steps = blocks * ps;
        let mut packed = vec![0.0f32; total_rows * total_steps * q];
        let mut block_offset_steps = 0usize;
        let mut remaining = blocks;
        let mut offset = 0usize;
        let horizon = rows
            .known_future
            .as_ref()
            .map_or(0, |values| values.len() / total_rows);
        while remaining > 0 {
            let block = self.per_shot_horizon_patches.min(remaining);
            let blen = block * ps;
            let known_future_block = rows.known_future.as_ref().map(|values| {
                let mut buffer = vec![0.0f32; total_rows * blen];
                for row in 0..total_rows {
                    let src = row * horizon + offset;
                    buffer[row * blen..(row + 1) * blen].copy_from_slice(&values[src..src + blen]);
                }
                buffer
            });
            let known_future_mask_block = rows.known_future_mask.as_ref().map(|values| {
                let mut buffer = vec![false; total_rows * blen];
                for row in 0..total_rows {
                    let src = row * horizon + offset;
                    buffer[row * blen..(row + 1) * blen].copy_from_slice(&values[src..src + blen]);
                }
                buffer
            });
            let mut quantiles = self.model.predict_block(
                &context,
                &mask,
                &group,
                block,
                rows.roles.as_deref(),
                known_future_block.as_deref(),
                known_future_mask_block.as_deref(),
            )?;
            refine_block(
                &mut quantiles,
                &context,
                &mask,
                total_rows,
                ps,
                self.model.config.median_index(),
                q,
                self.model.config.minimum_scale,
                &target_rows,
                &self.model,
            );
            for row in 0..total_rows {
                let dst = (row * total_steps + block_offset_steps) * q;
                let src = row * blen * q;
                packed[dst..dst + blen * q].copy_from_slice(&quantiles[src..src + blen * q]);
            }
            remaining -= block;
            offset += blen;
            block_offset_steps += blen;
            if remaining == 0 {
                break;
            }

            let mut feedback = vec![0.0f32; total_rows * blen];
            let mut feedback_mask = vec![true; total_rows * blen];
            for row in 0..total_rows {
                for step in 0..blen {
                    let value =
                        quantiles[(row * blen + step) * q + self.model.config.median_index()];
                    let value = if value.is_finite() && value.abs() < self.extreme_value_threshold {
                        value
                    } else {
                        0.0
                    };
                    feedback[row * blen + step] = if target_rows[row] { value } else { 0.0 };
                }
                if known_rows[row] {
                    let values = rows.known_future.as_ref().unwrap();
                    let masks = rows.known_future_mask.as_ref().unwrap();
                    let src = row * horizon + offset - blen;
                    feedback[row * blen..(row + 1) * blen]
                        .copy_from_slice(&values[src..src + blen]);
                    feedback_mask[row * blen..(row + 1) * blen]
                        .copy_from_slice(&masks[src..src + blen]);
                }
                if past_rows[row] {
                    feedback_mask[row * blen..(row + 1) * blen].fill(false);
                }
            }

            let old_t = context.len() / total_rows;
            let new_t = old_t + blen;
            let cap_len = (self.max_context.max(ps) / ps) * ps;
            let keep = if new_t > self.max_context {
                cap_len
            } else {
                new_t
            };
            let mut combined = vec![0.0f32; total_rows * new_t];
            let mut combined_mask = vec![false; total_rows * new_t];
            for row in 0..total_rows {
                let dst = row * new_t;
                combined[dst..dst + old_t]
                    .copy_from_slice(&context[row * old_t..(row + 1) * old_t]);
                combined_mask[dst..dst + old_t]
                    .copy_from_slice(&mask[row * old_t..(row + 1) * old_t]);
                combined[dst + old_t..dst + new_t]
                    .copy_from_slice(&feedback[row * blen..(row + 1) * blen]);
                combined_mask[dst + old_t..dst + new_t]
                    .copy_from_slice(&feedback_mask[row * blen..(row + 1) * blen]);
            }
            if keep == new_t {
                context = combined;
                mask = combined_mask;
            } else {
                let mut capped = vec![0.0f32; total_rows * keep];
                let mut capped_mask = vec![false; total_rows * keep];
                for row in 0..total_rows {
                    let src = row * new_t + (new_t - keep);
                    let dst = row * keep;
                    capped[dst..dst + keep].copy_from_slice(&combined[src..src + keep]);
                    capped_mask[dst..dst + keep].copy_from_slice(&combined_mask[src..src + keep]);
                }
                context = capped;
                mask = capped_mask;
            }
        }

        let steps = prediction_length;
        let mut out = vec![0.0f32; total_rows * steps * q];
        for row in 0..total_rows {
            out[row * steps * q..(row + 1) * steps * q]
                .copy_from_slice(&packed[row * total_steps * q..row * total_steps * q + steps * q]);
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn cap_quantiles(
        &self,
        quantiles: &mut [f32],
        inputs: &[f32],
        mask: &[bool],
        rows: usize,
        prediction_length: usize,
        t: usize,
        q: usize,
    ) {
        if self.quantile_real_cap_k <= 0.0 {
            return;
        }
        let (_, scale) = self.model.scale(inputs, mask, rows, t);
        for row in 0..rows {
            let base = row * t;
            let mut max_value = f32::NEG_INFINITY;
            let mut min_value = f32::INFINITY;
            let mut any = false;
            for i in 0..t {
                if mask[base + i] {
                    any = true;
                    max_value = max_value.max(inputs[base + i]);
                    min_value = min_value.min(inputs[base + i]);
                }
            }
            if !any {
                max_value = 0.0;
                min_value = 0.0;
            }
            let cap = scale[base + t - 1] * self.quantile_real_cap_k as f32;
            let low = min_value - cap;
            let high = max_value + cap;
            for step in 0..prediction_length {
                for i in 0..q {
                    let idx = (row * prediction_length + step) * q + i;
                    quantiles[idx] = quantiles[idx].clamp(low, high);
                }
            }
        }
    }
}
