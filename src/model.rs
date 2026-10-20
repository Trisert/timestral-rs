use std::collections::HashMap;
use std::path::Path;

use crate::attention::{
    apply_rope, build_rope_tables, time_attention, variate_attention, TimeMask, TimeMaskKind,
    VariateMask,
};
use crate::config::{Config, Gate, LayerKind};
use crate::error::{Error, Result};
use crate::ops::{
    advancing_block_frames, causal_patch_loc_scale, clamp_nonfinite, instance_norm_stats,
    monotone_quantiles, rms_norm, scale_rows, Affine, Mat,
};
use crate::safetensors::{SafeTensors, Tensor};

pub const ROLE_TARGET: u8 = 0;
pub const ROLE_PAST_COV: u8 = 1;
pub const ROLE_KNOWN_COV: u8 = 2;

#[derive(Debug)]
struct AttentionWeights {
    wqkv: Affine,
    gate: Option<Affine>,
    wo: Affine,
}

#[derive(Debug)]
struct LayerWeights {
    attn_norms: Vec<Vec<f32>>,
    attns: Vec<AttentionWeights>,
    mlp_norm: Vec<f32>,
    w12: Affine,
    w3: Affine,
}

#[derive(Debug)]
struct ModelWeights {
    patch_hidden: Affine,
    patch_out: Affine,
    patch_res: Affine,
    forecast_query: Vec<f32>,
    layers: Vec<LayerWeights>,
    out_norm: Vec<f32>,
    head_hidden: Affine,
    head_out: Affine,
    head_res: Affine,
}

#[derive(Debug)]
pub struct Model {
    pub config: Config,
    kinds: Vec<LayerKind>,
    w: ModelWeights,
}

fn tensor_of<'a>(st: &'a SafeTensors, name: &str, shape: &[usize]) -> Result<&'a Tensor> {
    let tensor = st.get(name)?;
    tensor.expect_shape(name, shape)?;
    Ok(tensor)
}

fn mat_of(st: &SafeTensors, name: &str, shape: &[usize]) -> Result<Mat> {
    let tensor = tensor_of(st, name, shape)?;
    Mat::from_row_major(shape[0], shape[1], &tensor.data)
}

fn affine_of(st: &SafeTensors, prefix: &str, out: usize, inp: usize, bias: bool) -> Result<Affine> {
    let w = mat_of(st, &format!("{prefix}.weight"), &[out, inp])?;
    let b = if bias {
        Some(
            tensor_of(st, &format!("{prefix}.bias"), &[out])?
                .data
                .clone(),
        )
    } else {
        None
    };
    Ok(Affine { w, b })
}

fn vector_of(st: &SafeTensors, name: &str, len: usize) -> Result<Vec<f32>> {
    Ok(tensor_of(st, name, &[len])?.data.clone())
}

impl Model {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::load(&dir.join("config.json"))?;
        let weights = SafeTensors::load(&dir.join("model.safetensors"))?;
        Self::from_parts(config, &weights)
    }

    pub fn from_parts(config: Config, st: &SafeTensors) -> Result<Self> {
        config.validate()?;
        let kinds = config.layer_kinds()?;
        let time_gate = config.time_gate()?;
        let variate_gate = config.variate_gate()?;
        let d = config.d_model;
        let ps = config.patch_size;
        let hidden = 4 * d;
        let q = config.num_quantiles();
        let patch_hidden = affine_of(
            st,
            "representation.patch_proj.hidden_layer",
            hidden,
            2 * ps,
            true,
        )?;
        let patch_out = affine_of(
            st,
            "representation.patch_proj.output_layer",
            d,
            hidden,
            true,
        )?;
        let patch_res = affine_of(
            st,
            "representation.patch_proj.residual_layer",
            d,
            2 * ps,
            true,
        )?;
        let forecast_query = vector_of(st, "representation.forecast_query", d)?;
        let out_norm = vector_of(st, "transformer.out_norm.scale", d)?;
        let head_hidden = affine_of(st, "head.projection.hidden_layer", hidden, d, false)?;
        let head_out = affine_of(st, "head.projection.output_layer", ps * q, hidden, false)?;
        let head_res = affine_of(st, "head.projection.residual_layer", ps * q, d, false)?;

        let mut layers = Vec::with_capacity(kinds.len());
        for (index, kind) in kinds.iter().enumerate() {
            let mut attn_norms = Vec::new();
            let mut attns = Vec::new();
            for (axis, &is_time) in kind.axes().iter().enumerate() {
                let norm = vector_of(
                    st,
                    &format!("transformer.layers.{index}.attn_norms.{axis}.scale"),
                    d,
                )?;
                attn_norms.push(norm);
                let prefix = format!("transformer.layers.{index}.attns.{axis}");
                let wqkv = affine_of(st, &format!("{prefix}.wQKV"), 3 * d, d, config.attn_bias)?;
                let wo = affine_of(st, &format!("{prefix}.wO"), d, d, config.attn_bias)?;
                let gate_kind = if is_time { time_gate } else { variate_gate };
                let gate = match gate_kind {
                    Gate::None => None,
                    Gate::SdpaSigmoid => Some(affine_of(
                        st,
                        &format!("{prefix}._gate"),
                        d,
                        d,
                        config.attn_bias,
                    )?),
                };
                attns.push(AttentionWeights { wqkv, gate, wo });
            }
            let mlp_norm = vector_of(st, &format!("transformer.layers.{index}.mlp_norm.scale"), d)?;
            let w12 = affine_of(
                st,
                &format!("transformer.layers.{index}.mlp.w12"),
                2 * config.mlp_hidden_dim,
                d,
                config.mlp_bias,
            )?;
            let w3 = affine_of(
                st,
                &format!("transformer.layers.{index}.mlp.w3"),
                d,
                config.mlp_hidden_dim,
                config.mlp_bias,
            )?;
            layers.push(LayerWeights {
                attn_norms,
                attns,
                mlp_norm,
                w12,
                w3,
            });
        }

        Ok(Self {
            config,
            kinds,
            w: ModelWeights {
                patch_hidden,
                patch_out,
                patch_res,
                forecast_query,
                layers,
                out_norm,
                head_hidden,
                head_out,
                head_res,
            },
        })
    }

    fn patch_embed(&self, scaled: &[f32], observed: &[bool], rows: usize, n: usize) -> Vec<f32> {
        let ps = self.config.patch_size;
        let d = self.config.d_model;
        let tokens = rows * n;
        let feat = 2 * ps;
        let t = n * ps;
        let mut features = vec![0.0f32; tokens * feat];
        for row in 0..rows {
            for patch in 0..n {
                let token = row * n + patch;
                let src = row * t + patch * ps;
                let dst = token * feat;
                features[dst..dst + ps].copy_from_slice(&scaled[src..src + ps]);
                for i in 0..ps {
                    features[dst + ps + i] = if observed[src + i] { 0.0 } else { 1.0 };
                }
            }
        }
        let hidden_dim = self.w.patch_hidden.w.rows;
        let mut hidden = vec![0.0f32; tokens * hidden_dim];
        self.w.patch_hidden.linear(&features, tokens, &mut hidden);
        for value in hidden.iter_mut() {
            *value = crate::ops::silu(*value);
        }
        let mut out = vec![0.0f32; tokens * d];
        let mut res = vec![0.0f32; tokens * d];
        self.w.patch_out.linear(&hidden, tokens, &mut out);
        self.w.patch_res.linear(&features, tokens, &mut res);
        for (o, r) in out.iter_mut().zip(res) {
            *o += r;
        }
        out
    }

    fn forecast_tokens(&self, rows: usize, n: usize) -> Vec<f32> {
        let d = self.config.d_model;
        let mut out = Vec::with_capacity(rows * n * d);
        for _ in 0..rows * n {
            out.extend_from_slice(&self.w.forecast_query);
        }
        out
    }

    fn transformer(
        &self,
        x: &mut [f32],
        rows: usize,
        total: usize,
        positions: &[i64],
        time_mask: &TimeMask,
        variate_mask: &VariateMask,
    ) {
        let c = &self.config;
        let d = c.d_model;
        let heads = c.num_heads;
        let hd = c.head_dim();
        let tokens = rows * total;
        let eps = c.norm_eps as f32;
        let (cos, sin) = build_rope_tables(positions, hd, c.rope_dim(), c.rope_theta);
        let mut xn = vec![0.0f32; tokens * d];
        let mut qkv = vec![0.0f32; tokens * 3 * d];
        let mut q = vec![0.0f32; tokens * d];
        let mut k = vec![0.0f32; tokens * d];
        let mut v = vec![0.0f32; tokens * d];
        let mut attn_out = vec![0.0f32; tokens * d];
        let mut delta = vec![0.0f32; tokens * d];

        for (layer, kind) in self.w.layers.iter().zip(&self.kinds) {
            for (axis, &is_time) in kind.axes().iter().enumerate() {
                let attn = &layer.attns[axis];
                rms_norm(x, &layer.attn_norms[axis], eps, &mut xn);
                attn.wqkv.linear(&xn, tokens, &mut qkv);
                for token in 0..tokens {
                    for head in 0..heads {
                        let dst = (token * heads + head) * hd;
                        for i in 0..hd {
                            let fused = |channel: usize| {
                                qkv[token * 3 * d + channel * d + i * heads + head]
                            };
                            q[dst + i] = fused(0);
                            k[dst + i] = fused(1);
                            v[dst + i] = fused(2);
                        }
                    }
                }
                if is_time {
                    apply_rope(&mut q, &cos, &sin, tokens, heads, hd);
                    apply_rope(&mut k, &cos, &sin, tokens, heads, hd);
                    time_attention(&q, &k, &v, rows, total, heads, hd, time_mask, &mut attn_out);
                } else {
                    variate_attention(
                        &q,
                        &k,
                        &v,
                        rows,
                        total,
                        heads,
                        hd,
                        variate_mask,
                        &mut attn_out,
                    );
                }
                if let Some(gate) = &attn.gate {
                    let mut gate_values = vec![0.0f32; tokens * d];
                    gate.linear(&xn, tokens, &mut gate_values);
                    for (out, g) in attn_out.iter_mut().zip(gate_values) {
                        *out *= crate::ops::sigmoid(g);
                    }
                }
                attn.wo.linear(&attn_out, tokens, &mut delta);
                for (x, d) in x.iter_mut().zip(&delta) {
                    *x += *d;
                }
            }

            rms_norm(x, &layer.mlp_norm, eps, &mut xn);
            let hidden_dim = self.config.mlp_hidden_dim;
            let mut w12 = vec![0.0f32; tokens * 2 * hidden_dim];
            layer.w12.linear(&xn, tokens, &mut w12);
            let mut act = vec![0.0f32; tokens * hidden_dim];
            for token in 0..tokens {
                for i in 0..hidden_dim {
                    let gate = w12[token * 2 * hidden_dim + i];
                    let up = w12[token * 2 * hidden_dim + hidden_dim + i];
                    act[token * hidden_dim + i] = gate * crate::ops::silu(up);
                }
            }
            layer.w3.linear(&act, tokens, &mut delta);
            for (x, d) in x.iter_mut().zip(&delta) {
                *x += *d;
            }
        }
    }

    fn out_norm_in_place(&self, x: &mut [f32]) {
        let copy = x.to_vec();
        rms_norm(&copy, &self.w.out_norm, self.config.norm_eps as f32, x);
    }

    fn head(&self, dec: &[f32], rows: usize, patches: usize) -> Vec<f32> {
        let c = &self.config;
        let d = c.d_model;
        let hidden_dim = 4 * d;
        let q = c.num_quantiles();
        let tokens = rows * patches;
        let mut hidden = vec![0.0f32; tokens * hidden_dim];
        self.w.head_hidden.linear(dec, tokens, &mut hidden);
        for value in hidden.iter_mut() {
            *value = crate::ops::silu(*value);
        }
        let out_dim = c.patch_size * q;
        let mut out = vec![0.0f32; tokens * out_dim];
        let mut res = vec![0.0f32; tokens * out_dim];
        self.w.head_out.linear(&hidden, tokens, &mut out);
        self.w.head_res.linear(dec, tokens, &mut res);
        for (o, r) in out.iter_mut().zip(res) {
            *o += r;
        }
        let mut quantiles = vec![0.0f32; out.len()];
        for token in 0..tokens {
            for step in 0..c.patch_size {
                let raw =
                    &out[(token * c.patch_size + step) * q..(token * c.patch_size + step + 1) * q];
                let dst = &mut quantiles
                    [(token * c.patch_size + step) * q..(token * c.patch_size + step + 1) * q];
                monotone_quantiles(raw, &c.quantile_knots, c.quantile_width_factor, dst);
            }
        }
        quantiles
    }

    fn scaled(
        &self,
        inputs: &[f32],
        observed: &[bool],
        rows: usize,
        t: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = &self.config;
        let (loc, scale) = causal_patch_loc_scale(
            inputs,
            observed,
            rows,
            t,
            c.patch_size,
            c.minimum_scale,
            c.stabilize_with_global,
        );
        let mut scaled = vec![0.0f32; inputs.len()];
        scale_rows(inputs, &loc, &scale, observed, c.use_asinh, &mut scaled);
        (scaled, loc, scale)
    }

    pub fn scale(
        &self,
        inputs: &[f32],
        observed: &[bool],
        rows: usize,
        t: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        causal_patch_loc_scale(
            inputs,
            observed,
            rows,
            t,
            self.config.patch_size,
            self.config.minimum_scale,
            self.config.stabilize_with_global,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn known_cov_scaling(
        &self,
        inputs: &[f32],
        observed: &[bool],
        scaled: &mut [f32],
        loc: &mut [f32],
        scale: &mut [f32],
        rows: usize,
        t: usize,
        known_rows: &[bool],
    ) {
        if !known_rows.iter().any(|&x| x) {
            return;
        }
        let (loc_c, scale_c, is_binary) =
            instance_norm_stats(inputs, observed, rows, t, self.config.minimum_scale as f32);
        for row in 0..rows {
            if !known_rows[row] {
                continue;
            }
            let base = row * t;
            for i in 0..t {
                let z = (inputs[base + i] - loc_c[row]) / scale_c[row].max(f32::EPSILON);
                let mut value = if observed[base + i] { z } else { 0.0 };
                if self.config.use_asinh && observed[base + i] && !is_binary[row] {
                    value = value.asinh();
                }
                scaled[base + i] = value;
                loc[base + i] = loc_c[row];
                scale[base + i] = scale_c[row];
            }
        }
    }

    fn obs_per_patch(&self, observed: &[bool], rows: usize, n: usize) -> Vec<bool> {
        let ps = self.config.patch_size;
        let mut out = vec![false; rows * n];
        for row in 0..rows {
            for patch in 0..n {
                let base = row * n * ps + patch * ps;
                out[row * n + patch] = observed[base..base + ps].iter().any(|&x| x);
            }
        }
        out
    }

    fn key_valid(&self, obs: &[bool], rows: usize, n: usize) -> Vec<bool> {
        if !self.config.attend_missing_patches {
            return obs.to_vec();
        }
        let mut out = obs.to_vec();
        for row in 0..rows {
            let mut started = false;
            for patch in 0..n {
                let observed = obs[row * n + patch];
                started |= observed;
                out[row * n + patch] = observed || started;
            }
        }
        out
    }

    fn first_observed(&self, obs: &[bool], rows: usize, n: usize, group: &[usize]) -> Vec<usize> {
        let mut first: HashMap<usize, usize> = HashMap::new();
        for row in 0..rows {
            let g = group[row];
            for patch in 0..n {
                if obs[row * n + patch] {
                    first.entry(g).or_insert(patch);
                }
            }
        }
        (0..rows)
            .map(|row| *first.get(&group[row]).unwrap_or(&0))
            .collect()
    }

    fn denormalize(
        &self,
        quantiles: &mut [f32],
        rows: usize,
        steps: usize,
        q: usize,
        loc0: &[f32],
        scale0: &[f32],
    ) {
        for row in 0..rows {
            let loc = loc0[row];
            let scale = scale0[row];
            for step in 0..steps {
                for i in 0..q {
                    let idx = (row * steps + step) * q + i;
                    let z = quantiles[idx];
                    quantiles[idx] = if self.config.use_asinh {
                        z.sinh() * scale + loc
                    } else {
                        z * scale + loc
                    };
                }
            }
        }
        clamp_nonfinite(quantiles, q);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn predict_block(
        &self,
        inputs: &[f32],
        observed: &[bool],
        group_ids: &[usize],
        horizon_patches: usize,
        role_ids: Option<&[u8]>,
        known_future: Option<&[f32]>,
        known_future_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return Err(Error("empty context".into()));
        }
        let rows = group_ids.len();
        if !inputs.len().is_multiple_of(rows) {
            return Err(Error("inputs length is not a multiple of rows".into()));
        }
        let t = inputs.len() / rows;
        if !t.is_multiple_of(self.config.patch_size) {
            return Err(Error(format!(
                "context length {t} must be a multiple of patch_size {}",
                self.config.patch_size
            )));
        }
        if observed.len() != inputs.len() {
            return Err(Error("observation mask shape mismatch".into()));
        }
        if horizon_patches < 1 {
            return Err(Error("horizon_patches must be >= 1".into()));
        }
        let context_only = role_ids.is_none()
            || !self.config.covariate_enabled
            || role_ids.unwrap().iter().all(|&r| r == ROLE_TARGET);
        if context_only {
            self.predict_context_only(inputs, observed, group_ids, horizon_patches)
        } else {
            self.predict_window(
                inputs,
                observed,
                group_ids,
                horizon_patches,
                role_ids.unwrap(),
                known_future,
                known_future_mask,
            )
        }
    }

    fn predict_context_only(
        &self,
        inputs: &[f32],
        observed: &[bool],
        group_ids: &[usize],
        horizon: usize,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let ps = c.patch_size;
        let d = c.d_model;
        let rows = group_ids.len();
        let t = inputs.len() / rows;
        let ll = t / ps;
        let (scaled, loc, scale) = self.scaled(inputs, observed, rows, t);
        let x_enc = self.patch_embed(&scaled, observed, rows, ll);
        let x_dec = self.forecast_tokens(rows, horizon);
        let total = ll + horizon;
        let mut x = vec![0.0f32; rows * total * d];
        for row in 0..rows {
            let enc_src = row * ll * d;
            let dst = row * total * d;
            x[dst..dst + ll * d].copy_from_slice(&x_enc[enc_src..enc_src + ll * d]);
            x[dst + ll * d..dst + total * d]
                .copy_from_slice(&x_dec[row * horizon * d..(row + 1) * horizon * d]);
        }
        let obs = self.obs_per_patch(observed, rows, ll);
        let key_valid = self.key_valid(&obs, rows, ll);
        let first = self.first_observed(&obs, rows, ll, group_ids);
        let mut positions = vec![0i64; rows * total];
        for row in 0..rows {
            for i in 0..total {
                positions[row * total + i] = i as i64 - first[row] as i64;
            }
        }
        let time_mask = TimeMask {
            rows,
            total,
            enc_len: ll,
            key_valid: key_valid.clone(),
            kind: TimeMaskKind::Forecast,
        };
        let variate_mask = VariateMask {
            rows,
            slots: total,
            enc_len: ll,
            key_valid,
            group: group_ids.to_vec(),
            forbid: None,
        };
        self.transformer(&mut x, rows, total, &positions, &time_mask, &variate_mask);
        self.out_norm_in_place(&mut x);

        let mut dec = vec![0.0f32; rows * horizon * d];
        for row in 0..rows {
            let src = (row * total + ll) * d;
            dec[row * horizon * d..(row + 1) * horizon * d]
                .copy_from_slice(&x[src..src + horizon * d]);
        }
        let q = c.num_quantiles();
        let steps = horizon * ps;
        let mut quantiles = self.head(&dec, rows, horizon);
        let mut loc0 = vec![0.0f32; rows];
        let mut scale0 = vec![0.0f32; rows];
        for row in 0..rows {
            loc0[row] = loc[row * t + t - 1];
            scale0[row] = scale[row * t + t - 1];
        }
        self.denormalize(&mut quantiles, rows, steps, q, &loc0, &scale0);
        Ok(quantiles)
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_window(
        &self,
        inputs: &[f32],
        observed: &[bool],
        group_ids: &[usize],
        horizon: usize,
        role_ids: &[u8],
        known_future: Option<&[f32]>,
        known_future_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let ps = c.patch_size;
        let d = c.d_model;
        let rows = group_ids.len();
        let t = inputs.len() / rows;
        let horizon_len = horizon * ps;
        let zeros;
        let falses;
        let full_future: &[f32] = match known_future {
            Some(values) => {
                if values.len() != rows * horizon_len {
                    return Err(Error(format!(
                        "known_future must have {} values, got {}",
                        rows * horizon_len,
                        values.len()
                    )));
                }
                values
            }
            None => {
                zeros = vec![0.0f32; rows * horizon_len];
                &zeros
            }
        };
        let future_mask: &[bool] = match known_future_mask {
            Some(values) => {
                if values.len() != rows * horizon_len {
                    return Err(Error("known_future_mask shape mismatch".into()));
                }
                values
            }
            None => {
                falses = vec![false; rows * horizon_len];
                &falses
            }
        };
        if role_ids.len() != rows {
            return Err(Error("role_ids must have one entry per row".into()));
        }
        if horizon > c.block_size {
            return Err(Error(format!(
                "decode block {horizon} must be in [1, {}] with covariates",
                c.block_size
            )));
        }
        let window = t + horizon_len;
        let n = window / ps;
        if n <= horizon {
            return Err(Error("context must have at least one patch".into()));
        }
        let ll = n - horizon;
        let mut full_inputs = Vec::with_capacity(rows * window);
        let mut full_observed = Vec::with_capacity(rows * window);
        for row in 0..rows {
            full_inputs.extend_from_slice(&inputs[row * t..(row + 1) * t]);
            full_inputs.extend_from_slice(&full_future[row * horizon_len..(row + 1) * horizon_len]);
            full_observed.extend_from_slice(&observed[row * t..(row + 1) * t]);
            full_observed
                .extend_from_slice(&future_mask[row * horizon_len..(row + 1) * horizon_len]);
        }

        let (mut scaled, mut loc, mut scale) =
            self.scaled(&full_inputs, &full_observed, rows, window);
        let known_rows: Vec<bool> = role_ids.iter().map(|&r| r == ROLE_KNOWN_COV).collect();
        self.known_cov_scaling(
            &full_inputs,
            &full_observed,
            &mut scaled,
            &mut loc,
            &mut scale,
            rows,
            window,
            &known_rows,
        );

        let x_enc = self.patch_embed(&scaled, &full_observed, rows, n);
        let x_dec = self.forecast_tokens(rows, n);
        let total = 2 * n;
        let mut x = vec![0.0f32; rows * total * d];
        for row in 0..rows {
            let src = row * n * d;
            let dst = row * total * d;
            x[dst..dst + n * d].copy_from_slice(&x_enc[src..src + n * d]);
            x[dst + n * d..dst + total * d].copy_from_slice(&x_dec[src..src + n * d]);
        }
        let obs = self.obs_per_patch(&full_observed, rows, n);
        let key_valid = self.key_valid(&obs, rows, n);
        let first = self.first_observed(&obs, rows, n, group_ids);
        let mut positions = vec![0i64; rows * total];
        for row in 0..rows {
            for i in 0..n {
                let p = i as i64 - first[row] as i64;
                positions[row * total + i] = p;
                positions[row * total + n + i] = p;
            }
        }
        let time_mask = TimeMask {
            rows,
            total,
            enc_len: n,
            key_valid: key_valid.clone(),
            kind: TimeMaskKind::TwoStream {
                ll,
                block_size: c.block_size,
                n,
                bidirectional: known_rows.clone(),
            },
        };
        let target_rows: Vec<bool> = role_ids.iter().map(|&r| r == ROLE_TARGET).collect();
        let mut forbid = vec![false; rows * rows];
        for query in 0..rows {
            for key in 0..rows {
                let covariate_to_target = !target_rows[query] && target_rows[key];
                let known_to_causal = known_rows[query] && !known_rows[key];
                forbid[query * rows + key] = covariate_to_target || known_to_causal;
            }
        }
        let variate_mask = VariateMask {
            rows,
            slots: total,
            enc_len: n,
            key_valid,
            group: group_ids.to_vec(),
            forbid: Some(forbid),
        };
        self.transformer(&mut x, rows, total, &positions, &time_mask, &variate_mask);
        self.out_norm_in_place(&mut x);

        let mut dec = vec![0.0f32; rows * horizon * d];
        for row in 0..rows {
            for patch in 0..horizon {
                let src = (row * total + n + ll + patch) * d;
                let dst = (row * horizon + patch) * d;
                dec[dst..dst + d].copy_from_slice(&x[src..src + d]);
            }
        }
        let q = c.num_quantiles();
        let steps = horizon * ps;
        let mut quantiles = self.head(&dec, rows, horizon);
        let mut loc0 = vec![0.0f32; rows];
        let mut scale0 = vec![0.0f32; rows];
        for row in 0..rows {
            loc0[row] = loc[row * window + ll * ps - 1];
            scale0[row] = scale[row * window + ll * ps - 1];
        }
        self.denormalize(&mut quantiles, rows, steps, q, &loc0, &scale0);
        Ok(quantiles)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn refine_block(
    quantiles: &mut [f32],
    context: &[f32],
    mask: &[bool],
    rows: usize,
    patch_size: usize,
    median_index: usize,
    num_quantiles: usize,
    minimum_scale: f64,
    target_rows: &[bool],
    model: &Model,
) {
    let steps = quantiles.len() / (rows * num_quantiles);
    let n_pred = steps / patch_size;
    if n_pred <= 1 {
        return;
    }
    let t = context.len() / rows;
    let (loc, scale) = model.scale(context, mask, rows, t);
    let mut loc0 = vec![0.0f32; rows];
    let mut scale0 = vec![0.0f32; rows];
    for row in 0..rows {
        loc0[row] = loc[row * t + t - 1];
        scale0[row] = scale[row * t + t - 1];
    }
    let mut sinh_median = vec![0.0f64; rows * n_pred * patch_size];
    for row in 0..rows {
        for time in 0..steps {
            let value = quantiles[(row * steps + time) * num_quantiles + median_index] as f64;
            let patch = time / patch_size;
            let step = time % patch_size;
            sinh_median[(row * n_pred + patch) * patch_size + step] =
                (value - loc0[row] as f64) / scale0[row] as f64;
        }
    }
    let (locs, scales) = advancing_block_frames(
        context,
        mask,
        &sinh_median,
        rows,
        n_pred,
        patch_size,
        minimum_scale,
    );
    let mut loc_j = vec![0.0f32; rows * n_pred];
    let mut scale_j = vec![0.0f32; rows * n_pred];
    for row in 0..rows {
        for patch in 0..n_pred {
            if patch == 0 {
                loc_j[row * n_pred] = loc0[row];
                scale_j[row * n_pred] = scale0[row];
            } else {
                loc_j[row * n_pred + patch] = locs[row * n_pred + patch] as f32;
                scale_j[row * n_pred + patch] = scales[row * n_pred + patch] as f32;
            }
        }
    }
    for row in 0..rows {
        if !target_rows[row] {
            continue;
        }
        for time in 0..steps {
            let patch = time / patch_size;
            let l = loc_j[row * n_pred + patch];
            let s = scale_j[row * n_pred + patch];
            for q in 0..num_quantiles {
                let idx = (row * steps + time) * num_quantiles + q;
                let z = (quantiles[idx] - loc0[row]) / scale0[row];
                quantiles[idx] = z * s + l;
            }
        }
    }
    clamp_nonfinite(quantiles, num_quantiles);
}
