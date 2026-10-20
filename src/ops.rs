use matrixmultiply::sgemm;
use rayon::prelude::*;

use crate::error::{Error, Result};

struct SendPtr(*mut f32);

unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct Mat {
    /// Output features (`rows`) and input features (`cols`).
    pub rows: usize,
    pub cols: usize,
    /// Transposed row-major layout: `data[col * rows + row]`.
    pub data: Vec<f32>,
}

impl Mat {
    pub fn from_row_major(rows: usize, cols: usize, data: &[f32]) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error(format!(
                "matrix shape {rows}x{cols} does not match {} values",
                data.len()
            )));
        }
        let mut transposed = vec![0.0f32; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        Ok(Self {
            rows,
            cols,
            data: transposed,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Affine {
    pub w: Mat,
    pub b: Option<Vec<f32>>,
}

impl Affine {
    pub fn linear(&self, x: &[f32], tokens: usize, out: &mut [f32]) {
        self.w.linear(x, tokens, self.b.as_deref(), out);
    }
}

impl Mat {
    pub fn linear(&self, x: &[f32], tokens: usize, bias: Option<&[f32]>, out: &mut [f32]) {
        assert_eq!(x.len(), tokens * self.cols);
        assert_eq!(out.len(), tokens * self.rows);
        let m = tokens;
        let k = self.cols;
        let n = self.rows;
        let total = m.saturating_mul(n).saturating_mul(k);
        let parallel = total >= (1 << 24) && m > 1;
        if parallel && n >= 512 {
            let threads = rayon::current_num_threads().max(1);
            let chunk = n.div_ceil(threads).max(1);
            let out_ptr = SendPtr(out.as_mut_ptr());
            (0..threads).into_par_iter().for_each(move |index| {
                let col0 = index * chunk;
                if col0 >= n {
                    return;
                }
                let cols = (n - col0).min(chunk);
                unsafe {
                    sgemm(
                        m,
                        k,
                        cols,
                        1.0,
                        x.as_ptr(),
                        k as isize,
                        1,
                        self.data.as_ptr().add(col0),
                        n as isize,
                        1,
                        0.0,
                        out_ptr.get().add(col0),
                        n as isize,
                        1,
                    );
                }
            });
        } else if parallel {
            let threads = rayon::current_num_threads().max(1);
            let chunk = m.div_ceil(threads).max(1);
            out.par_chunks_mut(chunk * n)
                .enumerate()
                .for_each(|(index, chunk_out)| {
                    let row0 = index * chunk;
                    let rows = chunk_out.len() / n;
                    debug_assert!(rows > 0);
                    unsafe {
                        sgemm(
                            rows,
                            k,
                            n,
                            1.0,
                            x.as_ptr().add(row0 * k),
                            k as isize,
                            1,
                            self.data.as_ptr(),
                            n as isize,
                            1,
                            0.0,
                            chunk_out.as_mut_ptr(),
                            n as isize,
                            1,
                        );
                    }
                });
        } else {
            unsafe {
                sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    x.as_ptr(),
                    k as isize,
                    1,
                    self.data.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    out.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        }
        if let Some(bias) = bias {
            for row in out.chunks_exact_mut(n) {
                for (value, b) in row.iter_mut().zip(bias) {
                    *value += *b;
                }
            }
        }
    }
}

pub fn rms_norm(x: &[f32], scale: &[f32], eps: f32, out: &mut [f32]) {
    let d = scale.len();
    debug_assert_eq!(x.len(), out.len());
    debug_assert_eq!(x.len() % d, 0);
    for (src, dst) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mut sum = 0.0f32;
        for v in src {
            sum += v * v;
        }
        let r = 1.0 / (sum / d as f32 + eps).sqrt();
        for i in 0..d {
            dst[i] = src[i] * r * scale[i];
        }
    }
}

#[inline]
pub fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).floor();
    let r = x - n * std::f32::consts::LN_2;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    let exponent = (n as i32 + 127) as u32;
    p * f32::from_bits(exponent << 23)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp_fast(-x))
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let base = i * 8;
        for j in 0..8 {
            acc[j] += a[base + j] * b[base + j];
        }
    }
    let mut sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for i in chunks * 8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Causal per-patch mean/std, broadcast back to every step of the patch.
pub fn causal_patch_loc_scale(
    data: &[f32],
    mask: &[bool],
    rows: usize,
    t: usize,
    patch_size: usize,
    minimum_scale: f64,
    stabilize_with_global: bool,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(data.len(), rows * t);
    assert_eq!(mask.len(), rows * t);
    assert_eq!(
        t % patch_size,
        0,
        "length {t} is not a multiple of patch_size {patch_size}"
    );
    let mut loc = vec![0.0f32; rows * t];
    let mut scale = vec![0.0f32; rows * t];
    let patches = t / patch_size;
    for row in 0..rows {
        let base = row * t;
        let (global_mean, global_scale) = if stabilize_with_global {
            let mut w_sum = 0.0f64;
            let mut x_sum = 0.0f64;
            for i in 0..t {
                if mask[base + i] {
                    w_sum += 1.0;
                    x_sum += data[base + i] as f64;
                }
            }
            let mean = if w_sum > 0.0 { x_sum / w_sum } else { 0.0 };
            let mut m2 = 0.0f64;
            for i in 0..t {
                if mask[base + i] {
                    let d = data[base + i] as f64 - mean;
                    m2 += d * d;
                }
            }
            let den = w_sum.max(1.0);
            let var = m2 / den;
            (mean, (var + minimum_scale).sqrt())
        } else {
            (0.0, 0.0)
        };
        let mut cum_w = 0.0f64;
        let mut cum_x = 0.0f64;
        let mut m2 = 0.0f64;
        let mut prev_mean = 0.0f64;
        for i in 0..t {
            let x = data[base + i] as f64;
            let w = if mask[base + i] { 1.0 } else { 0.0 };
            cum_w += w;
            cum_x += w * x;
            let denom = cum_w.max(1.0);
            let mean = cum_x / denom;
            let shifted = if i == 0 { 0.0 } else { prev_mean };
            m2 += (x - shifted) * (x - mean) * w;
            let var = (m2 / (denom - 1.0).max(1.0)).max(0.0);
            let mut sc = var.sqrt().max(minimum_scale);
            if stabilize_with_global {
                let lo = 10.0f64.powf(-10.0);
                let hi = 10.0f64.powf(10.0);
                let floor = minimum_scale.max(global_scale * lo);
                sc = sc.clamp(floor, global_scale * hi);
            }
            prev_mean = mean;
            if i % patch_size == patch_size - 1 {
                let patch = i / patch_size;
                for j in 0..patch_size {
                    loc[base + patch * patch_size + j] = mean as f32;
                    scale[base + patch * patch_size + j] = sc as f32;
                }
            }
        }
        debug_assert_eq!(patches * patch_size, t);
        let _ = global_mean;
    }
    (loc, scale)
}

pub fn scale_rows(
    data: &[f32],
    loc: &[f32],
    scale: &[f32],
    mask: &[bool],
    use_asinh: bool,
    out: &mut [f32],
) {
    let eps = f32::EPSILON;
    for i in 0..data.len() {
        if mask[i] {
            let z = (data[i] - loc[i]) / scale[i].max(eps);
            out[i] = if use_asinh { z.asinh() } else { z };
        } else {
            out[i] = 0.0;
        }
    }
}

pub fn instance_norm_stats(
    values: &[f32],
    mask: &[bool],
    rows: usize,
    len: usize,
    minimum_scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<bool>) {
    let mut loc = vec![0.0f32; rows];
    let mut scale = vec![0.0f32; rows];
    let mut is_binary = vec![false; rows];
    for row in 0..rows {
        let base = row * len;
        let mut n = 0.0f32;
        let mut sum = 0.0f32;
        let mut any = false;
        let mut zero_one = true;
        for i in 0..len {
            if mask[base + i] {
                n += 1.0;
                sum += values[base + i];
                any = true;
                let v = values[base + i];
                if v != 0.0 && v != 1.0 {
                    zero_one = false;
                }
            }
        }
        let den = n.max(1.0);
        let mean = sum / den;
        let mut var_sum = 0.0f32;
        for i in 0..len {
            if mask[base + i] {
                let d = (values[base + i] - mean) * 1.0;
                var_sum += d * d;
            }
        }
        let var = var_sum / den;
        let mut l = mean;
        let mut s = var.sqrt().max(minimum_scale);
        let binary = zero_one && any;
        if binary {
            l = 0.0;
            s = 1.0;
        }
        loc[row] = l;
        scale[row] = s;
        is_binary[row] = binary;
    }
    (loc, scale, is_binary)
}

/// Welford prefix of per-patch moments. Rows are processed independently.
pub fn cumulative_welford(
    rows: usize,
    mut w: Vec<f64>,
    mut mean: Vec<f64>,
    mut m2: Vec<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = w.len() / rows;
    for row in 0..rows {
        let off = row * n;
        let mut d = 1usize;
        while d < n {
            let w_l = w[off..off + n - d].to_vec();
            let mean_l = mean[off..off + n - d].to_vec();
            let m2_l = m2[off..off + n - d].to_vec();
            for i in d..n {
                let b = off + i;
                let wa = w_l[i - d];
                let ma = mean_l[i - d];
                let m2a = m2_l[i - d];
                let wb = w[b];
                let mb = mean[b];
                let m2b = m2[b];
                let ww = wa + wb;
                let safe = if ww == 0.0 { f64::MIN_POSITIVE } else { ww };
                let delta = mb - ma;
                let new_mean = ma + delta * wb / safe;
                let new_m2 = m2a + m2b + delta * delta * wa * wb / safe;
                w[b] = ww;
                if ww == 0.0 {
                    mean[b] = 0.0;
                    m2[b] = 0.0;
                } else {
                    mean[b] = new_mean;
                    m2[b] = new_m2;
                }
            }
            d *= 2;
        }
    }
    (w, mean, m2)
}

/// Per-patch `(loc, scale)` for a decoded block in the advancing refine frame.
pub fn advancing_block_frames(
    context: &[f32],
    mask: &[bool],
    sinh_median: &[f64],
    rows: usize,
    n_pred: usize,
    patch_size: usize,
    minimum_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let t = context.len() / rows;
    let context_patches = t / patch_size;
    let total = context_patches + n_pred;
    let mut cnts = vec![0.0f64; rows * total];
    let mut means = vec![0.0f64; rows * total];
    let mut m2s = vec![0.0f64; rows * total];

    for row in 0..rows {
        for patch in 0..context_patches {
            let base = row * t + patch * patch_size;
            let mut cnt = 0.0f64;
            let mut sum = 0.0f64;
            for i in 0..patch_size {
                let w = if mask[base + i] { 1.0 } else { 0.0 };
                cnt += w;
                sum += w * context[base + i] as f64;
            }
            let mean = sum / cnt.max(1.0);
            let mut m2 = 0.0f64;
            for i in 0..patch_size {
                if mask[base + i] {
                    let d = context[base + i] as f64 - mean;
                    m2 += d * d;
                }
            }
            let zero = cnt == 0.0;
            let slot = row * total + patch;
            cnts[slot] = cnt;
            means[slot] = if zero { 0.0 } else { mean };
            m2s[slot] = if zero { 0.0 } else { m2 };
        }
        for patch in context_patches..total {
            let slot = row * total + patch;
            cnts[slot] = 0.0;
            means[slot] = 0.0;
            m2s[slot] = 0.0;
        }
    }

    let (cnt_all, mean_all, m2_all) = cumulative_welford(rows, cnts, means, m2s);
    let state_index = context_patches - 1;
    let mut locs = vec![0.0f64; rows * n_pred];
    let mut scales = vec![0.0f64; rows * n_pred];
    let clip = (f32::MAX as f64).sqrt() / 1e3;

    for row in 0..rows {
        let slot = row * total + state_index;
        let mut cnt = cnt_all[slot];
        let mut mean = mean_all[slot];
        let mut m2 = m2_all[slot];
        let offset = mean;
        for h in 0..n_pred {
            let var = (m2 / (cnt - 1.0).max(1.0)).max(0.0);
            let scale = var.sqrt().max(minimum_scale);
            locs[row * n_pred + h] = mean;
            scales[row * n_pred + h] = scale;
            let mut folded = vec![0.0f64; patch_size];
            let mut b_sum = 0.0f64;
            for i in 0..patch_size {
                let v = (sinh_median[(row * n_pred + h) * patch_size + i] * scale + mean - offset)
                    .clamp(-clip, clip)
                    + offset;
                folded[i] = v;
                b_sum += v;
            }
            let b_mean = b_sum / patch_size as f64;
            let mut b_m2 = 0.0f64;
            for v in &folded {
                let d = v - b_mean;
                b_m2 += d * d;
            }
            let ww = cnt + patch_size as f64;
            let safe = if ww == 0.0 { f64::MIN_POSITIVE } else { ww };
            let delta = b_mean - mean;
            let new_mean = mean + delta * patch_size as f64 / safe;
            let new_m2 = m2 + b_m2 + delta * delta * cnt * patch_size as f64 / safe;
            cnt = ww;
            if ww == 0.0 {
                mean = 0.0;
                m2 = 0.0;
            } else {
                mean = new_mean;
                m2 = new_m2;
            }
        }
    }
    (locs, scales)
}

pub fn clamp_nonfinite(values: &mut [f32], q: usize) {
    for row in values.chunks_exact_mut(q) {
        let mut max_finite = f32::NEG_INFINITY;
        let mut min_finite = f32::INFINITY;
        let mut any_finite = false;
        for &v in row.iter() {
            if v.is_finite() {
                any_finite = true;
                max_finite = max_finite.max(v);
                min_finite = min_finite.min(v);
            }
        }
        if !any_finite {
            for v in row.iter_mut() {
                if v.is_infinite() {
                    *v = 0.0;
                }
            }
        } else {
            for v in row.iter_mut() {
                if *v == f32::INFINITY {
                    *v = max_finite;
                } else if *v == f32::NEG_INFINITY {
                    *v = min_finite;
                }
            }
        }
    }
}

/// Non-crossing quantiles from raw head outputs (`monotone_cumsum`).
pub fn monotone_quantiles(raw: &[f32], levels: &[f64], width_factor: f64, out: &mut [f32]) {
    let q = levels.len();
    let m = levels
        .iter()
        .position(|&level| level == 0.5)
        .expect("validated quantile levels contain 0.5");
    let mut widths = vec![0.0f32; q];
    for i in m + 1..q {
        widths[i] = (levels[i] - levels[i - 1]) as f32;
    }
    for i in 0..m {
        widths[i] = (levels[i + 1] - levels[i]) as f32;
    }
    let factor = width_factor as f32;
    let mut inc = vec![0.0f32; q];
    for i in 0..q {
        inc[i] = factor * widths[i] * softplus(raw[i]);
    }
    let median = raw[m];
    let mut up = 0.0f32;
    for i in m + 1..q {
        up += inc[i];
        out[i] = (median + up).clamp(-f32::MAX, f32::MAX);
    }
    let mut down = 0.0f32;
    for i in (0..m).rev() {
        down += inc[i];
        out[i] = (median - down).clamp(-f32::MAX, f32::MAX);
    }
    out[m] = median.clamp(-f32::MAX, f32::MAX);
}
