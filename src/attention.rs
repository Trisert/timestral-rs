use matrixmultiply::sgemm;
use rayon::prelude::*;

use crate::ops::{dot, exp_fast};

#[derive(Debug, Clone)]
pub enum TimeMaskKind {
    Forecast,
    TwoStream {
        ll: usize,
        block_size: usize,
        n: usize,
        bidirectional: Vec<bool>,
    },
}

#[derive(Debug, Clone)]
pub struct TimeMask {
    pub rows: usize,
    pub total: usize,
    pub enc_len: usize,
    /// `(rows, enc_len)`, true when the encoder key patch may be attended.
    pub key_valid: Vec<bool>,
    pub kind: TimeMaskKind,
}

impl TimeMask {
    #[inline]
    fn key_valid(&self, row: usize, key: usize) -> bool {
        self.key_valid[row * self.enc_len + key]
    }

    #[inline]
    pub fn allowed(&self, row: usize, query: usize, key: usize) -> bool {
        if query < self.enc_len {
            if key >= self.enc_len {
                return false;
            }
            if key == query {
                return true;
            }
            match &self.kind {
                TimeMaskKind::Forecast => key <= query && self.key_valid(row, key),
                TimeMaskKind::TwoStream { bidirectional, .. } => {
                    if bidirectional[row] {
                        self.key_valid(row, key)
                    } else {
                        key <= query && self.key_valid(row, key)
                    }
                }
            }
        } else {
            match &self.kind {
                TimeMaskKind::Forecast => {
                    if key < self.enc_len {
                        self.key_valid(row, key)
                    } else {
                        key <= query
                    }
                }
                TimeMaskKind::TwoStream { ll, block_size, .. } => {
                    let p = query - self.enc_len;
                    if key < self.enc_len {
                        if p < *ll {
                            return false;
                        }
                        let anchor = *ll + block_size * ((p - *ll) / block_size);
                        key < anchor && self.key_valid(row, key)
                    } else {
                        let pk = key - self.enc_len;
                        if pk == p {
                            return true;
                        }
                        if p < *ll || pk < *ll {
                            return false;
                        }
                        let anchor_p = *ll + block_size * ((p - *ll) / block_size);
                        let anchor_k = *ll + block_size * ((pk - *ll) / block_size);
                        anchor_p == anchor_k && pk <= p
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct VariateMask {
    pub rows: usize,
    pub slots: usize,
    pub enc_len: usize,
    /// `(rows, enc_len)`, valid key rows per encoder slot.
    pub key_valid: Vec<bool>,
    pub group: Vec<usize>,
    /// `(rows, rows)`, forbidden query-to-key row pairs.
    pub forbid: Option<Vec<bool>>,
}

impl VariateMask {
    #[inline]
    pub fn allowed(&self, query_row: usize, key_row: usize, slot: usize) -> bool {
        if query_row == key_row {
            return true;
        }
        if self.group[query_row] != self.group[key_row] {
            return false;
        }
        if slot < self.enc_len && !self.key_valid[key_row * self.enc_len + slot] {
            return false;
        }
        if let Some(forbid) = &self.forbid {
            if forbid[query_row * self.rows + key_row] {
                return false;
            }
        }
        true
    }
}

pub fn build_rope_tables(
    positions: &[i64],
    hd: usize,
    rope_dim: usize,
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let mut inv_freq = vec![0.0f32; half];
    for (i, value) in inv_freq.iter_mut().enumerate().take(rope_dim / 2) {
        *value = (1.0 / theta.powf((2 * i) as f64 / rope_dim as f64)) as f32;
    }
    let tokens = positions.len();
    let mut cos = vec![0.0f32; tokens * hd];
    let mut sin = vec![0.0f32; tokens * hd];
    for (token, &pos) in positions.iter().enumerate() {
        let p = pos as f32;
        for d in 0..half {
            let angle = p * inv_freq[d];
            let c = angle.cos();
            let s = angle.sin();
            cos[token * hd + d] = c;
            cos[token * hd + half + d] = c;
            sin[token * hd + d] = s;
            sin[token * hd + half + d] = s;
        }
    }
    (cos, sin)
}

pub fn apply_rope(
    qk: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    tokens: usize,
    heads: usize,
    hd: usize,
) {
    let mut rotated = vec![0.0f32; hd];
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * hd;
            rotated.copy_from_slice(&qk[base..base + hd]);
            let c = &cos[token * hd..token * hd + hd];
            let s = &sin[token * hd..token * hd + hd];
            for d in 0..hd {
                let other = if d < hd / 2 {
                    -rotated[d + hd / 2]
                } else {
                    rotated[d - hd / 2]
                };
                qk[base + d] = rotated[d] * c[d] + other * s[d];
            }
        }
    }
}

/// Multi-head attention over the time axis. `q/k/v/out` are `[token][head][hd]`.
#[allow(clippy::too_many_arguments)]
pub fn time_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    total: usize,
    heads: usize,
    hd: usize,
    mask: &TimeMask,
    out: &mut [f32],
) {
    let d_model = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    debug_assert_eq!(out.len(), rows * total * d_model);
    let mut probs = vec![0.0f32; total * total];
    let stride = (heads * hd) as isize;
    for row in 0..rows {
        let row_base = ((row * total) * heads) * hd;
        for head in 0..heads {
            let head_base = row_base + head * hd;
            unsafe {
                sgemm(
                    total,
                    hd,
                    total,
                    scale,
                    q.as_ptr().add(head_base),
                    stride,
                    1,
                    k.as_ptr().add(head_base),
                    1,
                    stride,
                    0.0,
                    probs.as_mut_ptr(),
                    total as isize,
                    1,
                );
            }
            for query in 0..total {
                let scores = &mut probs[query * total..(query + 1) * total];
                let mut max_score = f32::NEG_INFINITY;
                for (key, score) in scores.iter_mut().enumerate() {
                    if mask.allowed(row, query, key) {
                        if *score > max_score {
                            max_score = *score;
                        }
                    } else {
                        *score = f32::NEG_INFINITY;
                    }
                }
                let mut denominator = 0.0f32;
                for score in scores.iter_mut() {
                    if *score > f32::NEG_INFINITY {
                        let e = exp_fast(*score - max_score);
                        *score = e;
                        denominator += e;
                    } else {
                        *score = 0.0;
                    }
                }
                let inv = 1.0 / denominator;
                for score in scores.iter_mut() {
                    *score *= inv;
                }
            }
            unsafe {
                sgemm(
                    total,
                    total,
                    hd,
                    1.0,
                    probs.as_ptr(),
                    total as isize,
                    1,
                    v.as_ptr().add(head_base),
                    stride,
                    1,
                    0.0,
                    out.as_mut_ptr().add(head_base),
                    stride,
                    1,
                );
            }
        }
    }
}

/// Multi-head attention over the variate axis, per time slot.
#[allow(clippy::too_many_arguments)]
pub fn variate_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    slots: usize,
    heads: usize,
    hd: usize,
    mask: &VariateMask,
    out: &mut [f32],
) {
    let d_model = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let compute = |token: usize, chunk: &mut [f32]| {
        let query_row = token / slots;
        let slot = token % slots;
        chunk.fill(0.0);
        if rows == 1 {
            for head in 0..heads {
                let base = (token * heads + head) * hd;
                chunk[head * hd..(head + 1) * hd].copy_from_slice(&v[base..base + hd]);
            }
            return;
        }
        let mut scores = vec![f32::NEG_INFINITY; rows];
        for head in 0..heads {
            let q_base = (token * heads + head) * hd;
            let qv = &q[q_base..q_base + hd];
            for (key_row, score) in scores.iter_mut().enumerate() {
                if mask.allowed(query_row, key_row, slot) {
                    let k_base = ((key_row * slots + slot) * heads + head) * hd;
                    *score = dot(qv, &k[k_base..k_base + hd]) * scale;
                } else {
                    *score = f32::NEG_INFINITY;
                }
            }
            let mut max_score = f32::NEG_INFINITY;
            for &score in &scores {
                if score > max_score {
                    max_score = score;
                }
            }
            let mut denominator = 0.0f32;
            for score in scores.iter_mut() {
                if *score > f32::NEG_INFINITY {
                    let e = exp_fast(*score - max_score);
                    *score = e;
                    denominator += e;
                } else {
                    *score = 0.0;
                }
            }
            let inv = 1.0 / denominator;
            let out_head = head * hd;
            for (key_row, &weight) in scores.iter().enumerate() {
                if weight != 0.0 {
                    let weight = weight * inv;
                    let v_base = ((key_row * slots + slot) * heads + head) * hd;
                    for i in 0..hd {
                        chunk[out_head + i] += weight * v[v_base + i];
                    }
                }
            }
        }
    };
    if rows * slots >= 256 {
        out.par_chunks_mut(d_model)
            .enumerate()
            .for_each(|(token, chunk)| compute(token, chunk));
    } else {
        out.chunks_mut(d_model)
            .enumerate()
            .for_each(|(token, chunk)| compute(token, chunk));
    }
}
