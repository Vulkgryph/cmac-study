//! Minimal plain-Rust WiSARD + MLP classifiers for Stage 3 MNIST comparability.
//! Carried in spirit from ramnet-study; no burn dependency.

use crate::hash_cmac::BinarySample;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// -------------------- WiSARD (write + optional bleach thr=0 for speed) --------------------

pub struct WisardPlain {
    mapping: Vec<Vec<usize>>,
    /// discs[class][tuple][addr] = u8 count
    discs: Vec<Vec<Vec<u8>>>,
    pub n_bits: usize,
    pub n_tuples: usize,
    pub n_classes: usize,
    pub bleach: u8,
}

impl WisardPlain {
    pub fn new(n_features: usize, n_bits: usize, n_tuples: usize, n_classes: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA15A_4D00);
        let mut mapping = Vec::with_capacity(n_tuples);
        let mut pool: Vec<usize> = (0..n_features).collect();
        pool.shuffle(&mut rng);
        let mut cursor = 0usize;
        for _ in 0..n_tuples {
            let mut tup = Vec::with_capacity(n_bits);
            for _ in 0..n_bits {
                if cursor >= pool.len() {
                    pool.shuffle(&mut rng);
                    cursor = 0;
                }
                tup.push(pool[cursor]);
                cursor += 1;
            }
            mapping.push(tup);
        }
        let n_addr = 1usize << n_bits;
        let discs = (0..n_classes)
            .map(|_| (0..n_tuples).map(|_| vec![0u8; n_addr]).collect())
            .collect();
        Self {
            mapping,
            discs,
            n_bits,
            n_tuples,
            n_classes,
            bleach: 0,
        }
    }

    #[inline]
    fn address(&self, x: &[u8], t: usize) -> usize {
        let mut addr = 0usize;
        for (i, &fi) in self.mapping[t].iter().enumerate() {
            let bit = if fi < x.len() { x[fi] & 1 } else { 0 };
            addr |= (bit as usize) << i;
        }
        addr
    }

    pub fn fit(&mut self, data: &[BinarySample]) {
        for disc in self.discs.iter_mut() {
            for ram in disc.iter_mut() {
                for c in ram.iter_mut() {
                    *c = 0;
                }
            }
        }
        for s in data {
            let c = s.y.min(self.n_classes - 1);
            for t in 0..self.n_tuples {
                let a = self.address(&s.x, t);
                let cell = &mut self.discs[c][t][a];
                *cell = cell.saturating_add(1);
            }
        }
        // Bleach search on a stratified train subset (comparable to #1 protocol).
        self.bleach = self.search_bleach(data);
    }

    /// Threshold sweep: pick bleach maximizing train-subset accuracy.
    /// Candidates: 0..=max_count with coarse steps above 32 (same spirit as #1).
    fn search_bleach(&self, data: &[BinarySample]) -> u8 {
        if data.is_empty() {
            return 0;
        }
        // Stratified val cap ~4k (or all if smaller)
        let cap = 4000.min(data.len());
        let mut by: Vec<Vec<&BinarySample>> = vec![Vec::new(); self.n_classes];
        for s in data {
            if s.y < self.n_classes {
                by[s.y].push(s);
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0xB1EA_C400);
        let per = (cap / self.n_classes).max(1);
        let mut val: Vec<&BinarySample> = Vec::with_capacity(cap);
        for b in by.iter_mut() {
            b.shuffle(&mut rng);
            val.extend(b.iter().take(per.min(b.len())).copied());
        }
        val.truncate(cap);

        let mut max_c = 0u8;
        for s in &val {
            for c in 0..self.n_classes {
                for t in 0..self.n_tuples {
                    let v = self.discs[c][t][self.address(&s.x, t)];
                    if v > max_c {
                        max_c = v;
                    }
                }
            }
        }
        let mut cands = Vec::new();
        let dense = 32u8.min(max_c);
        for b in 0..=dense {
            cands.push(b);
        }
        let mut b = dense.saturating_add(1);
        while b <= max_c {
            cands.push(b);
            let step = if b < 64 { 2 } else if b < 128 { 4 } else { 8 };
            let nb = b.saturating_add(step);
            if nb <= b {
                break;
            }
            b = nb;
        }
        if !cands.contains(&max_c) {
            cands.push(max_c);
        }

        let mut best_b = 0u8;
        let mut best_acc = -1.0f64;
        for &thr in &cands {
            let mut ok = 0usize;
            for s in &val {
                if self.predict_with(s.x.as_slice(), thr) == s.y {
                    ok += 1;
                }
            }
            let acc = ok as f64 / val.len() as f64;
            if acc > best_acc || (acc == best_acc && thr > best_b) {
                best_acc = acc;
                best_b = thr;
            }
        }
        best_b
    }

    fn predict_with(&self, x: &[u8], bleach: u8) -> usize {
        let mut best_c = 0usize;
        let mut best_s = 0usize;
        for c in 0..self.n_classes {
            let mut score = 0usize;
            for t in 0..self.n_tuples {
                let a = self.address(x, t);
                if self.discs[c][t][a] > bleach {
                    score += 1;
                }
            }
            if score > best_s {
                best_s = score;
                best_c = c;
            }
        }
        best_c
    }

    pub fn predict(&self, x: &[u8]) -> usize {
        self.predict_with(x, self.bleach)
    }

    pub fn accuracy(&self, data: &[BinarySample]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut ok = 0usize;
        for s in data {
            if self.predict(&s.x) == s.y {
                ok += 1;
            }
        }
        ok as f64 / data.len() as f64
    }

    pub fn trainable_params(&self) -> u64 {
        0
    }

    pub fn ram_entries(&self) -> u64 {
        (self.n_classes as u64)
            * (self.n_tuples as u64)
            * (1u64 << self.n_bits)
    }
}

// -------------------- Plain MLP classifier (SGD, one hidden ReLU) --------------------

pub struct MlpClass {
    n_in: usize,
    n_hidden: usize,
    n_out: usize,
    lr: f64,
    epochs: usize,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

impl MlpClass {
    pub fn new(n_in: usize, n_hidden: usize, n_out: usize, lr: f64, epochs: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA11B_C1A5);
        let s1 = (2.0 / n_in as f64).sqrt();
        let s2 = (2.0 / n_hidden as f64).sqrt();
        let w1 = (0..n_hidden * n_in)
            .map(|_| rng.gen_range(-s1..s1))
            .collect();
        let b1 = vec![0.0; n_hidden];
        let w2 = (0..n_out * n_hidden)
            .map(|_| rng.gen_range(-s2..s2))
            .collect();
        let b2 = vec![0.0; n_out];
        Self {
            n_in,
            n_hidden,
            n_out,
            lr,
            epochs,
            w1,
            b1,
            w2,
            b2,
        }
    }

    pub fn trainable_params(&self) -> u64 {
        (self.n_hidden * self.n_in + self.n_hidden + self.n_out * self.n_hidden + self.n_out) as u64
    }

    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut h = vec![0.0; self.n_hidden];
        for j in 0..self.n_hidden {
            let mut s = self.b1[j];
            for i in 0..self.n_in {
                s += self.w1[j * self.n_in + i] * x[i];
            }
            h[j] = s.max(0.0); // ReLU
        }
        let mut logits = vec![0.0; self.n_out];
        for o in 0..self.n_out {
            let mut s = self.b2[o];
            for j in 0..self.n_hidden {
                s += self.w2[o * self.n_hidden + j] * h[j];
            }
            logits[o] = s;
        }
        (h, logits)
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = logits.iter().map(|z| (z - m).exp()).collect();
        let s: f64 = ex.iter().sum();
        ex.into_iter().map(|e| e / s).collect()
    }

    pub fn fit(&mut self, data: &[BinarySample], seed: u64) {
        // re-init
        *self = Self::new(self.n_in, self.n_hidden, self.n_out, self.lr, self.epochs, seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x56D0_0001);
        let mut order: Vec<usize> = (0..data.len()).collect();
        let lr = self.lr;
        for _ in 0..self.epochs {
            order.shuffle(&mut rng);
            for &idx in &order {
                let s = &data[idx];
                let x: Vec<f64> = s.x.iter().take(self.n_in).map(|&b| b as f64).collect();
                let (h, logits) = self.forward(&x);
                let p = Self::softmax(&logits);
                // d_logits = p - onehot
                let mut d_out = p;
                d_out[s.y.min(self.n_out - 1)] -= 1.0;
                // backprop W2
                let mut d_h = vec![0.0; self.n_hidden];
                for o in 0..self.n_out {
                    for j in 0..self.n_hidden {
                        d_h[j] += self.w2[o * self.n_hidden + j] * d_out[o];
                        self.w2[o * self.n_hidden + j] -= lr * d_out[o] * h[j];
                    }
                    self.b2[o] -= lr * d_out[o];
                }
                // ReLU'
                for j in 0..self.n_hidden {
                    if h[j] <= 0.0 {
                        d_h[j] = 0.0;
                    }
                    for i in 0..self.n_in {
                        self.w1[j * self.n_in + i] -= lr * d_h[j] * x[i];
                    }
                    self.b1[j] -= lr * d_h[j];
                }
            }
        }
    }

    pub fn predict(&self, x_bits: &[u8]) -> usize {
        let x: Vec<f64> = x_bits.iter().take(self.n_in).map(|&b| b as f64).collect();
        let (_, logits) = self.forward(&x);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn accuracy(&self, data: &[BinarySample]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut ok = 0usize;
        for s in data {
            if self.predict(&s.x) == s.y {
                ok += 1;
            }
        }
        ok as f64 / data.len() as f64
    }
}
