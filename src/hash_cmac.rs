//! Hashing-CMAC classifier — **out-of-domain / CMAC-inspired** for high-dim binary input.
//!
//! NOT the faithful tiling CMAC. Each of C "tilings" is a random bit-subset hash
//! into a bounded table. Local geometric generalization is not claimed.
//! Used only for the MNIST comparability arm (SPEC M1–M3).

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct BinarySample {
    pub x: Vec<u8>,
    pub y: usize,
}

#[derive(Clone, Debug)]
pub struct HashCmac {
    bit_sets: Vec<Vec<usize>>,
    weights: Vec<Vec<f32>>,
    pub table_size: usize,
    pub n_classes: usize,
    pub c: usize,
    pub bits_per_tile: usize,
    pub eta: f32,
    pub epochs: usize,
}

impl HashCmac {
    pub fn new(
        n_features: usize,
        n_classes: usize,
        c: usize,
        bits_per_tile: usize,
        table_size: usize,
        eta: f32,
        epochs: usize,
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC0AC_0001);
        let mut bit_sets = Vec::with_capacity(c);
        for _ in 0..c {
            let mut pool: Vec<usize> = (0..n_features).collect();
            pool.shuffle(&mut rng);
            bit_sets.push(pool[..bits_per_tile.min(n_features)].to_vec());
        }
        let weights = (0..c)
            .map(|_| vec![0.0f32; table_size * n_classes])
            .collect();
        Self {
            bit_sets,
            weights,
            table_size,
            n_classes,
            c,
            bits_per_tile,
            eta,
            epochs,
        }
    }

    #[inline]
    fn address(&self, x: &[u8], t: usize) -> usize {
        let mut h: u64 = 0xcbf29ce484222325;
        for &fi in &self.bit_sets[t] {
            let b = if fi < x.len() { x[fi] & 1 } else { 0 };
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= (fi as u64).wrapping_mul(0x9e3779b97f4a7c15);
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % self.table_size
    }

    fn scores(&self, x: &[u8]) -> Vec<f32> {
        let mut s = vec![0.0f32; self.n_classes];
        for t in 0..self.c {
            let a = self.address(x, t);
            let base = a * self.n_classes;
            for k in 0..self.n_classes {
                s[k] += self.weights[t][base + k];
            }
        }
        s
    }

    pub fn predict(&self, x: &[u8]) -> usize {
        let s = self.scores(x);
        s.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn train_one(&mut self, x: &[u8], y: usize) {
        let out = self.scores(x);
        let mut err = vec![0.0f32; self.n_classes];
        for k in 0..self.n_classes {
            let t = if k == y { 1.0 } else { 0.0 };
            err[k] = t - out[k];
        }
        let step = self.eta / self.c as f32;
        for t in 0..self.c {
            let a = self.address(x, t);
            let base = a * self.n_classes;
            for k in 0..self.n_classes {
                self.weights[t][base + k] += step * err[k];
            }
        }
    }

    pub fn fit(&mut self, data: &[BinarySample], seed: u64) {
        // reset
        for w in self.weights.iter_mut() {
            for v in w.iter_mut() {
                *v = 0.0;
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5A0F_0001);
        let mut order: Vec<usize> = (0..data.len()).collect();
        for _ in 0..self.epochs {
            order.shuffle(&mut rng);
            for &i in &order {
                self.train_one(&data[i].x, data[i].y);
            }
        }
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
        (self.c as u64) * (self.table_size as u64) * (self.n_classes as u64)
    }

    pub fn active_cells(&self) -> usize {
        self.c
    }
}
