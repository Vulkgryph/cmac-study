//! Tiny plain-Rust MLP for continuous regression (native-task baseline).
//! No autograd framework — hand-rolled SGD, one hidden layer, tanh.

use crate::tasks::ContSample;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct MlpCont {
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_out: usize,
    pub lr: f64,
    // W1: [hidden, in], b1: [hidden]
    w1: Vec<f64>,
    b1: Vec<f64>,
    // W2: [out, hidden], b2: [out]
    w2: Vec<f64>,
    b2: Vec<f64>,
}

impl MlpCont {
    pub fn new(n_in: usize, n_hidden: usize, n_out: usize, lr: f64, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA11B_0001);
        let scale1 = (2.0 / n_in as f64).sqrt();
        let scale2 = (2.0 / n_hidden as f64).sqrt();
        let w1 = (0..n_hidden * n_in)
            .map(|_| rng.gen_range(-scale1..scale1))
            .collect();
        let b1 = vec![0.0; n_hidden];
        let w2 = (0..n_out * n_hidden)
            .map(|_| rng.gen_range(-scale2..scale2))
            .collect();
        let b2 = vec![0.0; n_out];
        Self {
            n_in,
            n_hidden,
            n_out,
            lr,
            w1,
            b1,
            w2,
            b2,
        }
    }

    pub fn trainable_params(&self) -> u64 {
        (self.n_hidden * self.n_in + self.n_hidden + self.n_out * self.n_hidden + self.n_out) as u64
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let h = self.forward_hidden(x);
        let mut out = vec![0.0; self.n_out];
        for o in 0..self.n_out {
            let mut s = self.b2[o];
            for j in 0..self.n_hidden {
                s += self.w2[o * self.n_hidden + j] * h[j];
            }
            out[o] = s;
        }
        out
    }

    fn forward_hidden(&self, x: &[f64]) -> Vec<f64> {
        let mut h = vec![0.0; self.n_hidden];
        for j in 0..self.n_hidden {
            let mut s = self.b1[j];
            for i in 0..self.n_in {
                s += self.w1[j * self.n_in + i] * x[i];
            }
            h[j] = s.tanh();
        }
        h
    }

    pub fn train_one(&mut self, x: &[f64], target: &[f64]) -> f64 {
        // Forward
        let mut z1 = vec![0.0; self.n_hidden];
        let mut h = vec![0.0; self.n_hidden];
        for j in 0..self.n_hidden {
            let mut s = self.b1[j];
            for i in 0..self.n_in {
                s += self.w1[j * self.n_in + i] * x[i];
            }
            z1[j] = s;
            h[j] = s.tanh();
        }
        let mut out = vec![0.0; self.n_out];
        for o in 0..self.n_out {
            let mut s = self.b2[o];
            for j in 0..self.n_hidden {
                s += self.w2[o * self.n_hidden + j] * h[j];
            }
            out[o] = s;
        }
        // Error
        let mut d_out = vec![0.0; self.n_out];
        let mut se = 0.0;
        for o in 0..self.n_out {
            d_out[o] = out[o] - target[o];
            se += d_out[o] * d_out[o];
        }
        // dW2, db2
        let mut d_h = vec![0.0; self.n_hidden];
        for o in 0..self.n_out {
            for j in 0..self.n_hidden {
                d_h[j] += self.w2[o * self.n_hidden + j] * d_out[o];
                self.w2[o * self.n_hidden + j] -= self.lr * d_out[o] * h[j];
            }
            self.b2[o] -= self.lr * d_out[o];
        }
        // dW1, db1 through tanh'
        for j in 0..self.n_hidden {
            let dt = d_h[j] * (1.0 - h[j] * h[j]);
            for i in 0..self.n_in {
                self.w1[j * self.n_in + i] -= self.lr * dt * x[i];
            }
            self.b1[j] -= self.lr * dt;
        }
        se
    }

    pub fn fit_early_stop(
        &mut self,
        train: &[ContSample],
        val: &[ContSample],
        max_epochs: usize,
        patience: usize,
        seed: u64,
    ) -> FitTrace {
        use crate::tasks::rmse_of;
        use rand::seq::SliceRandom;

        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xF17E_5A00);
        let mut order: Vec<usize> = (0..train.len()).collect();
        let mut best_val = f64::INFINITY;
        let mut best_epoch = 0usize;
        let mut best_snap = self.clone();
        let mut stale = 0usize;
        let mut history = Vec::new();

        let t0 = std::time::Instant::now();
        for ep in 1..=max_epochs {
            order.shuffle(&mut rng);
            for &i in &order {
                self.train_one(&train[i].x, &train[i].y);
            }
            let tr = rmse_of(train, |x| self.predict(x));
            let va = rmse_of(val, |x| self.predict(x));
            history.push((ep, tr, va));
            if va < best_val - 1e-6 {
                best_val = va;
                best_epoch = ep;
                best_snap = self.clone();
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }
        *self = best_snap;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        FitTrace {
            best_epoch,
            best_val_rmse: best_val,
            train_ms: elapsed_ms,
            history,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FitTrace {
    pub best_epoch: usize,
    pub best_val_rmse: f64,
    pub train_ms: f64,
    pub history: Vec<(usize, f64, f64)>, // epoch, train_rmse, val_rmse
}
