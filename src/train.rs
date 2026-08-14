//! Early-stop training loops for faithful CMAC on continuous tasks.

use crate::cmac::TilingCmac;
use crate::mlp_cont::FitTrace;
use crate::tasks::{rmse_of, ContSample};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Fit CMAC with held-out early stopping. Restores best-val weights.
pub fn fit_cmac_early_stop(
    cmac: &mut TilingCmac,
    train: &[ContSample],
    val: &[ContSample],
    max_epochs: usize,
    patience: usize,
    seed: u64,
) -> FitTrace {
    cmac.reset_weights();
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xCFAE_0570);
    let mut order: Vec<usize> = (0..train.len()).collect();

    let mut best_val = f64::INFINITY;
    let mut best_epoch = 0usize;
    let mut best_weights: Option<TilingCmac> = None;
    let mut stale = 0usize;
    let mut history = Vec::new();

    let t0 = Instant::now();
    for ep in 1..=max_epochs {
        order.shuffle(&mut rng);
        for &i in &order {
            cmac.train_one(&train[i].x, &train[i].y);
        }
        let tr = rmse_of(train, |x| cmac.predict(x));
        let va = rmse_of(val, |x| cmac.predict(x));
        history.push((ep, tr, va));
        if va < best_val - 1e-6 {
            best_val = va;
            best_epoch = ep;
            best_weights = Some(cmac.clone());
            stale = 0;
        } else {
            stale += 1;
            if stale >= patience {
                break;
            }
        }
    }
    if let Some(best) = best_weights {
        *cmac = best;
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    FitTrace {
        best_epoch,
        best_val_rmse: best_val,
        train_ms: elapsed_ms,
        history,
    }
}

/// One pass timing: mean µs per train_one call (sparse O(C) update).
pub fn time_update_us(cmac: &mut TilingCmac, data: &[ContSample], reps: usize) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let t0 = Instant::now();
    let mut n = 0usize;
    for _ in 0..reps {
        for s in data {
            cmac.train_one(&s.x, &s.y);
            n += 1;
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    (ms * 1000.0) / n as f64 // µs
}
