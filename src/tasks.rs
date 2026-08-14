//! Native continuous tasks for faithful CMAC (Stage 2).
//!
//! - Fn-approx: fixed 2-D nonlinear surface over [0,1]².
//! - Control: open-loop inverse kinematics for a 2-link planar arm.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// One supervised continuous example.
#[derive(Clone, Debug)]
pub struct ContSample {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Fn-approx: sum-of-Gaussians + sin·cos field on [0,1]²
// ---------------------------------------------------------------------------

/// Fixed nonlinear surface (deterministic; no seed).
/// f(x,y) = sin(2πx)·cos(2πy) + Σ_i a_i exp(−||p−c_i||² / (2σ²))
pub fn fn_approx_target(x: f64, y: f64) -> f64 {
    let wave = (2.0 * std::f64::consts::PI * x).sin() * (2.0 * std::f64::consts::PI * y).cos();
    // Three fixed Gaussians
    let gaussians = [
        (0.25, 0.30, 0.80, 0.08),
        (0.70, 0.65, -0.60, 0.10),
        (0.45, 0.80, 0.50, 0.07),
    ];
    let mut g = 0.0;
    for &(cx, cy, a, sig) in &gaussians {
        let dx = x - cx;
        let dy = y - cy;
        g += a * (-0.5 * (dx * dx + dy * dy) / (sig * sig)).exp();
    }
    wave + g
}

pub fn sample_fn_approx(n: usize, seed: u64) -> Vec<ContSample> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xF4A9_0001);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.gen_range(0.0..1.0);
        let y = rng.gen_range(0.0..1.0);
        let t = fn_approx_target(x, y);
        out.push(ContSample {
            x: vec![x, y],
            y: vec![t],
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Open-loop IK: 2-link planar arm
// ---------------------------------------------------------------------------

/// Link lengths (unit workspace roughly in disk of radius L1+L2).
pub const ARM_L1: f64 = 1.0;
pub const ARM_L2: f64 = 1.0;

/// Forward kinematics (for building the dataset via sampling joints).
pub fn fk(theta1: f64, theta2: f64) -> (f64, f64) {
    let x = ARM_L1 * theta1.cos() + ARM_L2 * (theta1 + theta2).cos();
    let y = ARM_L1 * theta1.sin() + ARM_L2 * (theta1 + theta2).sin();
    (x, y)
}

/// Analytic IK (elbow-down branch). Returns None if unreachable.
pub fn ik_analytic(x: f64, y: f64) -> Option<(f64, f64)> {
    let r2 = x * x + y * y;
    let r = r2.sqrt();
    let l1 = ARM_L1;
    let l2 = ARM_L2;
    if r > l1 + l2 - 1e-9 || r < (l1 - l2).abs() + 1e-9 {
        return None;
    }
    let cos_t2 = ((r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
    let theta2 = -cos_t2.acos(); // elbow-down
    let k1 = l1 + l2 * cos_t2;
    let k2 = l2 * theta2.sin();
    let theta1 = y.atan2(x) - k2.atan2(k1);
    Some((theta1, theta2))
}

/// Normalize target (x,y) from arm workspace into roughly [0,1]² for CMAC domain.
/// Workspace is disk radius ~2 centered at origin → map via (x+2)/4, (y+2)/4.
pub fn workspace_to_unit(x: f64, y: f64) -> (f64, f64) {
    ((x + 2.0) / 4.0, (y + 2.0) / 4.0)
}

pub fn unit_to_workspace(u: f64, v: f64) -> (f64, f64) {
    (u * 4.0 - 2.0, v * 4.0 - 2.0)
}

/// Sample reachable targets by sampling joint angles, then store
/// input = unit-scaled (x,y), target = (θ1, θ2).
pub fn sample_ik(n: usize, seed: u64) -> Vec<ContSample> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x1C00_0001);
    let mut out = Vec::with_capacity(n);
    let mut tries = 0;
    while out.len() < n && tries < n * 20 {
        tries += 1;
        // θ1 ∈ [-π, π], θ2 ∈ [-2.5, -0.2] elbow-down band (avoids singularities)
        let t1 = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
        let t2 = rng.gen_range(-2.5..-0.15);
        let (x, y) = fk(t1, t2);
        // Prefer analytic IK as target so the mapping is a true inverse
        let Some((th1, th2)) = ik_analytic(x, y) else {
            continue;
        };
        let (u, v) = workspace_to_unit(x, y);
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
            continue;
        }
        out.push(ContSample {
            x: vec![u, v],
            y: vec![th1, th2],
        });
    }
    assert_eq!(out.len(), n, "failed to sample enough IK points");
    out
}

/// RMSE over a dataset given a predict fn.
pub fn rmse_of<F>(data: &[ContSample], mut predict: F) -> f64
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    if data.is_empty() {
        return f64::NAN;
    }
    let mut se = 0.0;
    let mut n_elem = 0usize;
    for s in data {
        let p = predict(&s.x);
        assert_eq!(p.len(), s.y.len());
        for k in 0..p.len() {
            let e = p[k] - s.y[k];
            se += e * e;
            n_elem += 1;
        }
    }
    (se / n_elem as f64).sqrt()
}
