//! Faithful tiling CMAC (Albus, 1975) for low-dimensional continuous inputs.
//!
//! - C overlapping tilings of the input domain.
//! - Each tiling partitions each dimension into tiles of width `tile_width`.
//! - Tiling `i` is offset by `(i/C) * tile_width` along every dimension
//!   (classic uniform stagger — guarantees local generalization).
//! - An input activates exactly one tile per tiling → C active cells.
//! - Tile coordinates are hashed into a bounded table of size `table_size`
//!   (memory bound only; the *geometry* is the tiling, not the hash).
//! - Output = sum of the C active cells' weight vectors.
//! - Learning = local delta: w_a += (η/C) · (target − output) on the C cells.
//!
//! This module is the FAITHFUL anchor. Do not use random-bit hashing here;
//! that lives in `hash_cmac` for the out-of-domain MNIST arm.

use serde::Serialize;

/// Multi-output faithful CMAC over a hyper-rectangle domain.
#[derive(Clone, Debug)]
pub struct TilingCmac {
    /// Input dimensionality (e.g. 2 for fn-approx / IK).
    pub n_in: usize,
    /// Output dimensionality (1 for scalar fn-approx; 2 for joint angles).
    pub n_out: usize,
    /// Number of overlapping tilings (= number of active cells per input).
    pub c: usize,
    /// Tile width along each dimension (same scale as the domain).
    pub tile_width: f64,
    /// Domain lower bound per dimension (inclusive).
    pub domain_lo: Vec<f64>,
    /// Domain upper bound per dimension (exclusive-ish; clamped).
    pub domain_hi: Vec<f64>,
    /// Hash table bins per tiling.
    pub table_size: usize,
    /// Learning rate η.
    pub eta: f64,
    /// Per-tiling offsets: offsets[t][d] ∈ [0, tile_width).
    offsets: Vec<Vec<f64>>,
    /// weights[t][addr * n_out + k]
    weights: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CmacInfo {
    pub name: String,
    pub n_in: usize,
    pub n_out: usize,
    pub c: usize,
    pub tile_width: f64,
    pub table_size: usize,
    pub eta: f64,
    /// Total float weights = C · table_size · n_out.
    pub trainable_params: u64,
    /// Active cells per example (always = C for faithful CMAC).
    pub active_cells: usize,
}

impl TilingCmac {
    /// Build a faithful CMAC.
    ///
    /// `domain_lo` / `domain_hi` length = `n_in`.
    /// Offsets: tiling `t` is shifted by `(t as f64 / c) * tile_width` on every dim.
    pub fn new(
        n_in: usize,
        n_out: usize,
        c: usize,
        tile_width: f64,
        domain_lo: Vec<f64>,
        domain_hi: Vec<f64>,
        table_size: usize,
        eta: f64,
    ) -> Self {
        assert!(n_in > 0 && n_out > 0);
        assert!(c > 0);
        assert!(tile_width > 0.0);
        assert_eq!(domain_lo.len(), n_in);
        assert_eq!(domain_hi.len(), n_in);
        assert!(table_size > 0);
        assert!(eta >= 0.0);

        let offsets: Vec<Vec<f64>> = (0..c)
            .map(|t| {
                let o = (t as f64 / c as f64) * tile_width;
                vec![o; n_in]
            })
            .collect();

        let weights = (0..c)
            .map(|_| vec![0.0f64; table_size * n_out])
            .collect();

        Self {
            n_in,
            n_out,
            c,
            tile_width,
            domain_lo,
            domain_hi,
            table_size,
            eta,
            offsets,
            weights,
        }
    }

    /// Convenience: unit hypercube [0,1]^n_in.
    pub fn unit_cube(n_in: usize, n_out: usize, c: usize, tile_width: f64, table_size: usize, eta: f64) -> Self {
        Self::new(
            n_in,
            n_out,
            c,
            tile_width,
            vec![0.0; n_in],
            vec![1.0; n_in],
            table_size,
            eta,
        )
    }

    /// Clamp input into the domain.
    fn clamp_input(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n_in);
        x.iter()
            .enumerate()
            .map(|(d, &v)| v.clamp(self.domain_lo[d], self.domain_hi[d] - 1e-12))
            .collect()
    }

    /// Integer tile coordinate along dimension `d` for tiling `t`.
    #[inline]
    fn tile_coord(&self, x_clamped: &[f64], t: usize, d: usize) -> i64 {
        // Shift by offset, then quantize.
        let shifted = x_clamped[d] - self.domain_lo[d] + self.offsets[t][d];
        (shifted / self.tile_width).floor() as i64
    }

    /// Hash tile coordinates for tiling `t` → [0, table_size).
    /// Geometry is the coords; hash is memory-only.
    #[inline]
    fn hash_coords(&self, coords: &[i64]) -> usize {
        // FNV-1a over i64 bytes
        let mut h: u64 = 0xcbf29ce484222325;
        for &c in coords {
            for b in c.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        (h as usize) % self.table_size
    }

    /// Active addresses: one per tiling. Also returns the raw tile coords
    /// (for the local-generalization gate — no hash).
    pub fn active_tiles(&self, x: &[f64]) -> (Vec<usize>, Vec<Vec<i64>>) {
        let xc = self.clamp_input(x);
        let mut addrs = Vec::with_capacity(self.c);
        let mut all_coords = Vec::with_capacity(self.c);
        for t in 0..self.c {
            let mut coords = Vec::with_capacity(self.n_in);
            for d in 0..self.n_in {
                coords.push(self.tile_coord(&xc, t, d));
            }
            addrs.push(self.hash_coords(&coords));
            all_coords.push(coords);
        }
        (addrs, all_coords)
    }

    /// How many tilings share the *exact same tile coordinates* between x and y?
    /// This is the geometric overlap (pre-hash) — the local-generalization signal.
    pub fn shared_tile_count(&self, x: &[f64], y: &[f64]) -> usize {
        let (_, cx) = self.active_tiles(x);
        let (_, cy) = self.active_tiles(y);
        cx.iter().zip(cy.iter()).filter(|(a, b)| a == b).count()
    }

    /// Forward: sum of C active weight vectors → length n_out.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        let (addrs, _) = self.active_tiles(x);
        let mut out = vec![0.0f64; self.n_out];
        for (t, &a) in addrs.iter().enumerate() {
            let base = a * self.n_out;
            for k in 0..self.n_out {
                out[k] += self.weights[t][base + k];
            }
        }
        out
    }

    /// One supervised step. `target` length = n_out.
    /// Returns squared error ||target − output||² (pre-update).
    pub fn train_one(&mut self, x: &[f64], target: &[f64]) -> f64 {
        assert_eq!(target.len(), self.n_out);
        let (addrs, _) = self.active_tiles(x);
        let mut out = vec![0.0f64; self.n_out];
        for (t, &a) in addrs.iter().enumerate() {
            let base = a * self.n_out;
            for k in 0..self.n_out {
                out[k] += self.weights[t][base + k];
            }
        }
        let mut se = 0.0f64;
        let mut err = vec![0.0f64; self.n_out];
        for k in 0..self.n_out {
            err[k] = target[k] - out[k];
            se += err[k] * err[k];
        }
        let step = self.eta / self.c as f64;
        for (t, &a) in addrs.iter().enumerate() {
            let base = a * self.n_out;
            for k in 0..self.n_out {
                self.weights[t][base + k] += step * err[k];
            }
        }
        se
    }

    /// Zero all weights (fresh fit).
    pub fn reset_weights(&mut self) {
        for w in self.weights.iter_mut() {
            for v in w.iter_mut() {
                *v = 0.0;
            }
        }
    }

    pub fn describe(&self) -> CmacInfo {
        CmacInfo {
            name: format!(
                "tiling_cmac(C={}, w={:.4}, table={}, η={})",
                self.c, self.tile_width, self.table_size, self.eta
            ),
            n_in: self.n_in,
            n_out: self.n_out,
            c: self.c,
            tile_width: self.tile_width,
            table_size: self.table_size,
            eta: self.eta,
            trainable_params: (self.c as u64) * (self.table_size as u64) * (self.n_out as u64),
            active_cells: self.c,
        }
    }

    /// Active cells per example — always C (faithfulness invariant).
    pub fn active_cells_per_example(&self) -> usize {
        self.c
    }
}

/// Local-generalization probe: shared-tile count vs Euclidean distance.
///
/// Gate criteria (for unit-cube 2-D, tile_width = w):
/// - distance ≪ w  → shared ≈ C
/// - distance ≫ C·w (or several tile widths) → shared ≈ 0
#[derive(Clone, Debug, Serialize)]
pub struct GenProbeRow {
    pub distance: f64,
    pub mean_shared: f64,
    pub min_shared: usize,
    pub max_shared: usize,
    pub n_pairs: usize,
}

pub fn probe_local_generalization(
    cmac: &TilingCmac,
    distances: &[f64],
    pairs_per_distance: usize,
    seed: u64,
) -> Vec<GenProbeRow> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x10CA_17E5);
    let mut rows = Vec::new();

    for &dist in distances {
        let mut shareds = Vec::with_capacity(pairs_per_distance);
        let mut attempts = 0;
        while shareds.len() < pairs_per_distance && attempts < pairs_per_distance * 50 {
            attempts += 1;
            // Random anchor in domain.
            let mut x = vec![0.0f64; cmac.n_in];
            for d in 0..cmac.n_in {
                x[d] = rng.gen_range(cmac.domain_lo[d]..cmac.domain_hi[d]);
            }
            // Random direction, fixed length `dist`.
            let mut dir = vec![0.0f64; cmac.n_in];
            let mut norm = 0.0f64;
            for d in 0..cmac.n_in {
                dir[d] = rng.gen_range(-1.0..1.0);
                norm += dir[d] * dir[d];
            }
            if norm < 1e-12 {
                continue;
            }
            norm = norm.sqrt();
            let mut y = vec![0.0f64; cmac.n_in];
            let mut in_domain = true;
            for d in 0..cmac.n_in {
                y[d] = x[d] + dist * dir[d] / norm;
                if y[d] < cmac.domain_lo[d] || y[d] >= cmac.domain_hi[d] {
                    in_domain = false;
                    break;
                }
            }
            if !in_domain {
                continue;
            }
            shareds.push(cmac.shared_tile_count(&x, &y));
        }
        if shareds.is_empty() {
            rows.push(GenProbeRow {
                distance: dist,
                mean_shared: f64::NAN,
                min_shared: 0,
                max_shared: 0,
                n_pairs: 0,
            });
            continue;
        }
        let n = shareds.len();
        let sum: usize = shareds.iter().sum();
        let min = *shareds.iter().min().unwrap();
        let max = *shareds.iter().max().unwrap();
        rows.push(GenProbeRow {
            distance: dist,
            mean_shared: sum as f64 / n as f64,
            min_shared: min,
            max_shared: max,
            n_pairs: n,
        });
    }
    rows
}
