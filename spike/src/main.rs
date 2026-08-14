//! Minimal hashing-CMAC spike — plain Rust, no ML framework.
//!
//! Question: does hashing-CMAC + local delta rule learn *anything* on
//! high-dim binary input (binarized MNIST)?
//!
//! - C tilings: each tiling = fixed random bit-subset → hash → table index
//! - Table: [C][table_size][n_classes] f32 weights
//! - Forward: sum active cells' class vectors → argmax
//! - Learn: error = onehot(y) − output; Δw = (η/C) · error on the C active cells only
//!
//! Tiny task: 3 classes, few hundred train, held-out test, fixed seed.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const N_FEATURES: usize = 28 * 28;
const SEED: u64 = 0;

// --- knobs (printed at start) ---
const N_CLASSES: usize = 3; // digits 0,1,2
const C: usize = 32; // number of tilings / active cells
const BITS_PER_TILE: usize = 16; // random input bits per tiling
const TABLE_SIZE: usize = 4096; // hash table bins per tiling
const ETA: f32 = 0.25; // learning rate
const EPOCHS: usize = 20;
const TRAIN_PER_CLASS: usize = 200;
const TEST_PER_CLASS: usize = 100;

struct Sample {
    x: Vec<u8>, // {0,1}^784
    y: usize,
}

/// Hashing-CMAC classifier.
struct HashCmac {
    /// For each of C tilings: which input bit indices to read.
    bit_sets: Vec<Vec<usize>>,
    /// weights[c][addr * n_classes + k]
    weights: Vec<Vec<f32>>,
    table_size: usize,
    n_classes: usize,
    c: usize,
    eta: f32,
}

impl HashCmac {
    fn new(n_features: usize, n_classes: usize, c: usize, bits: usize, table_size: usize, eta: f32, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC0AC_0001);
        let mut bit_sets = Vec::with_capacity(c);
        for _ in 0..c {
            let mut pool: Vec<usize> = (0..n_features).collect();
            pool.shuffle(&mut rng);
            bit_sets.push(pool[..bits.min(n_features)].to_vec());
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
            eta,
        }
    }

    /// Hash the selected bits of x for tiling t → [0, table_size).
    #[inline]
    fn address(&self, x: &[u8], t: usize) -> usize {
        // FNV-1a over the ordered bit values in this tiling's subset.
        let mut h: u64 = 0xcbf29ce484222325;
        for &fi in &self.bit_sets[t] {
            let b = if fi < x.len() { x[fi] & 1 } else { 0 };
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
            // mix position so bit-order matters
            h ^= (fi as u64).wrapping_mul(0x9e3779b97f4a7c15);
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % self.table_size
    }

    /// Softmax-free raw scores: sum of C active class-vectors.
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

    fn predict(&self, x: &[u8]) -> usize {
        let s = self.scores(x);
        s.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// One example: local delta on the C active cells.
    /// Returns squared error ||target − output||^2 (pre-update scores).
    fn train_one(&mut self, x: &[u8], y: usize) -> f32 {
        let out = self.scores(x);
        // target one-hot
        let mut err = vec![0.0f32; self.n_classes];
        let mut se = 0.0f32;
        for k in 0..self.n_classes {
            let t = if k == y { 1.0 } else { 0.0 };
            err[k] = t - out[k];
            se += err[k] * err[k];
        }
        let step = self.eta / self.c as f32;
        for t in 0..self.c {
            let a = self.address(x, t);
            let base = a * self.n_classes;
            for k in 0..self.n_classes {
                self.weights[t][base + k] += step * err[k];
            }
        }
        se
    }

    fn accuracy(&self, data: &[Sample]) -> f64 {
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

    fn mean_sq_error(&self, data: &[Sample]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut se = 0.0f64;
        for s in data {
            let out = self.scores(&s.x);
            for k in 0..self.n_classes {
                let t = if k == s.y { 1.0 } else { 0.0 };
                let e = t - out[k];
                se += (e * e) as f64;
            }
        }
        se / data.len() as f64
    }
}

// --- MNIST IDX (binarized @ 0.5), 3-class subset ---

fn read_idx_images(path: &Path) -> Vec<Vec<u8>> {
    let mut r = BufReader::new(File::open(path).expect("open images"));
    let mut header = [0u8; 16];
    r.read_exact(&mut header).unwrap();
    let magic = u32::from_be_bytes(header[0..4].try_into().unwrap());
    assert_eq!(magic, 2051);
    let n = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(header[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(header[12..16].try_into().unwrap()) as usize;
    assert_eq!(rows * cols, N_FEATURES);
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0u8; rows * cols];
    for _ in 0..n {
        r.read_exact(&mut buf).unwrap();
        out.push(buf.clone());
    }
    out
}

fn read_idx_labels(path: &Path) -> Vec<u8> {
    let mut r = BufReader::new(File::open(path).expect("open labels"));
    let mut header = [0u8; 8];
    r.read_exact(&mut header).unwrap();
    let magic = u32::from_be_bytes(header[0..4].try_into().unwrap());
    assert_eq!(magic, 2049);
    let n = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;
    let mut lbl = vec![0u8; n];
    r.read_exact(&mut lbl).unwrap();
    lbl
}

fn load_binarized(img: &Path, lbl: &Path) -> Vec<Sample> {
    let imgs = read_idx_images(img);
    let lbls = read_idx_labels(lbl);
    imgs.into_iter()
        .zip(lbls)
        .map(|(px, y)| Sample {
            x: px.into_iter().map(|p| if p >= 128 { 1u8 } else { 0u8 }).collect(),
            y: y as usize,
        })
        .collect()
}

fn subset_classes(data: &[Sample], classes: &[usize], per_class: usize, seed: u64) -> Vec<Sample> {
    let mut by: Vec<Vec<&Sample>> = vec![Vec::new(); classes.len()];
    for s in data {
        if let Some(pos) = classes.iter().position(|&c| c == s.y) {
            by[pos].push(s);
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::new();
    for (i, bucket) in by.iter_mut().enumerate() {
        bucket.shuffle(&mut rng);
        let take = per_class.min(bucket.len());
        assert!(take > 0, "no samples for class {}", classes[i]);
        for s in bucket.iter().take(take) {
            // Remap label to 0..n_classes-1
            out.push(Sample {
                x: s.x.clone(),
                y: i,
            });
        }
    }
    out.shuffle(&mut rng);
    out
}

fn main() {
    // Locate MNIST next to ramnet-study (sibling under NN-Revival).
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mnist = manifest
        .join("../../ramnet-study/data/mnist")
        .canonicalize()
        .unwrap_or_else(|_| manifest.join("../../ramnet-study/data/mnist"));
    let train_img = mnist.join("train-images-idx3-ubyte");
    let train_lbl = mnist.join("train-labels-idx1-ubyte");
    let test_img = mnist.join("t10k-images-idx3-ubyte");
    let test_lbl = mnist.join("t10k-labels-idx1-ubyte");
    assert!(train_img.exists(), "missing {}", train_img.display());

    println!("=== hashing-CMAC spike ===");
    println!(
        "knobs: C={C}  bits/tile={BITS_PER_TILE}  table_size={TABLE_SIZE}  η={ETA}  epochs={EPOCHS}"
    );
    println!(
        "task:  classes={{0,1,2}}  train/class={TRAIN_PER_CLASS}  test/class={TEST_PER_CLASS}  seed={SEED}"
    );
    println!("data:  {}", mnist.display());

    let train_full = load_binarized(&train_img, &train_lbl);
    let test_full = load_binarized(&test_img, &test_lbl);
    let classes = [0usize, 1, 2];
    let train = subset_classes(&train_full, &classes, TRAIN_PER_CLASS, SEED ^ 0x71A1_0001);
    let test = subset_classes(&test_full, &classes, TEST_PER_CLASS, SEED ^ 0x7E57_0001);
    println!(
        "loaded: train={}  test={}  (remapped labels 0..{})",
        train.len(),
        test.len(),
        N_CLASSES - 1
    );

    let mut model = HashCmac::new(
        N_FEATURES,
        N_CLASSES,
        C,
        BITS_PER_TILE,
        TABLE_SIZE,
        ETA,
        SEED,
    );

    // Collision diagnostic: how many distinct addresses hit on the train set?
    {
        let mut seen = vec![std::collections::HashSet::new(); C];
        for s in &train {
            for t in 0..C {
                seen[t].insert(model.address(&s.x, t));
            }
        }
        let mean_occ: f64 = seen.iter().map(|h| h.len() as f64).sum::<f64>() / C as f64;
        println!(
            "addr occupancy (train): mean distinct/tiling = {:.1} / {TABLE_SIZE}  ({:.1}% full)",
            mean_occ,
            100.0 * mean_occ / TABLE_SIZE as f64
        );
    }

    let chance = 1.0 / N_CLASSES as f64;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x5A0F_0001);
    println!();
    println!("epoch | train_acc | test_acc | train_MSE | test_MSE");
    println!("------+-----------+----------+-----------+---------");

    // epoch 0 = before any learning
    {
        let ta = model.accuracy(&train);
        let te = model.accuracy(&test);
        let tm = model.mean_sq_error(&train);
        let em = model.mean_sq_error(&test);
        println!(
            "{:>5} | {:>9.4} | {:>8.4} | {:>9.4} | {:>7.4}",
            0, ta, te, tm, em
        );
    }

    let mut order: Vec<usize> = (0..train.len()).collect();
    for ep in 1..=EPOCHS {
        order.shuffle(&mut rng);
        let mut se_sum = 0.0f64;
        for &i in &order {
            se_sum += model.train_one(&train[i].x, train[i].y) as f64;
        }
        let ta = model.accuracy(&train);
        let te = model.accuracy(&test);
        let tm = se_sum / train.len() as f64;
        let em = model.mean_sq_error(&test);
        println!(
            "{:>5} | {:>9.4} | {:>8.4} | {:>9.4} | {:>7.4}",
            ep, ta, te, tm, em
        );
    }

    let final_test = model.accuracy(&test);
    println!();
    println!("=== verdict ===");
    println!("chance = {:.4}", chance);
    println!("final test_acc = {:.4}", final_test);
    if final_test > chance + 0.05 {
        println!("LEARNED: test accuracy beats chance by >5pp.");
    } else if final_test > chance {
        println!("WEAK: above chance but marginal.");
    } else {
        println!("FAILED: at or below chance.");
    }
    // silence unused import warning path
    let _ = rng.gen::<u32>();
}
