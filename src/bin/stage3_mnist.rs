//! Stage 3 — MNIST comparability (hashing-CMAC vs wisard vs mlp).
//! Hashing-CMAC is labeled out-of-domain. 3 seeds, k-curve.

use cmac_study::hash_cmac::{BinarySample, HashCmac};
use cmac_study::metrics::{MnistRecord, MnistSummary};
use cmac_study::mnist_arms::{MlpClass, WisardPlain};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SEEDS: &[u64] = &[0, 1, 2];
const N_FEATURES: usize = 784;
const N_CLASSES: usize = 10;

// Hashing-CMAC natural budget (from spike, scaled for 10-class)
const HC_C: usize = 64;
const HC_BITS: usize = 16;
const HC_TABLE: usize = 8192;
const HC_ETA: f32 = 0.15;
const HC_EPOCHS: usize = 15;

// WiSARD
const WIS_N: usize = 1000;
const WIS_BITS: usize = 8;

// MLP
const MLP_H: usize = 64;
const MLP_LR: f64 = 0.01;
const MLP_EPOCHS: usize = 15;

const LOW_K_DRAWS: usize = 5;
const K_VALUES: &[Option<usize>] = &[
    Some(1),
    Some(5),
    Some(10),
    Some(50),
    Some(100),
    None,
];

fn read_idx_images(path: &Path) -> Vec<Vec<u8>> {
    let mut r = BufReader::new(File::open(path).expect("images"));
    let mut header = [0u8; 16];
    r.read_exact(&mut header).unwrap();
    assert_eq!(u32::from_be_bytes(header[0..4].try_into().unwrap()), 2051);
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
    let mut r = BufReader::new(File::open(path).expect("labels"));
    let mut header = [0u8; 8];
    r.read_exact(&mut header).unwrap();
    assert_eq!(u32::from_be_bytes(header[0..4].try_into().unwrap()), 2049);
    let n = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;
    let mut lbl = vec![0u8; n];
    r.read_exact(&mut lbl).unwrap();
    lbl
}

fn load_binarized(img: &Path, lbl: &Path) -> Vec<BinarySample> {
    let imgs = read_idx_images(img);
    let lbls = read_idx_labels(lbl);
    imgs.into_iter()
        .zip(lbls)
        .map(|(px, y)| BinarySample {
            x: px.into_iter().map(|p| if p >= 128 { 1u8 } else { 0u8 }).collect(),
            y: y as usize,
        })
        .collect()
}

fn subsample_per_class(
    data: &[BinarySample],
    k: Option<usize>,
    seed: u64,
) -> Vec<BinarySample> {
    let mut by: Vec<Vec<&BinarySample>> = vec![Vec::new(); N_CLASSES];
    for s in data {
        if s.y < N_CLASSES {
            by[s.y].push(s);
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC0FF_EE01);
    let mut out = Vec::new();
    for bucket in by.iter_mut() {
        bucket.shuffle(&mut rng);
        let take = match k {
            Some(kk) => kk.min(bucket.len()),
            None => bucket.len(),
        };
        for s in bucket.iter().take(take) {
            out.push((*s).clone());
        }
    }
    out.shuffle(&mut rng);
    out
}

fn n_draws(k: Option<usize>) -> usize {
    match k {
        Some(1) | Some(5) => LOW_K_DRAWS,
        _ => 1,
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mnist = root
        .join("../ramnet-study/data/mnist")
        .canonicalize()
        .expect("mnist path");
    let train_full = load_binarized(
        &mnist.join("train-images-idx3-ubyte"),
        &mnist.join("train-labels-idx1-ubyte"),
    );
    let test = load_binarized(
        &mnist.join("t10k-images-idx3-ubyte"),
        &mnist.join("t10k-labels-idx1-ubyte"),
    );
    assert_eq!(train_full.len(), 60_000);
    assert_eq!(test.len(), 10_000);
    println!("=== Stage 3 MNIST (hashing-CMAC comparability) ===");
    println!("data OK train={} test={}", train_full.len(), test.len());
    println!(
        "hash_cmac: C={HC_C} bits={HC_BITS} table={HC_TABLE} η={HC_ETA} ep={HC_EPOCHS}"
    );
    println!("wisard: N={WIS_N} n={WIS_BITS}");
    println!("mlp: h={MLP_H} lr={MLP_LR} ep={MLP_EPOCHS}");
    println!("seeds={SEEDS:?} low_k_draws={LOW_K_DRAWS}");

    let mut records = Vec::new();

    for &seed in SEEDS {
        for &k in K_VALUES {
            let draws = n_draws(k);
            let k_label = k.map(|v| v.to_string()).unwrap_or_else(|| "full".into());
            println!("-- seed={seed} k={k_label} draws={draws} --");

            // hash_cmac
            {
                let mut accs = Vec::new();
                let mut mss = Vec::new();
                for d in 0..draws {
                    let draw_seed = seed
                        .wrapping_mul(1_000_003)
                        .wrapping_add(d as u64)
                        .wrapping_add(0xD4A7);
                    let train = subsample_per_class(&train_full, k, draw_seed);
                    let mut m = HashCmac::new(
                        N_FEATURES,
                        N_CLASSES,
                        HC_C,
                        HC_BITS,
                        HC_TABLE,
                        HC_ETA,
                        HC_EPOCHS,
                        seed.wrapping_add(d as u64 * 17),
                    );
                    let t0 = Instant::now();
                    m.fit(&train, seed.wrapping_add(d as u64));
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let acc = m.accuracy(&test);
                    accs.push(acc);
                    mss.push(ms);
                }
                let acc = accs.iter().sum::<f64>() / accs.len() as f64;
                let ms = mss.iter().sum::<f64>() / mss.len() as f64;
                println!("  hash_cmac  acc={acc:.4}  ms={ms:.1}  params={}", HC_C * HC_TABLE * N_CLASSES);
                records.push(MnistRecord {
                    question: "Q1".into(),
                    arm: "hash_cmac".into(),
                    seed,
                    k_per_class: k,
                    test_acc: acc,
                    train_ms: ms,
                    active_cells: HC_C,
                    trainable_params: (HC_C * HC_TABLE * N_CLASSES) as u64,
                    c: Some(HC_C),
                    table_size: Some(HC_TABLE),
                    bits_per_tile: Some(HC_BITS),
                    eta: Some(HC_ETA as f64),
                    n_tuples: None,
                    n_bits: None,
                    hidden: None,
                    notes: "hashing-CMAC out-of-domain".into(),
                });
            }

            // wisard
            {
                let mut accs = Vec::new();
                let mut mss = Vec::new();
                for d in 0..draws {
                    let draw_seed = seed
                        .wrapping_mul(1_000_003)
                        .wrapping_add(d as u64)
                        .wrapping_add(0xD4A7);
                    let train = subsample_per_class(&train_full, k, draw_seed);
                    let mut m = WisardPlain::new(
                        N_FEATURES,
                        WIS_BITS,
                        WIS_N,
                        N_CLASSES,
                        seed.wrapping_add(d as u64 * 17),
                    );
                    let t0 = Instant::now();
                    m.fit(&train);
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let acc = m.accuracy(&test);
                    accs.push(acc);
                    mss.push(ms);
                }
                let acc = accs.iter().sum::<f64>() / accs.len() as f64;
                let ms = mss.iter().sum::<f64>() / mss.len() as f64;
                println!("  wisard     acc={acc:.4}  ms={ms:.1}");
                records.push(MnistRecord {
                    question: "Q1".into(),
                    arm: "wisard".into(),
                    seed,
                    k_per_class: k,
                    test_acc: acc,
                    train_ms: ms,
                    active_cells: WIS_N,
                    trainable_params: 0,
                    c: None,
                    table_size: None,
                    bits_per_tile: None,
                    eta: None,
                    n_tuples: Some(WIS_N),
                    n_bits: Some(WIS_BITS),
                    hidden: None,
                    notes: format!("wisard N={WIS_N} n={WIS_BITS}"),
                });
            }

            // mlp
            {
                let mut accs = Vec::new();
                let mut mss = Vec::new();
                for d in 0..draws {
                    let draw_seed = seed
                        .wrapping_mul(1_000_003)
                        .wrapping_add(d as u64)
                        .wrapping_add(0xD4A7);
                    let train = subsample_per_class(&train_full, k, draw_seed);
                    let mut m = MlpClass::new(
                        N_FEATURES,
                        MLP_H,
                        N_CLASSES,
                        MLP_LR,
                        MLP_EPOCHS,
                        seed.wrapping_add(d as u64 * 17),
                    );
                    let t0 = Instant::now();
                    m.fit(&train, seed.wrapping_add(d as u64));
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    let acc = m.accuracy(&test);
                    accs.push(acc);
                    mss.push(ms);
                }
                let acc = accs.iter().sum::<f64>() / accs.len() as f64;
                let ms = mss.iter().sum::<f64>() / mss.len() as f64;
                let params = (MLP_H * N_FEATURES + MLP_H + N_CLASSES * MLP_H + N_CLASSES) as u64;
                println!("  mlp        acc={acc:.4}  ms={ms:.1}  params={params}");
                records.push(MnistRecord {
                    question: "Q1".into(),
                    arm: "mlp".into(),
                    seed,
                    k_per_class: k,
                    test_acc: acc,
                    train_ms: ms,
                    active_cells: MLP_H,
                    trainable_params: params,
                    c: None,
                    table_size: None,
                    bits_per_tile: None,
                    eta: Some(MLP_LR),
                    n_tuples: None,
                    n_bits: None,
                    hidden: Some(MLP_H),
                    notes: format!("mlp h={MLP_H}"),
                });
            }
        }
    }

    let out = root.join("results");
    let frozen = out.join("stage3_mnist_frozen");
    fs::create_dir_all(&frozen).unwrap();
    let summary = MnistSummary {
        mode: "stage3-mnist-full".into(),
        seeds: SEEDS.to_vec(),
        records,
    };
    summary.write(&out, "stage3_mnist_summary").unwrap();
    summary.write(&frozen, "summary").unwrap();
    println!("\n=== Stage 3 done ===");
    println!("records: {}", summary.records.len());
}
