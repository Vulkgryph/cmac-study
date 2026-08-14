//! Stage 4 — Q2 capacity, Q3 collisions, Q4 online adaptation.
//! Faithful tiling CMAC on native fn-approx. Freezes results.

use cmac_study::cmac::TilingCmac;
use cmac_study::mlp_cont::MlpCont;
use cmac_study::tasks::{rmse_of, sample_fn_approx, ContSample};
use cmac_study::train::fit_cmac_early_stop;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

const SEEDS: &[u64] = &[0, 1, 2];
const N_VAL: usize = 1000;
const N_TEST: usize = 2000;
const N_TRAIN: usize = 4000; // covered regime (post-crossover)
const MAX_EPOCHS: usize = 60;
const PATIENCE: usize = 8;
const ETA: f64 = 0.35;

#[derive(Clone, Debug, Serialize)]
struct CapRec {
    question: String,
    seed: u64,
    c: usize,
    tile_width: f64,
    table_size: usize,
    test_rmse: f64,
    val_rmse: f64,
    best_epoch: usize,
    train_ms: f64,
    active_cells: usize,
    trainable_params: u64,
}

#[derive(Clone, Debug, Serialize)]
struct CollRec {
    question: String,
    seed: u64,
    table_size: usize,
    c: usize,
    test_rmse: f64,
    val_rmse: f64,
    best_epoch: usize,
    train_ms: f64,
    trainable_params: u64,
}

#[derive(Clone, Debug, Serialize)]
struct OnlineRec {
    question: String,
    arm: String,
    seed: u64,
    phase: String, // "pre" | "during" | "post"
    window: usize,
    rmse: f64,
    notes: String,
}

#[derive(Clone, Debug, Serialize)]
struct Stage4Summary {
    mode: String,
    seeds: Vec<u64>,
    q2_capacity: Vec<CapRec>,
    q3_collisions: Vec<CollRec>,
    q4_online: Vec<OnlineRec>,
    /// #1 WiSARD N-scaling (cited from frozen baseline; not re-run).
    wisard_n_scaling_cited: Vec<WisardCite>,
}

#[derive(Clone, Debug, Serialize)]
struct WisardCite {
    n: usize,
    acc: f64,
    acc_range: f64,
    source: String,
}

fn mean_range(xs: &[f64]) -> (f64, f64) {
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    if xs.len() == 1 {
        return (m, 0.0);
    }
    let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (m, (hi - lo) / 2.0)
}

fn fmt(xs: &[f64]) -> String {
    let (m, r) = mean_range(xs);
    if xs.len() <= 1 {
        format!("{m:.5}")
    } else {
        format!("{m:.5}±{r:.5}")
    }
}

fn run_q2(records: &mut Vec<CapRec>) {
    println!("\n======== Q2 capacity sweep (fn_approx, n_train={N_TRAIN}) ========");
    // Sweep C and tile_width (resolution); table fixed large so collisions don't confound.
    let grid: &[(usize, f64)] = &[
        (8, 0.10),
        (16, 0.10),
        (32, 0.10),
        (32, 0.05),
        (64, 0.05),
        (64, 0.025),
        (128, 0.025),
    ];
    let table = 16384usize;
    for &seed in SEEDS {
        let val = sample_fn_approx(N_VAL, seed ^ 0xBA11_0200);
        let test = sample_fn_approx(N_TEST, seed ^ 0x7E57_0200);
        let train = sample_fn_approx(N_TRAIN, seed ^ 0x71A1_0200);
        for &(c, tw) in grid {
            let mut cmac = TilingCmac::unit_cube(2, 1, c, tw, table, ETA);
            let info = cmac.describe();
            let tr = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, seed);
            let te = rmse_of(&test, |x| cmac.predict(x));
            assert_eq!(cmac.active_cells_per_example(), c);
            println!(
                "  seed={seed} C={c:>3} w={tw:.3} table={table} test={te:.5} val={:.5} ep={} params={}",
                tr.best_val_rmse, tr.best_epoch, info.trainable_params
            );
            records.push(CapRec {
                question: "Q2".into(),
                seed,
                c,
                tile_width: tw,
                table_size: table,
                test_rmse: te,
                val_rmse: tr.best_val_rmse,
                best_epoch: tr.best_epoch,
                train_ms: tr.train_ms,
                active_cells: c,
                trainable_params: info.trainable_params,
            });
        }
    }
}

fn run_q3(records: &mut Vec<CollRec>) {
    println!("\n======== Q3 collision sweep (shrink table, fixed C=32 w=0.05) ========");
    let c = 32usize;
    let tw = 0.05f64;
    let tables = &[64usize, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    for &seed in SEEDS {
        let val = sample_fn_approx(N_VAL, seed ^ 0xBA11_0300);
        let test = sample_fn_approx(N_TEST, seed ^ 0x7E57_0300);
        let train = sample_fn_approx(N_TRAIN, seed ^ 0x71A1_0300);
        for &table in tables {
            let mut cmac = TilingCmac::unit_cube(2, 1, c, tw, table, ETA);
            let info = cmac.describe();
            let tr = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, seed);
            let te = rmse_of(&test, |x| cmac.predict(x));
            println!(
                "  seed={seed} table={table:>5} test={te:.5} val={:.5} ep={} params={}",
                tr.best_val_rmse, tr.best_epoch, info.trainable_params
            );
            records.push(CollRec {
                question: "Q3".into(),
                seed,
                table_size: table,
                c,
                test_rmse: te,
                val_rmse: tr.best_val_rmse,
                best_epoch: tr.best_epoch,
                train_ms: tr.train_ms,
                trainable_params: info.trainable_params,
            });
        }
    }
}

/// Nonstationary: train on f0, then stream f1 (phase-shifted surface), track online.
fn fn_phase(x: f64, y: f64, phase: f64) -> f64 {
    // Same family as fn_approx_target but phase-shifted sin·cos + moved gaussians.
    let wave = (2.0 * std::f64::consts::PI * x + phase).sin()
        * (2.0 * std::f64::consts::PI * y + phase * 0.7).cos();
    let gaussians = [
        (0.25 + 0.15 * phase.sin(), 0.30, 0.80, 0.08),
        (0.70, 0.65 + 0.10 * phase.cos(), -0.60, 0.10),
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

fn sample_phase(n: usize, seed: u64, phase: f64) -> Vec<ContSample> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xFA5E_0001 ^ (phase.to_bits()));
    (0..n)
        .map(|_| {
            let x = rng.gen_range(0.0..1.0);
            let y = rng.gen_range(0.0..1.0);
            ContSample {
                x: vec![x, y],
                y: vec![fn_phase(x, y, phase)],
            }
        })
        .collect()
}

fn run_q4(records: &mut Vec<OnlineRec>) {
    println!("\n======== Q4 online adaptation (phase shift 0 → π/2) ========");
    let c = 32usize;
    let tw = 0.05f64;
    let table = 8192usize;
    let stream_n = 2000usize;
    let window = 200usize;

    for &seed in SEEDS {
        let pre_train = sample_phase(N_TRAIN, seed ^ 0xA0E0_0001, 0.0);
        let pre_test = sample_phase(N_TEST, seed ^ 0xA0E1_0001, 0.0);
        let stream = sample_phase(stream_n, seed ^ 0x57BE_0001, std::f64::consts::FRAC_PI_2);
        let post_test = sample_phase(N_TEST, seed ^ 0xB057_0001, std::f64::consts::FRAC_PI_2);

        // --- CMAC: fit on phase0, then online local updates on stream ---
        {
            let mut cmac = TilingCmac::unit_cube(2, 1, c, tw, table, ETA);
            let val = sample_phase(N_VAL, seed ^ 0xBA10_0001, 0.0);
            let _ = fit_cmac_early_stop(&mut cmac, &pre_train, &val, MAX_EPOCHS, PATIENCE, seed);
            let pre_rmse = rmse_of(&pre_test, |x| cmac.predict(x));
            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "cmac".into(),
                seed,
                phase: "pre".into(),
                window: 0,
                rmse: pre_rmse,
                notes: "after fit on phase=0; before stream".into(),
            });

            // Stream: one pass, record rolling RMSE on last `window` preds vs targets
            let mut recent_se = 0.0f64;
            let mut recent_n = 0usize;
            let mut mid_rmse = f64::NAN;
            for (i, s) in stream.iter().enumerate() {
                let p = cmac.predict(&s.x);
                let e = p[0] - s.y[0];
                recent_se += e * e;
                recent_n += 1;
                if recent_n > window {
                    // recompute simply every window boundary
                }
                cmac.train_one(&s.x, &s.y); // online local update
                if (i + 1) == stream_n / 2 {
                    // evaluate on a held chunk of upcoming? use recent window of stream itself
                    mid_rmse = (recent_se / recent_n as f64).sqrt();
                }
                if (i + 1) % window == 0 {
                    recent_se = 0.0;
                    recent_n = 0;
                }
            }
            let post_rmse = rmse_of(&post_test, |x| cmac.predict(x));
            // also error if frozen (no online) — measure with a copy
            let mut frozen = TilingCmac::unit_cube(2, 1, c, tw, table, ETA);
            let val2 = sample_phase(N_VAL, seed ^ 0xBA10_0001, 0.0);
            let _ = fit_cmac_early_stop(&mut frozen, &pre_train, &val2, MAX_EPOCHS, PATIENCE, seed);
            let frozen_post = rmse_of(&post_test, |x| frozen.predict(x));

            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "cmac".into(),
                seed,
                phase: "during_mid".into(),
                window,
                rmse: mid_rmse,
                notes: "rolling stream RMSE mid-stream".into(),
            });
            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "cmac".into(),
                seed,
                phase: "post_online".into(),
                window: 0,
                rmse: post_rmse,
                notes: "after one-pass online on phase=π/2".into(),
            });
            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "cmac_frozen".into(),
                seed,
                phase: "post_no_adapt".into(),
                window: 0,
                rmse: frozen_post,
                notes: "same pre-fit, no online updates (baseline)".into(),
            });
            println!(
                "  cmac seed={seed} pre={pre_rmse:.4} mid={mid_rmse:.4} post_online={post_rmse:.4} post_frozen={frozen_post:.4}"
            );
        }

        // --- MLP: fit on phase0; "online" = continue SGD on stream (no replay of phase0) ---
        {
            let mut mlp = MlpCont::new(2, 64, 1, 0.05, seed);
            let val = sample_phase(N_VAL, seed ^ 0xBA10_0001, 0.0);
            let _ = mlp.fit_early_stop(&pre_train, &val, MAX_EPOCHS, PATIENCE, seed);
            let pre_rmse = rmse_of(&pre_test, |x| mlp.predict(x));
            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "mlp".into(),
                seed,
                phase: "pre".into(),
                window: 0,
                rmse: pre_rmse,
                notes: "after fit on phase=0".into(),
            });

            let mut recent_se = 0.0f64;
            let mut recent_n = 0usize;
            let mut mid_rmse = f64::NAN;
            for (i, s) in stream.iter().enumerate() {
                let p = mlp.predict(&s.x);
                let e = p[0] - s.y[0];
                recent_se += e * e;
                recent_n += 1;
                mlp.train_one(&s.x, &s.y);
                if (i + 1) == stream_n / 2 {
                    mid_rmse = (recent_se / recent_n as f64).sqrt();
                }
                if (i + 1) % window == 0 {
                    recent_se = 0.0;
                    recent_n = 0;
                }
            }
            let post_rmse = rmse_of(&post_test, |x| mlp.predict(x));
            // frozen MLP
            let mut frozen = MlpCont::new(2, 64, 1, 0.05, seed);
            let val2 = sample_phase(N_VAL, seed ^ 0xBA10_0001, 0.0);
            let _ = frozen.fit_early_stop(&pre_train, &val2, MAX_EPOCHS, PATIENCE, seed);
            let frozen_post = rmse_of(&post_test, |x| frozen.predict(x));

            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "mlp".into(),
                seed,
                phase: "during_mid".into(),
                window,
                rmse: mid_rmse,
                notes: "rolling stream RMSE mid-stream".into(),
            });
            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "mlp".into(),
                seed,
                phase: "post_online".into(),
                window: 0,
                rmse: post_rmse,
                notes: "after one-pass SGD on phase=π/2 (no replay)".into(),
            });
            records.push(OnlineRec {
                question: "Q4".into(),
                arm: "mlp_frozen".into(),
                seed,
                phase: "post_no_adapt".into(),
                window: 0,
                rmse: frozen_post,
                notes: "same pre-fit, no online".into(),
            });
            println!(
                "  mlp  seed={seed} pre={pre_rmse:.4} mid={mid_rmse:.4} post_online={post_rmse:.4} post_frozen={frozen_post:.4}"
            );
        }
    }
}

fn to_markdown(s: &Stage4Summary) -> String {
    let mut md = String::new();
    md.push_str(&format!("# cmac-study Stage 4 ({})\n\n", s.mode));
    md.push_str(&format!("seeds: {:?}\n\n", s.seeds));

    md.push_str("## Q2 — capacity sweep (fn_approx, n_train=4000, table=16384)\n\n");
    md.push_str("| C | tile_w | test_rmse | active | params |\n");
    md.push_str("|---|--------|-----------|--------|--------|\n");
    let mut keys: Vec<(usize, i64)> = Vec::new();
    for r in &s.q2_capacity {
        let k = (r.c, (r.tile_width * 1000.0).round() as i64);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys.sort();
    for (c, tw_i) in keys {
        let xs: Vec<f64> = s
            .q2_capacity
            .iter()
            .filter(|r| r.c == c && (r.tile_width * 1000.0).round() as i64 == tw_i)
            .map(|r| r.test_rmse)
            .collect();
        let r0 = s
            .q2_capacity
            .iter()
            .find(|r| r.c == c && (r.tile_width * 1000.0).round() as i64 == tw_i)
            .unwrap();
        md.push_str(&format!(
            "| {} | {:.3} | {} | {} | {} |\n",
            c,
            r0.tile_width,
            fmt(&xs),
            r0.active_cells,
            r0.trainable_params
        ));
    }

    md.push_str("\n### #1 WiSARD N-scaling (cited from baseline_full_frozen — same framing)\n\n");
    md.push_str("| N | acc (mean±range) | source |\n|---|------------------|--------|\n");
    for w in &s.wisard_n_scaling_cited {
        md.push_str(&format!(
            "| {} | {:.4}±{:.4} | {} |\n",
            w.n, w.acc, w.acc_range, w.source
        ));
    }
    md.push_str(
        "\n_N4 test: does CMAC test_rmse soften/plateau as C↑ / w↓, vs WiSARD's N=10k dip?_\n",
    );

    md.push_str("\n## Q3 — collision sweep (C=32, w=0.05, shrink table)\n\n");
    md.push_str("| table_size | test_rmse | params |\n");
    md.push_str("|------------|-----------|--------|\n");
    let mut tabs: Vec<usize> = s.q3_collisions.iter().map(|r| r.table_size).collect();
    tabs.sort();
    tabs.dedup();
    for t in tabs {
        let xs: Vec<f64> = s
            .q3_collisions
            .iter()
            .filter(|r| r.table_size == t)
            .map(|r| r.test_rmse)
            .collect();
        let p = s
            .q3_collisions
            .iter()
            .find(|r| r.table_size == t)
            .unwrap()
            .trainable_params;
        md.push_str(&format!("| {} | {} | {} |\n", t, fmt(&xs), p));
    }

    md.push_str("\n## Q4 — online adaptation (phase 0 → π/2)\n\n");
    md.push_str("| arm | pre (phase0) | mid-stream | post_online (phase π/2) | post_frozen |\n");
    md.push_str("|-----|--------------|------------|-------------------------|-------------|\n");
    for arm_base in ["cmac", "mlp"] {
        let pre: Vec<f64> = s
            .q4_online
            .iter()
            .filter(|r| r.arm == arm_base && r.phase == "pre")
            .map(|r| r.rmse)
            .collect();
        let mid: Vec<f64> = s
            .q4_online
            .iter()
            .filter(|r| r.arm == arm_base && r.phase == "during_mid")
            .map(|r| r.rmse)
            .collect();
        let post: Vec<f64> = s
            .q4_online
            .iter()
            .filter(|r| r.arm == arm_base && r.phase == "post_online")
            .map(|r| r.rmse)
            .collect();
        let froz_arm = format!("{arm_base}_frozen");
        let frozen: Vec<f64> = s
            .q4_online
            .iter()
            .filter(|r| r.arm == froz_arm && r.phase == "post_no_adapt")
            .map(|r| r.rmse)
            .collect();
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            arm_base,
            fmt(&pre),
            fmt(&mid),
            fmt(&post),
            fmt(&frozen)
        ));
    }
    md.push_str(
        "\n_N3: online local updates should cut post RMSE vs frozen. MLP continues SGD without replay._\n",
    );
    md
}

fn main() {
    println!("=== Stage 4 sweeps ===");
    let mut q2 = Vec::new();
    let mut q3 = Vec::new();
    let mut q4 = Vec::new();
    run_q2(&mut q2);
    run_q3(&mut q3);
    run_q4(&mut q4);

    // Cite #1 WiSARD N-scaling from frozen baseline (verbatim).
    let wisard = vec![
        WisardCite {
            n: 100,
            acc: 0.8477,
            acc_range: 0.0064,
            source: "ramnet-study/results/baseline_full_frozen".into(),
        },
        WisardCite {
            n: 500,
            acc: 0.8713,
            acc_range: 0.0013,
            source: "ramnet-study/results/baseline_full_frozen".into(),
        },
        WisardCite {
            n: 1000,
            acc: 0.8769,
            acc_range: 0.0002,
            source: "ramnet-study/results/baseline_full_frozen".into(),
        },
        WisardCite {
            n: 5000,
            acc: 0.8759,
            acc_range: 0.0039,
            source: "ramnet-study/results/baseline_full_frozen".into(),
        },
        WisardCite {
            n: 10000,
            acc: 0.8474,
            acc_range: 0.0051,
            source: "ramnet-study/results/baseline_full_frozen".into(),
        },
    ];

    let summary = Stage4Summary {
        mode: "stage4-full".into(),
        seeds: SEEDS.to_vec(),
        q2_capacity: q2,
        q3_collisions: q3,
        q4_online: q4,
        wisard_n_scaling_cited: wisard,
    };

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("results");
    let frozen = out.join("stage4_full_frozen");
    fs::create_dir_all(&frozen).unwrap();
    let json = serde_json::to_string_pretty(&summary).unwrap();
    fs::write(out.join("stage4_summary.json"), &json).unwrap();
    fs::write(frozen.join("summary.json"), &json).unwrap();
    let md = to_markdown(&summary);
    fs::write(out.join("stage4_summary.md"), &md).unwrap();
    fs::write(frozen.join("summary.md"), &md).unwrap();
    println!("{}", md);
    println!("=== Stage 4 done ===");
}
