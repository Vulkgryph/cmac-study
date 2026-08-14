//! Stage 2 — native faithful anchors, 1-seed sanity.
//!
//! Tasks:
//!   - fn-approx: 2-D nonlinear surface on [0,1]² → scalar
//!   - ik: open-loop 2-link inverse kinematics → (θ1, θ2)
//!
//! Arms: tiling CMAC (faithful) + small plain-Rust MLP baseline.
//! Protocol: early-stop on held-out val RMSE; report test RMSE at best-val.
//! Sample-efficiency: n_train ∈ {50, 200, 1000, 4000}.
//!
//! STOP after this 1-seed pass — do not launch multi-seed until approved.

use cmac_study::cmac::TilingCmac;
use cmac_study::metrics::{NativeRecord, Summary};
use cmac_study::mlp_cont::MlpCont;
use cmac_study::tasks::{rmse_of, sample_fn_approx, sample_ik};
use cmac_study::train::{fit_cmac_early_stop, time_update_us};
use std::path::PathBuf;

const SEED: u64 = 0;

// Natural CMAC budget for native 2-D tasks.
const C: usize = 32;
const TILE_W: f64 = 0.05;
const TABLE: usize = 8192;
const ETA: f64 = 0.35;

// MLP baseline.
const MLP_HIDDEN: usize = 64;
const MLP_LR: f64 = 0.05;

const MAX_EPOCHS: usize = 80;
const PATIENCE: usize = 10;
const N_VAL: usize = 1000;
const N_TEST: usize = 2000;
const N_TRAINS: &[usize] = &[50, 200, 1000, 4000];

fn run_fn_approx(records: &mut Vec<NativeRecord>) {
    println!("\n======== TASK: fn_approx ========");
    let val = sample_fn_approx(N_VAL, SEED ^ 0xBA11_0001);
    let test = sample_fn_approx(N_TEST, SEED ^ 0x7E57_0001);

    // Target scale diagnostic
    let mut mean = 0.0;
    for s in &test {
        mean += s.y[0].abs();
    }
    mean /= test.len() as f64;
    println!("target |f| mean on test ≈ {:.4}  (RMSE scale reference)", mean);

    for &n_train in N_TRAINS {
        let train = sample_fn_approx(n_train, SEED ^ 0x71A1_0001 ^ (n_train as u64));

        // --- CMAC ---
        {
            let mut cmac = TilingCmac::unit_cube(2, 1, C, TILE_W, TABLE, ETA);
            let info = cmac.describe();
            let trace = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, SEED);
            let test_rmse = rmse_of(&test, |x| cmac.predict(x));
            // timing on a fresh copy so we don't wreck weights much — measure before restore isn't needed
            let mut timed = TilingCmac::unit_cube(2, 1, C, TILE_W, TABLE, ETA);
            let upd = time_update_us(&mut timed, &train[..train.len().min(200)], 2);
            assert_eq!(cmac.active_cells_per_example(), C);

            println!(
                "  cmac  n={:>5}  test_rmse={:.5}  val_best={:.5}  ep={}  ms={:.0}  upd_us={:.2}  active={}  params={}",
                n_train,
                test_rmse,
                trace.best_val_rmse,
                trace.best_epoch,
                trace.train_ms,
                upd,
                C,
                info.trainable_params
            );
            // print short curve head/tail
            if let Some(first) = trace.history.first() {
                println!(
                    "         curve: ep1 train={:.4} val={:.4} ... ep{} train={:.4} val={:.4}",
                    first.1,
                    first.2,
                    trace.history.last().unwrap().0,
                    trace.history.last().unwrap().1,
                    trace.history.last().unwrap().2
                );
            }

            records.push(NativeRecord {
                question: "Q1".into(),
                task: "fn_approx".into(),
                arm: "cmac".into(),
                seed: SEED,
                n_train,
                test_rmse,
                val_rmse: trace.best_val_rmse,
                best_epoch: trace.best_epoch,
                train_ms: trace.train_ms,
                update_us: Some(upd),
                active_cells: C,
                trainable_params: info.trainable_params,
                c: Some(C),
                tile_width: Some(TILE_W),
                table_size: Some(TABLE),
                eta: Some(ETA),
                notes: info.name,
            });
        }

        // --- MLP ---
        {
            let mut mlp = MlpCont::new(2, MLP_HIDDEN, 1, MLP_LR, SEED);
            let trace = mlp.fit_early_stop(&train, &val, MAX_EPOCHS, PATIENCE, SEED);
            let test_rmse = rmse_of(&test, |x| mlp.predict(x));
            println!(
                "  mlp   n={:>5}  test_rmse={:.5}  val_best={:.5}  ep={}  ms={:.0}  params={}",
                n_train,
                test_rmse,
                trace.best_val_rmse,
                trace.best_epoch,
                trace.train_ms,
                mlp.trainable_params()
            );
            records.push(NativeRecord {
                question: "Q1".into(),
                task: "fn_approx".into(),
                arm: "mlp".into(),
                seed: SEED,
                n_train,
                test_rmse,
                val_rmse: trace.best_val_rmse,
                best_epoch: trace.best_epoch,
                train_ms: trace.train_ms,
                update_us: None,
                active_cells: MLP_HIDDEN, // dense: all hidden units fire
                trainable_params: mlp.trainable_params(),
                c: None,
                tile_width: None,
                table_size: None,
                eta: Some(MLP_LR),
                notes: format!("mlp_cont(h={MLP_HIDDEN}, lr={MLP_LR})"),
            });
        }
    }
}

fn run_ik(records: &mut Vec<NativeRecord>) {
    println!("\n======== TASK: ik (open-loop 2-link) ========");
    let val = sample_ik(N_VAL, SEED ^ 0xBA11_0002);
    let test = sample_ik(N_TEST, SEED ^ 0x7E57_0002);

    for &n_train in N_TRAINS {
        let train = sample_ik(n_train, SEED ^ 0x71A1_0002 ^ (n_train as u64 * 17));

        // --- CMAC ---
        {
            let mut cmac = TilingCmac::unit_cube(2, 2, C, TILE_W, TABLE, ETA);
            let info = cmac.describe();
            let trace = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, SEED);
            let test_rmse = rmse_of(&test, |x| cmac.predict(x));
            let mut timed = TilingCmac::unit_cube(2, 2, C, TILE_W, TABLE, ETA);
            let upd = time_update_us(&mut timed, &train[..train.len().min(200)], 2);
            assert_eq!(cmac.active_cells_per_example(), C);

            println!(
                "  cmac  n={:>5}  test_rmse={:.5}  val_best={:.5}  ep={}  ms={:.0}  upd_us={:.2}  active={}  params={}",
                n_train,
                test_rmse,
                trace.best_val_rmse,
                trace.best_epoch,
                trace.train_ms,
                upd,
                C,
                info.trainable_params
            );

            records.push(NativeRecord {
                question: "Q1".into(),
                task: "ik".into(),
                arm: "cmac".into(),
                seed: SEED,
                n_train,
                test_rmse,
                val_rmse: trace.best_val_rmse,
                best_epoch: trace.best_epoch,
                train_ms: trace.train_ms,
                update_us: Some(upd),
                active_cells: C,
                trainable_params: info.trainable_params,
                c: Some(C),
                tile_width: Some(TILE_W),
                table_size: Some(TABLE),
                eta: Some(ETA),
                notes: info.name,
            });
        }

        // --- MLP ---
        {
            let mut mlp = MlpCont::new(2, MLP_HIDDEN, 2, MLP_LR, SEED);
            let trace = mlp.fit_early_stop(&train, &val, MAX_EPOCHS, PATIENCE, SEED);
            let test_rmse = rmse_of(&test, |x| mlp.predict(x));
            println!(
                "  mlp   n={:>5}  test_rmse={:.5}  val_best={:.5}  ep={}  ms={:.0}  params={}",
                n_train,
                test_rmse,
                trace.best_val_rmse,
                trace.best_epoch,
                trace.train_ms,
                mlp.trainable_params()
            );
            records.push(NativeRecord {
                question: "Q1".into(),
                task: "ik".into(),
                arm: "mlp".into(),
                seed: SEED,
                n_train,
                test_rmse,
                val_rmse: trace.best_val_rmse,
                best_epoch: trace.best_epoch,
                train_ms: trace.train_ms,
                update_us: None,
                active_cells: MLP_HIDDEN,
                trainable_params: mlp.trainable_params(),
                c: None,
                tile_width: None,
                table_size: None,
                eta: Some(MLP_LR),
                notes: format!("mlp_cont(h={MLP_HIDDEN}, lr={MLP_LR})"),
            });
        }
    }
}

fn main() {
    println!("=== Stage 2 native anchors — 1-seed sanity (seed={SEED}) ===");
    println!(
        "CMAC natural budget: C={C} tile_w={TILE_W} table={TABLE} η={ETA}"
    );
    println!("MLP baseline: hidden={MLP_HIDDEN} lr={MLP_LR}");
    println!("early-stop: max_epochs={MAX_EPOCHS} patience={PATIENCE}");
    println!("n_train grid: {:?}", N_TRAINS);

    let mut records = Vec::new();
    run_fn_approx(&mut records);
    run_ik(&mut records);

    // Invariants
    for r in &records {
        if r.arm == "cmac" {
            assert_eq!(r.active_cells, C, "N5: active cells must equal C");
        }
    }

    let summary = Summary {
        mode: "stage2-smoke-seed0".into(),
        seeds: vec![SEED],
        records,
    };
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results");
    summary.write(&out, "stage2_summary").expect("write results");

    println!("\n=== Stage 2 1-seed done ===");
    println!("records: {}", summary.records.len());
    println!("Review results/stage2_summary.md before approving multi-seed / Stage 3.");
}
