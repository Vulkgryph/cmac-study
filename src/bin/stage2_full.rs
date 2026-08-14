//! Stage 2 multi-seed native anchors (3 seeds).
//! Pins n∈{50,200,1000,4000} crossover; honest param cost; freezes results.

use cmac_study::cmac::{probe_local_generalization, TilingCmac};
use cmac_study::metrics::{NativeRecord, Summary};
use cmac_study::mlp_cont::MlpCont;
use cmac_study::tasks::{rmse_of, sample_fn_approx, sample_ik};
use cmac_study::train::{fit_cmac_early_stop, time_update_us};
use std::fs;
use std::path::PathBuf;

const SEEDS: &[u64] = &[0, 1, 2];
const C: usize = 32;
const TILE_W: f64 = 0.05;
const TABLE: usize = 8192;
const ETA: f64 = 0.35;
const MLP_HIDDEN: usize = 64;
const MLP_LR: f64 = 0.05;
const MAX_EPOCHS: usize = 80;
const PATIENCE: usize = 10;
const N_VAL: usize = 1000;
const N_TEST: usize = 2000;
const N_TRAINS: &[usize] = &[50, 200, 1000, 4000];

fn run_task(
    task: &str,
    seed: u64,
    n_out: usize,
    sample: impl Fn(usize, u64) -> Vec<cmac_study::tasks::ContSample>,
    records: &mut Vec<NativeRecord>,
) {
    // Fixed val/test per seed (not per n_train) so the k-curve is clean.
    let val = sample(N_VAL, seed ^ 0xBA11_0001);
    let test = sample(N_TEST, seed ^ 0x7E57_0001);

    for &n_train in N_TRAINS {
        let train = sample(n_train, seed ^ 0x71A1_0001 ^ (n_train as u64 * 13));

        // CMAC
        {
            let mut cmac = TilingCmac::unit_cube(2, n_out, C, TILE_W, TABLE, ETA);
            let info = cmac.describe();
            let trace = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, seed);
            let test_rmse = rmse_of(&test, |x| cmac.predict(x));
            let mut timed = TilingCmac::unit_cube(2, n_out, C, TILE_W, TABLE, ETA);
            let upd = time_update_us(&mut timed, &train[..train.len().min(200)], 1);
            assert_eq!(cmac.active_cells_per_example(), C);
            println!(
                "  [{task}] seed={seed} cmac n={n_train:>5} test={test_rmse:.5} val={:.5} ep={} ms={:.0} act={C} params={}",
                trace.best_val_rmse, trace.best_epoch, trace.train_ms, info.trainable_params
            );
            records.push(NativeRecord {
                question: "Q1".into(),
                task: task.into(),
                arm: "cmac".into(),
                seed,
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

        // MLP
        {
            let mut mlp = MlpCont::new(2, MLP_HIDDEN, n_out, MLP_LR, seed);
            let trace = mlp.fit_early_stop(&train, &val, MAX_EPOCHS, PATIENCE, seed);
            let test_rmse = rmse_of(&test, |x| mlp.predict(x));
            println!(
                "  [{task}] seed={seed} mlp  n={n_train:>5} test={test_rmse:.5} val={:.5} ep={} ms={:.0} params={}",
                trace.best_val_rmse,
                trace.best_epoch,
                trace.train_ms,
                mlp.trainable_params()
            );
            records.push(NativeRecord {
                question: "Q1".into(),
                task: task.into(),
                arm: "mlp".into(),
                seed,
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
                notes: format!("mlp_cont(h={MLP_HIDDEN})"),
            });
        }
    }
}

fn main() {
    println!("=== Stage 2 FULL multi-seed native ===");
    println!("seeds={SEEDS:?} C={C} tile_w={TILE_W} table={TABLE} η={ETA}");
    println!("n_train={N_TRAINS:?}  early-stop max={MAX_EPOCHS} patience={PATIENCE}");
    println!("N2 framing: crossover expected — MLP wins ultra-low-n; CMAC after coverage.\n");

    let mut records = Vec::new();
    for &seed in SEEDS {
        println!("---- seed {seed} ----");
        run_task("fn_approx", seed, 1, sample_fn_approx, &mut records);
        run_task("ik", seed, 2, sample_ik, &mut records);
    }

    for r in &records {
        if r.arm == "cmac" {
            assert_eq!(r.active_cells, C);
        }
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("results");
    let frozen = out.join("stage2_full_frozen");
    fs::create_dir_all(&frozen).unwrap();

    let summary = Summary {
        mode: "stage2-full".into(),
        seeds: SEEDS.to_vec(),
        records,
    };
    summary.write(&out, "stage2_full_summary").unwrap();
    summary.write(&frozen, "summary").unwrap();

    // Fold local-gen probe into frozen set
    let cmac = TilingCmac::unit_cube(2, 1, 16, 0.10, 4096, 0.0);
    let w = 0.10;
    let distances = [
        0.0,
        0.1 * w,
        0.25 * w,
        0.5 * w,
        0.9 * w,
        1.0 * w,
        1.5 * w,
        2.0 * w,
        3.0 * w,
        5.0 * w,
        8.0 * w,
    ];
    let probe = probe_local_generalization(&cmac, &distances, 200, 0);
    let probe_path = frozen.join("local_gen_probe.json");
    fs::write(&probe_path, serde_json::to_string_pretty(&probe).unwrap()).unwrap();
    println!("wrote {}", probe_path.display());

    println!("\n=== Stage 2 multi-seed done ===");
    println!("records: {}", summary.records.len());
    println!("frozen dir: {}", frozen.display());
}
