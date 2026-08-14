//! N4 control: is the w=0.025 "cliff" coverage (recovers with data) or saturation?
//! Also: C-monotonicity at fixed w — does adding C ever hurt?
//!
//! Freezes into results/stage4_n4_control_frozen/.

use cmac_study::cmac::TilingCmac;
use cmac_study::tasks::{rmse_of, sample_fn_approx};
use cmac_study::train::fit_cmac_early_stop;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

const SEEDS: &[u64] = &[0, 1, 2];
const N_VAL: usize = 1000;
const N_TEST: usize = 2000;
const MAX_EPOCHS: usize = 60;
const PATIENCE: usize = 8;
const ETA: f64 = 0.35;
const TABLE: usize = 16384; // large — collisions out of the picture

#[derive(Clone, Debug, Serialize)]
struct FineWRec {
    question: String, // "N4_control_fine_w"
    seed: u64,
    c: usize,
    tile_width: f64,
    n_train: usize,
    test_rmse: f64,
    val_rmse: f64,
    best_epoch: usize,
    train_ms: f64,
    trainable_params: u64,
    active_cells: usize,
}

#[derive(Clone, Debug, Serialize)]
struct CMonoRec {
    question: String, // "N4_control_c_mono"
    seed: u64,
    c: usize,
    tile_width: f64,
    n_train: usize,
    test_rmse: f64,
    val_rmse: f64,
    best_epoch: usize,
    train_ms: f64,
    trainable_params: u64,
    active_cells: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ControlSummary {
    mode: String,
    seeds: Vec<u64>,
    /// Fixed w=0.025, C=64, sweep n_train.
    fine_w_n_sweep: Vec<FineWRec>,
    /// Fixed w=0.10, n_train=4000, sweep C — never-degrade check.
    c_mono_fixed_w: Vec<CMonoRec>,
    /// Also at w=0.05 for completeness.
    c_mono_w05: Vec<CMonoRec>,
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

fn main() {
    println!("=== N4 control: fine-w × n_train + C-monotonicity ===");
    println!("table={TABLE} (collisions out)  η={ETA}");

    let mut fine = Vec::new();
    let mut mono10 = Vec::new();
    let mut mono05 = Vec::new();

    // ----- Control A: w=0.025, C=64, n_train sweep -----
    // If coverage: RMSE falls toward ~0.01 as n grows.
    // If saturation: stays cliffed.
    let fine_w = 0.025f64;
    let fine_c = 64usize;
    let n_grid = &[500usize, 1000, 2000, 4000, 8000, 16000, 32000];
    println!("\n-------- fine-w coverage control (C={fine_c}, w={fine_w}) --------");
    for &seed in SEEDS {
        let val = sample_fn_approx(N_VAL, seed ^ 0xBA11_0A00);
        let test = sample_fn_approx(N_TEST, seed ^ 0x7E57_0A00);
        for &n_train in n_grid {
            let train = sample_fn_approx(n_train, seed ^ 0x71A1_0A00 ^ (n_train as u64 * 19));
            let mut cmac = TilingCmac::unit_cube(2, 1, fine_c, fine_w, TABLE, ETA);
            let info = cmac.describe();
            let tr = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, seed);
            let te = rmse_of(&test, |x| cmac.predict(x));
            assert_eq!(cmac.active_cells_per_example(), fine_c);
            println!(
                "  seed={seed} n={n_train:>5} test={te:.5} val={:.5} ep={} params={}",
                tr.best_val_rmse, tr.best_epoch, info.trainable_params
            );
            fine.push(FineWRec {
                question: "N4_control_fine_w".into(),
                seed,
                c: fine_c,
                tile_width: fine_w,
                n_train,
                test_rmse: te,
                val_rmse: tr.best_val_rmse,
                best_epoch: tr.best_epoch,
                train_ms: tr.train_ms,
                trainable_params: info.trainable_params,
                active_cells: fine_c,
            });
        }
    }

    // ----- Control B: fixed w, sweep C — does adding C ever hurt? -----
    let n_train_mono = 4000usize;
    let c_grid = &[4usize, 8, 16, 32, 64, 128, 256];

    println!("\n-------- C-monotonicity @ w=0.10, n={n_train_mono} --------");
    for &seed in SEEDS {
        let val = sample_fn_approx(N_VAL, seed ^ 0xBA11_0B10);
        let test = sample_fn_approx(N_TEST, seed ^ 0x7E57_0B10);
        let train = sample_fn_approx(n_train_mono, seed ^ 0x71A1_0B10);
        for &c in c_grid {
            let mut cmac = TilingCmac::unit_cube(2, 1, c, 0.10, TABLE, ETA);
            let info = cmac.describe();
            let tr = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, seed);
            let te = rmse_of(&test, |x| cmac.predict(x));
            assert_eq!(cmac.active_cells_per_example(), c);
            println!(
                "  seed={seed} C={c:>3} w=0.10 test={te:.5} val={:.5} ep={}",
                tr.best_val_rmse, tr.best_epoch
            );
            mono10.push(CMonoRec {
                question: "N4_control_c_mono".into(),
                seed,
                c,
                tile_width: 0.10,
                n_train: n_train_mono,
                test_rmse: te,
                val_rmse: tr.best_val_rmse,
                best_epoch: tr.best_epoch,
                train_ms: tr.train_ms,
                trainable_params: info.trainable_params,
                active_cells: c,
            });
        }
    }

    println!("\n-------- C-monotonicity @ w=0.05, n={n_train_mono} --------");
    for &seed in SEEDS {
        let val = sample_fn_approx(N_VAL, seed ^ 0xBA11_0B05);
        let test = sample_fn_approx(N_TEST, seed ^ 0x7E57_0B05);
        let train = sample_fn_approx(n_train_mono, seed ^ 0x71A1_0B05);
        for &c in c_grid {
            let mut cmac = TilingCmac::unit_cube(2, 1, c, 0.05, TABLE, ETA);
            let info = cmac.describe();
            let tr = fit_cmac_early_stop(&mut cmac, &train, &val, MAX_EPOCHS, PATIENCE, seed);
            let te = rmse_of(&test, |x| cmac.predict(x));
            println!(
                "  seed={seed} C={c:>3} w=0.05 test={te:.5} val={:.5} ep={}",
                tr.best_val_rmse, tr.best_epoch
            );
            mono05.push(CMonoRec {
                question: "N4_control_c_mono".into(),
                seed,
                c,
                tile_width: 0.05,
                n_train: n_train_mono,
                test_rmse: te,
                val_rmse: tr.best_val_rmse,
                best_epoch: tr.best_epoch,
                train_ms: tr.train_ms,
                trainable_params: info.trainable_params,
                active_cells: c,
            });
        }
    }

    let summary = ControlSummary {
        mode: "stage4-n4-control".into(),
        seeds: SEEDS.to_vec(),
        fine_w_n_sweep: fine,
        c_mono_fixed_w: mono10,
        c_mono_w05: mono05,
    };

    // Markdown
    let mut md = String::new();
    md.push_str("# N4 control — coverage vs saturation + C-monotonicity\n\n");
    md.push_str(&format!("seeds: {:?}\n\n", SEEDS));

    md.push_str("## A. Fine-w × n_train (C=64, w=0.025, table=16384)\n\n");
    md.push_str("If cliff is **coverage**: RMSE → ~0.01 as n↑. If **saturation**: stays high.\n\n");
    md.push_str("| n_train | test_rmse |\n|---------|----------|\n");
    let mut ns: Vec<usize> = summary
        .fine_w_n_sweep
        .iter()
        .map(|r| r.n_train)
        .collect();
    ns.sort();
    ns.dedup();
    for n in &ns {
        let xs: Vec<f64> = summary
            .fine_w_n_sweep
            .iter()
            .filter(|r| r.n_train == *n)
            .map(|r| r.test_rmse)
            .collect();
        md.push_str(&format!("| {} | {} |\n", n, fmt(&xs)));
    }

    md.push_str("\n## B. C-monotonicity @ fixed w, n_train=4000\n\n");
    md.push_str("Does adding C ever *raise* test_rmse (hurt)?\n\n");
    for (label, rows) in [
        ("w=0.10", &summary.c_mono_fixed_w),
        ("w=0.05", &summary.c_mono_w05),
    ] {
        md.push_str(&format!("### {}\n\n", label));
        md.push_str("| C | test_rmse | active |\n|---|-----------|--------|\n");
        let mut cs: Vec<usize> = rows.iter().map(|r| r.c).collect();
        cs.sort();
        cs.dedup();
        let mut prev = f64::INFINITY;
        let mut ever_hurt = false;
        for c in &cs {
            let xs: Vec<f64> = rows.iter().filter(|r| r.c == *c).map(|r| r.test_rmse).collect();
            let (m, _) = mean_range(&xs);
            // "hurt" = mean rises by more than noise band ~0.002
            if m > prev + 0.003 {
                ever_hurt = true;
            }
            prev = m;
            md.push_str(&format!("| {} | {} | {} |\n", c, fmt(&xs), c));
        }
        md.push_str(&format!(
            "\n_Adding C ever hurts (Δ>+0.003)? **{}**_\n\n",
            if ever_hurt { "YES" } else { "NO" }
        ));
    }

    // Verdict block
    md.push_str("## Verdict sketch (filled by numbers above)\n\n");
    // compute recovery
    let at_4k: Vec<f64> = summary
        .fine_w_n_sweep
        .iter()
        .filter(|r| r.n_train == 4000)
        .map(|r| r.test_rmse)
        .collect();
    let at_32k: Vec<f64> = summary
        .fine_w_n_sweep
        .iter()
        .filter(|r| r.n_train == 32000)
        .map(|r| r.test_rmse)
        .collect();
    let (m4, _) = mean_range(&at_4k);
    let (m32, _) = mean_range(&at_32k);
    md.push_str(&format!(
        "- fine-w n=4000 RMSE = {:.4}; n=32000 RMSE = {:.4}  → {}\n",
        m4,
        m32,
        if m32 < m4 * 0.5 {
            "RECOVERS with data → **coverage**, not saturation"
        } else if m32 < m4 * 0.85 {
            "partial recovery"
        } else {
            "STAYS CLIFFED → genuine capacity failure"
        }
    ));

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = root.join("results");
    let frozen = out.join("stage4_n4_control_frozen");
    fs::create_dir_all(&frozen).unwrap();
    let json = serde_json::to_string_pretty(&summary).unwrap();
    fs::write(out.join("stage4_n4_control.json"), &json).unwrap();
    fs::write(frozen.join("summary.json"), &json).unwrap();
    fs::write(out.join("stage4_n4_control.md"), &md).unwrap();
    fs::write(frozen.join("summary.md"), &md).unwrap();
    print!("{}", md);
    println!("=== N4 control done → {} ===", frozen.display());
}
