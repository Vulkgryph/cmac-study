//! Run records + aggregation + writers.

use serde::Serialize;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct NativeRecord {
    pub question: String,
    pub task: String,
    pub arm: String,
    pub seed: u64,
    pub n_train: usize,
    pub test_rmse: f64,
    pub val_rmse: f64,
    pub best_epoch: usize,
    pub train_ms: f64,
    pub update_us: Option<f64>,
    pub active_cells: usize,
    pub trainable_params: u64,
    pub c: Option<usize>,
    pub tile_width: Option<f64>,
    pub table_size: Option<usize>,
    pub eta: Option<f64>,
    pub notes: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct MnistRecord {
    pub question: String,
    pub arm: String,
    pub seed: u64,
    pub k_per_class: Option<usize>,
    pub test_acc: f64,
    pub train_ms: f64,
    pub active_cells: usize,
    pub trainable_params: u64,
    pub c: Option<usize>,
    pub table_size: Option<usize>,
    pub bits_per_tile: Option<usize>,
    pub eta: Option<f64>,
    pub n_tuples: Option<usize>,
    pub n_bits: Option<usize>,
    pub hidden: Option<usize>,
    pub notes: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub mode: String,
    pub seeds: Vec<u64>,
    pub records: Vec<NativeRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MnistSummary {
    pub mode: String,
    pub seeds: Vec<u64>,
    pub records: Vec<MnistRecord>,
}

pub fn mean_range(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n as f64;
    if n == 1 {
        return (mean, 0.0);
    }
    let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, (max - min) / 2.0)
}

pub fn fmt_mr(xs: &[f64]) -> String {
    let (m, r) = mean_range(xs);
    if xs.len() <= 1 {
        format!("{:.6}", m)
    } else {
        format!("{:.6}±{:.6}", m, r)
    }
}

impl Summary {
    pub fn write(&self, out_dir: &Path, stem: &str) -> std::io::Result<()> {
        fs::create_dir_all(out_dir)?;
        let json_path = out_dir.join(format!("{stem}.json"));
        let md_path = out_dir.join(format!("{stem}.md"));
        fs::write(&json_path, serde_json::to_string_pretty(self).unwrap())?;
        fs::write(&md_path, self.to_markdown())?;
        println!("wrote {} and {}", json_path.display(), md_path.display());
        Ok(())
    }

    fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!(
            "# cmac-study Stage 2 native results ({})\n\n",
            self.mode
        ));
        md.push_str(&format!("seeds: {:?}\n\n", self.seeds));

        // Aggregated sample-efficiency table with crossover visibility
        md.push_str("## Q1 — sample-efficiency (test RMSE, mean±range over seeds)\n\n");
        md.push_str(
            "| task | arm | n=50 | n=200 | n=1000 | n=4000 | params | active |\n",
        );
        md.push_str(
            "|------|-----|------|-------|--------|--------|--------|--------|\n",
        );

        for task in ["fn_approx", "ik"] {
            for arm in ["cmac", "mlp"] {
                let mut cells = Vec::new();
                let mut params = 0u64;
                let mut active = 0usize;
                for &n in &[50usize, 200, 1000, 4000] {
                    let xs: Vec<f64> = self
                        .records
                        .iter()
                        .filter(|r| r.task == task && r.arm == arm && r.n_train == n)
                        .map(|r| r.test_rmse)
                        .collect();
                    if let Some(r0) = self
                        .records
                        .iter()
                        .find(|r| r.task == task && r.arm == arm && r.n_train == n)
                    {
                        params = r0.trainable_params;
                        active = r0.active_cells;
                    }
                    cells.push(if xs.is_empty() {
                        "—".into()
                    } else {
                        fmt_mr(&xs)
                    });
                }
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    task, arm, cells[0], cells[1], cells[2], cells[3], params, active
                ));
            }
        }

        // Crossover lines
        md.push_str("\n## Crossover (where CMAC test RMSE drops below MLP)\n\n");
        for task in ["fn_approx", "ik"] {
            let ns = [50usize, 200, 1000, 4000];
            let mut first_win: Option<usize> = None;
            let mut detail = Vec::new();
            for &n in &ns {
                let c_xs: Vec<f64> = self
                    .records
                    .iter()
                    .filter(|r| r.task == task && r.arm == "cmac" && r.n_train == n)
                    .map(|r| r.test_rmse)
                    .collect();
                let m_xs: Vec<f64> = self
                    .records
                    .iter()
                    .filter(|r| r.task == task && r.arm == "mlp" && r.n_train == n)
                    .map(|r| r.test_rmse)
                    .collect();
                if c_xs.is_empty() || m_xs.is_empty() {
                    continue;
                }
                let (cm, _) = mean_range(&c_xs);
                let (mm, _) = mean_range(&m_xs);
                let winner = if cm < mm { "cmac" } else { "mlp" };
                detail.push(format!("n={n}: cmac={cm:.4} mlp={mm:.4} → {winner}"));
                if cm < mm && first_win.is_none() {
                    first_win = Some(n);
                }
            }
            md.push_str(&format!(
                "- **{}**: crossover at n≥**{}** (CMAC wins from here). {}\n",
                task,
                first_win
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "none in grid".into()),
                detail.join("; ")
            ));
        }

        md.push_str("\n## Param cost (honest)\n\n");
        md.push_str(
            "CMAC efficiency claim is **compute per example O(C)**, not memory. \
             Table floats ≫ MLP weights (~1000×). active_cells ≡ C for every CMAC row.\n\n",
        );

        md.push_str("## Raw per-seed rows\n\n");
        md.push_str(
            "| task | arm | seed | n_train | test_rmse | val_rmse | best_ep | train_ms | update_us | active | params |\n",
        );
        md.push_str(
            "|------|-----|------|---------|-----------|----------|---------|----------|-----------|--------|--------|\n",
        );
        let mut rows = self.records.clone();
        rows.sort_by(|a, b| {
            (&a.task, &a.arm, a.n_train, a.seed).cmp(&(&b.task, &b.arm, b.n_train, b.seed))
        });
        for r in &rows {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {:.6} | {:.6} | {} | {:.1} | {} | {} | {} |\n",
                r.task,
                r.arm,
                r.seed,
                r.n_train,
                r.test_rmse,
                r.val_rmse,
                r.best_epoch,
                r.train_ms,
                r.update_us
                    .map(|u| format!("{:.2}", u))
                    .unwrap_or_else(|| "—".into()),
                r.active_cells,
                r.trainable_params,
            ));
        }
        md.push_str(
            "\n_mean±range over seeds; range=(max−min)/2. RMSE at best-val early-stop._\n",
        );
        md
    }
}

impl MnistSummary {
    pub fn write(&self, out_dir: &Path, stem: &str) -> std::io::Result<()> {
        fs::create_dir_all(out_dir)?;
        let json_path = out_dir.join(format!("{stem}.json"));
        let md_path = out_dir.join(format!("{stem}.md"));
        fs::write(&json_path, serde_json::to_string_pretty(self).unwrap())?;
        fs::write(&md_path, self.to_markdown())?;
        println!("wrote {} and {}", json_path.display(), md_path.display());
        Ok(())
    }

    fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!(
            "# cmac-study Stage 3 MNIST results ({})\n\n",
            self.mode
        ));
        md.push_str(&format!("seeds: {:?}\n\n", self.seeds));
        md.push_str(
            "**Label:** hashing-CMAC is out-of-domain / CMAC-inspired — not the faithful tiling claim.\n\n",
        );
        md.push_str("## Q1 MNIST k-curve (test acc, mean±range)\n\n");
        md.push_str(
            "| arm | k=1 | k=5 | k=10 | k=50 | k=100 | full | params | active |\n",
        );
        md.push_str(
            "|-----|-----|-----|------|------|-------|------|--------|--------|\n",
        );
        for arm in ["hash_cmac", "wisard", "mlp"] {
            let mut cells = Vec::new();
            let mut params = 0u64;
            let mut active = 0usize;
            for k in &[
                Some(1usize),
                Some(5),
                Some(10),
                Some(50),
                Some(100),
                None,
            ] {
                let xs: Vec<f64> = self
                    .records
                    .iter()
                    .filter(|r| r.arm == arm && r.k_per_class == *k)
                    .map(|r| r.test_acc)
                    .collect();
                if let Some(r0) = self
                    .records
                    .iter()
                    .find(|r| r.arm == arm && r.k_per_class == *k)
                {
                    params = r0.trainable_params;
                    active = r0.active_cells;
                }
                cells.push(if xs.is_empty() {
                    "—".into()
                } else {
                    let (m, r) = mean_range(&xs);
                    if xs.len() <= 1 {
                        format!("{:.4}", m)
                    } else {
                        format!("{:.4}±{:.4}", m, r)
                    }
                });
            }
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                arm, cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], params, active
            ));
        }
        md.push_str(
            "\n_mean±range over seeds. k∈{1,5} averaged over low_k_draws per seed before seed-agg._\n",
        );
        md
    }
}
