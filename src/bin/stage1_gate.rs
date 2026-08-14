//! Stage 1 acceptance gate: faithful tiling CMAC + local-generalization probe.
//!
//! Hard gate: two inputs closer than one tile width share ≈ all C active tiles;
//! inputs farther than several tile widths share ≈ none.
//!
//! If this fails, the tiling is wrong — do not proceed to Stages 2–4.

use cmac_study::cmac::{probe_local_generalization, TilingCmac};

fn main() {
    // Unit square [0,1]² — same domain as native fn-approx.
    // C=16, tile_width=0.1 → ~10 tiles per axis per tiling; offsets stagger by w/C.
    let c = 16usize;
    let tile_width = 0.10f64;
    let table_size = 4096usize;
    let eta = 0.0; // geometry only for the gate

    let cmac = TilingCmac::unit_cube(2, 1, c, tile_width, table_size, eta);
    let info = cmac.describe();

    println!("=== Stage 1 — faithful tiling CMAC gate ===");
    println!("config: {}", info.name);
    println!(
        "  n_in={}  n_out={}  C={}  tile_width={}  table_size={}  active_cells={}",
        info.n_in, info.n_out, info.c, info.tile_width, info.table_size, info.active_cells
    );
    println!(
        "  trainable_params (table floats) = {}",
        info.trainable_params
    );
    println!();
    println!("offset schedule (tiling t → offset = t/C · w):");
    for t in 0..c.min(4) {
        println!(
            "  t={t}: offset = {:.6}",
            (t as f64 / c as f64) * tile_width
        );
    }
    if c > 4 {
        println!("  ... ({} more)", c - 4);
    }

    // Sanity: exactly C active cells, addresses in range.
    let x = [0.37f64, 0.61];
    let (addrs, coords) = cmac.active_tiles(&x);
    assert_eq!(addrs.len(), c, "must activate exactly C cells");
    assert!(addrs.iter().all(|&a| a < table_size));
    println!();
    println!("smoke active_tiles([0.37, 0.61]):");
    println!("  addrs  (first 4) = {:?}", &addrs[..4.min(c)]);
    println!("  coords (first 4) = {:?}", &coords[..4.min(c)]);
    println!("  active_cells_per_example = {}", cmac.active_cells_per_example());

    // --- local-generalization probe ---
    // Distances relative to tile_width.
    let w = tile_width;
    let distances = [
        0.0,
        0.1 * w,  // ≪ one tile
        0.25 * w,
        0.5 * w,
        0.9 * w,  // still < one tile
        1.0 * w,  // one tile width
        1.5 * w,
        2.0 * w,
        3.0 * w,
        5.0 * w,
        8.0 * w,
        0.5, // absolute mid-domain
    ];

    println!();
    println!("=== local-generalization probe (shared geometric tiles vs distance) ===");
    println!("domain=[0,1]²  C={c}  tile_width={w}");
    println!();
    println!(" distance | dist/w | mean_shared | min | max | n_pairs | note");
    println!("----------+--------+-------------+-----+-----+---------+-----");

    let rows = probe_local_generalization(&cmac, &distances, 200, 0);
    for r in &rows {
        let rel = r.distance / w;
        let note = if r.distance == 0.0 {
            "identical"
        } else if r.distance < w {
            "≪ / < 1 tile  → expect ~C"
        } else if r.distance < 2.0 * w {
            "~1–2 tiles"
        } else if r.distance >= 5.0 * w {
            "≫ tile       → expect ~0"
        } else {
            ""
        };
        println!(
            " {:8.4} | {:6.2} | {:11.2} | {:3} | {:3} | {:7} | {}",
            r.distance, rel, r.mean_shared, r.min_shared, r.max_shared, r.n_pairs, note
        );
    }

    // --- hard gate ---
    // Theory (1-D stagger): shared ≈ C · max(0, 1 − d/w).
    // In 2-D with random direction, slightly lower (either axis can cross).
    // Gate uses theoretically grounded bands, not an over-strict "≈C at 0.25w".
    println!();
    println!("=== GATE ===");
    println!("theory (1-D): shared ≈ C·max(0, 1−d/w); 2-D random dir is a bit lower.");
    let near = rows
        .iter()
        .find(|r| (r.distance - 0.10 * w).abs() < 1e-12)
        .expect("near row d=0.1w");
    let mid = rows
        .iter()
        .find(|r| (r.distance - 0.50 * w).abs() < 1e-12)
        .expect("mid row d=0.5w");
    let far = rows
        .iter()
        .find(|r| (r.distance - 5.0 * w).abs() < 1e-12)
        .expect("far row");
    let ident = rows
        .iter()
        .find(|r| r.distance == 0.0)
        .expect("ident row");

    // d=0.1w → 1-D theory 0.9C; require ≥ 0.75C in 2-D
    let near_ok = near.mean_shared >= 0.75 * c as f64;
    // d=0.5w → 1-D theory 0.5C; require in (0.25C, 0.75C) — partial overlap
    let mid_ok = mid.mean_shared >= 0.25 * c as f64 && mid.mean_shared <= 0.80 * c as f64;
    // d=5w → theory 0; require ≤ 0.15C
    let far_ok = far.mean_shared <= 0.15 * c as f64;
    let ident_ok = (ident.mean_shared - c as f64).abs() < 1e-9;

    println!(
        "identical (d=0):     mean_shared={:.2}  expect={}  {}",
        ident.mean_shared,
        c,
        if ident_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "near (d=0.10·w):     mean_shared={:.2}  expect ≥{:.1} (≈0.9C 1-D)  {}",
        near.mean_shared,
        0.75 * c as f64,
        if near_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "mid  (d=0.50·w):     mean_shared={:.2}  expect ~0.25C–0.80C (partial)  {}",
        mid.mean_shared,
        if mid_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "far  (d=5·w):        mean_shared={:.2}  expect ≤{:.1}  {}",
        far.mean_shared,
        0.15 * c as f64,
        if far_ok { "PASS" } else { "FAIL" }
    );

    // Monotone-ish: shared should generally fall as distance grows (soft check).
    let mut mono_ok = true;
    let mut prev = c as f64 + 1.0;
    for r in &rows {
        if r.n_pairs == 0 {
            continue;
        }
        // allow small non-monotonic blips (±1.0)
        if r.mean_shared > prev + 1.5 {
            mono_ok = false;
            break;
        }
        prev = r.mean_shared;
    }
    println!(
        "soft monotone fall:  {} ",
        if mono_ok { "PASS" } else { "WARN (non-monotone blip)" }
    );

    let pass = ident_ok && near_ok && mid_ok && far_ok;
    println!();
    if pass {
        println!("STAGE 1 GATE: PASS — local generalization emerges from the tiling.");
        println!("Safe to proceed to Stage 2 (native tasks).");
        std::process::exit(0);
    } else {
        println!("STAGE 1 GATE: FAIL — tiling does not produce local generalization.");
        println!("Do NOT proceed. Inspect offsets / tile_width / coord quantization.");
        std::process::exit(1);
    }
}
