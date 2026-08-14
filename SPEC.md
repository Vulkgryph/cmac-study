# CMAC — Pre-Registered SPEC (Forgotten Architectures #2)

**Status:** pre-registered — predictions frozen **2026-08-12**, before any runs. Outcomes are **append-only**; predictions are never edited.
**Study:** `cmac-study` (series: NN-Revival). Reuses `ramnet-study` patterns; no shared-harness framework built up front.

## Core question

CMAC (Albus, 1975) computes by **sparse conditional computation**: an input activates a small *fixed* number **C** of memory cells (via overlapping quantization "tilings," hashed into bounded memory); output = **sum** of the C active cells; learning = a **local delta rule** on only those C cells (error×η/C), no backprop, **O(C) per example**. 

Does this sparse/specialized addressing **(a)** avoid the capacity saturation vanilla WiSARD hit past its knee (#1's headline finding), and **(b)** at what cost (hash collisions; loss of local generalization in high dimensions)? This directly tests the sparse-addressing fix predicted in #1.

## Faithfulness (priority — see ROADMAP)

The **faithful anchors are in CMAC's native domain** (low-dimensional function approximation / control). The MNIST arm is a **labeled, secondary comparability stretch** — out of native domain (784-d), requiring hashing-CMAC. **Rigor and defensibility concentrate on the native anchors.**

## Tasks & arms

**Faithful anchors (native — the priority):**
- **Fn-approx** — approximate a fixed 2-D nonlinear surface (sum-of-Gaussians / `sin·cos` field) over a bounded domain. Metric: held-out RMSE.
- **Control (legacy-native)** — CMAC's 1975 purpose: an **open-loop** inverse-kinematics-style mapping (target → joint/control output) for a simple 2-link arm. Metric: held-out error. *(Open-loop/static — no closed-loop dynamics; if it balloons, cut to a follow-up.)*

**Comparability arm (secondary, out-of-domain):**
- **MNIST** — hashing-CMAC on binarized MNIST vs `wisard` (#1) and `mlp` (#1).

**Arms:** `cmac` (tilings + local delta rule; hashing variant for MNIST) · `wisard` (carried from #1) · `mlp` (carried from #1).

## Experiments

- **Q1 — error/accuracy + sample-efficiency** on each native task and on MNIST (k-curve as in #1 for MNIST).
- **Q2 — the #1 tie-in (central):** capacity sweep (number of tilings C, hash-table size, resolution) on the native anchors + MNIST — does quality saturate/degrade like WiSARD, or plateau more softly?
- **Q3 — CMAC's own failure mode:** hash-collision sweep (table size) — where do collisions bite?
- **Q4 — online adaptation:** streaming/nonstationary (track a changing function) via local updates — CMAC's edge.

## Metrics

Held-out error (RMSE / accuracy) · sample-efficiency · **active-cells-per-example (O(C) sparse compute)** · update time · memory footprint.

## Pre-registered predictions (frozen 2026-08-12; outcomes append-only)

**Native anchors (the faithful priority):**
- **N1** — CMAC approximates the native tasks well (low error, competitive with a small MLP). *[med-high]*
- **N2** — sample-efficient (local generalization → low error from few samples). *[med-high]*
- **N3** — online adaptation works: tracks a changing function via cheap local updates. *[med]*
- **N4** — capacity scaling shows a **softer plateau than WiSARD** (no sharp degrade). *[med]*
- **N5** — active-cells-per-example stays **O(C)** regardless of capacity (the efficiency win). *[high]*

**MNIST comparability (secondary, out-of-domain):**
- **M1** — CMAC < MLP on MNIST (out of native domain). *[high]*
- **M2** — on 784-d, local generalization is weakened (curse of dimensionality → behaves more like a random sparse code than smooth interpolation). *[med]*
- **M3** — vs WiSARD: sparse-tiled vs n-tuple addressing — direction open. *[low-med]*

**Banked (NOT this study — hybrid-CMAC rethink extension):**
- **P7** — a static+plastic CMAC shows readout domination like #1's hybrid: the plastic (online) channel over-saturates and dominates the shared sum unless explicitly balanced/calibrated. *(Pre-registered when that extension runs.)*

## Method / integrity

Same discipline as #1: **3 seeds**, natural budgets, mean±range, results **frozen with SHA-256**, **figures generated from the frozen JSON** (no hand-typed numbers), honest scorecard **including the misses**, **no retcon**. **Faithful-first:** the native-anchor results are the priority for accuracy and defensibility; MNIST is labeled comparability.

## Spike-first (the risky unknown)

The native function-approx is CMAC's bread-and-butter (low risk). The **uncertain** part is hashing-CMAC learning on **784-d binarized MNIST** — spike that on a tiny subset before building the harness: *does the local delta rule + hashing learn anything on high-dim binary input?*

**Spike outcome (2026-08-12):** YES — 3-class subset, C=32, test_acc 0.96 ≫ chance 0.33. Gate cleared before harness.

---

## Outcomes (append only — predictions above never edited)

### Stage 1 gate — PASS (2026-08-12)

Faithful tiling CMAC: C=16, tile_w=0.1, domain [0,1]². Shared geometric tiles vs distance: d=0 → 16/16; d=0.1w → 13.92; d=0.5w → 7.23; d≥1.5w → 0. Local generalization emerges. Probe frozen in `results/stage2_full_frozen/local_gen_probe.json`.

### Stage 2 multi-seed native — FROZEN (2026-08-12)

**Protocol:** seeds {0,1,2}, n_train∈{50,200,1000,4000}, early-stop best-val, C=32, tile_w=0.05, table=8192, η=0.35. MLP h=64. Artifacts: `results/stage2_full_frozen/`  
SHA-256 `summary.json`: `d41c756fbe19b22706c639b8e6f0a28ac9ade27be424e100d877a5bbde6b3603`  
SHA-256 `local_gen_probe.json`: `8ca8b474bbad7ad034a86ff56503d4f7868cb20ee889cbcc19292cc6b434d698`

**Q1 test RMSE (mean±range):**

| task | arm | n=50 | n=200 | n=1000 | n=4000 | params | active |
|------|-----|------|-------|--------|--------|--------|--------|
| fn_approx | cmac | 0.4502±0.0036 | 0.3442±0.0121 | 0.1262±0.0032 | **0.0175±0.0047** | **262 144** | 32 |
| fn_approx | mlp | 0.4876±0.0043 | 0.3778±0.0718 | 0.1550±0.0058 | 0.1465±0.0004 | **257** | 64 |
| ik | cmac | 1.5635±0.0112 | 1.1420±0.0173 | **0.3978±0.0360** | **0.1697±0.0222** | **524 288** | 32 |
| ik | mlp | **0.8079±0.0396** | **0.5901±0.0439** | 0.4981±0.0849 | 0.3705±0.0343 | **322** | 64 |

**Crossover (headline):**
- **fn_approx:** mean CMAC < MLP at all n≥50; but at n=200 **per-seed mixed** (seeds 0,1: MLP better; seed 2: CMAC better because MLP failed to train). Solid CMAC lead from n=1000 (0.126 vs 0.155) and n=4000 (0.018 vs 0.147).
- **ik:** MLP wins n=50 and n=200 by a wide margin; CMAC crosses at **n≥1000** and leads strongly at n=4000 (0.170 vs 0.370).

#### Scorecard (predictions untouched)

| # | Verdict | Evidence |
|---|---------|----------|
| **N1** approx well | **TRUE above coverage** | At n=4000: fn-approx RMSE 0.018 (signal scale ~0.4); IK 0.17 rad. Competitive with / better than small MLP once covered. |
| **N2** sample-efficient from few samples | **PARTIAL MISS — crossover finding** | As literally pre-registered (“low error from few samples”), **contradicted** in the ultra-low-n regime: IK n=50/200 MLP ≪ CMAC; fn-approx n=200 per-seed often favors MLP. CMAC becomes sample-efficient **only after domain coverage** (local gen helps within ~tile_w of seen points; unvisited tiles contribute nothing). Honest story = **crossover**, not blanket efficiency. |
| **N5** active = O(C) | **TRUE** | active_cells ≡ 32 on every CMAC row; update ~0.8–1.4 µs/example. |
| **Param cost** | **stated** | CMAC 262k–524k table floats vs MLP 257–322 (~1000×). Efficiency claim is **compute/example O(C)**, not memory. On-ramp to Q2/Q3. |
| **N3, N4** | deferred | Stage 4 sweeps not run yet. |

### Stage 3 MNIST (hashing-CMAC, out-of-domain) — FROZEN + bleach fix (2026-08-12)

**Label:** hashing-CMAC is CMAC-inspired bit-subset hashing — **not** the faithful tiling claim.  
**WiSARD fix:** bleach search ON (stratified train subset) — full-data now comparable to #1 (~0.877).  
Broken no-bleach archive kept at `results/stage3_mnist_frozen_noblech_ARCHIVE/` (do not cite).  
SHA-256 `summary.json`: `1c818cbb49c635b9d3451c7d078f9bf889b57251970d0c93a8aa26646f2a51c8`

| arm | k=1 | k=5 | k=10 | k=50 | k=100 | full | params |
|-----|-----|-----|------|------|-------|------|--------|
| hash_cmac | 0.3552±0.0121 | 0.5872±0.0089 | 0.7002±0.0183 | 0.8370±0.0068 | 0.8731±0.0024 | 0.9496±0.0012 | 5 242 880 |
| wisard | 0.3979±0.0134 | **0.6614±0.0219** | **0.7706±0.0277** | **0.8629±0.0070** | 0.8696±0.0036 | **0.8749±0.0017** | 0 |
| mlp | 0.3741±0.0328 | 0.6630±0.0030 | 0.7517±0.0249 | 0.8526±0.0091 | 0.8739±0.0025 | **0.9718±0.0013** | 50 890 |

| # | Verdict | Evidence |
|---|---------|----------|
| **M1** hash-CMAC < MLP full | **TRUE** | 0.9496 < 0.9718 |
| **M2** local gen weakened in 784-d | **consistent with, not independently tested** | Hashing-CMAC is a random sparse code *by construction* (no geometric tilings). Result is consistent with M2 but cannot validate “loss of local gen” — there was none to lose. Do not claim M2 validated. |
| **M3** vs WiSARD | **low-data: WiSARD; full: MLP > hash_cmac > bleached WiSARD** | k=5/10: wisard 0.66/0.77 > hash_cmac 0.59/0.70 (n-tuple stronger sparse basis at few samples). Full: mlp 0.972 > hash_cmac 0.950 > wisard 0.875 (matches #1’s ~0.877). |

### Stage 4 sweeps — FROZEN (2026-08-12)

SHA-256 `summary.json`: `ab7fe382f4ff0ac72f4dc6624d1dd2419268d6ac3cc1bfad5866a8da4904acd1`  
Artifact: `results/stage4_full_frozen/`.

**Q2 capacity (fn_approx, n=4000, table=16384) vs #1 WiSARD N-scaling (cited):**

| C | tile_w | CMAC test_rmse | active | params |
|---|--------|----------------|--------|--------|
| 8 | 0.100 | 0.0220±0.0003 | 8 | 131k |
| 16 | 0.100 | 0.0147±0.0003 | 16 | 262k |
| 32 | 0.100 | **0.0121±0.0002** | 32 | 524k |
| 32 | 0.050 | 0.0170±0.0033 | 32 | 524k |
| 64 | 0.050 | 0.0170±0.0033 | 64 | 1.0M |
| 64 | 0.025 | **0.1232±0.0014** | 64 | 1.0M |
| 128 | 0.025 | 0.1231±0.0012 | 128 | 2.1M |

#1 WiSARD (cited): N=100→0.848, 500→0.871, 1k→0.877, 5k→0.876, **10k→0.847 (dip)**.

| # | Verdict | Evidence |
|---|---------|----------|
| **N4** (pre-control) | was MIXED | C↑ at fixed w softens; w=0.025 at n=4000 looked like a cliff (0.123). |

### N4 control — FROZEN (2026-08-12) → **N4 SUPPORTED**

SHA-256 `stage4_n4_control_frozen/summary.json`: `a211d0ef4a2d3beffbc200e6d7ca179506d1215115c25e83daa59991a8bfe2a0`

**A. Fine-w × n_train (C=64, w=0.025, table=16384 — collisions out):**

| n_train | test_rmse |
|--------:|----------:|
| 500 | 0.3828±0.0010 |
| 1000 | 0.3156±0.0066 |
| 2000 | 0.2144±0.0084 |
| 4000 | 0.1178±0.0130 |
| 8000 | 0.0403±0.0035 |
| 16000 | 0.0135±0.0020 |
| **32000** | **0.0036±0.0004** |

Cliff **fully recovers** with data (0.118 → 0.0036). Mechanism = **undersampling / coverage** (N2 local-gen again), **not** interference-saturation.

**B. C-monotonicity at fixed w, n=4000:**

| C | w=0.10 RMSE | w=0.05 RMSE |
|--:|------------:|------------:|
| 4 | 0.0379 | 0.0237 |
| 8 | 0.0223 | 0.0189 |
| 16 | 0.0149 | 0.0173 |
| 32 | 0.0124 | 0.0170 |
| 64 | 0.0117 | 0.0170 |
| 128 | 0.0114 | 0.0168 |
| 256 | 0.0113 | 0.0168 |

**Adding C never hurts** (no Δ>+0.003 rise). Only softens then plateaus.

| # | Verdict | Evidence |
|---|---------|----------|
| **N4** sparse addr avoids WiSARD saturation | **SUPPORTED** | (1) C-axis: more tilings never degrade quality — only help/plateau. (2) The w=0.025 “cliff” is coverage, proven by recovery with n (0.118@4k → 0.0036@32k). WiSARD’s N=10k dip is intrinsic interference and does **not** recover with the same fix. Two different phenomena; CMAC has no interference-saturation cliff on the capacity axis. |

**Q3 collisions (C=32, w=0.05, shrink table):**

| table | test_rmse | params |
|------:|----------:|-------:|
| 64 | 0.393±0.004 | 2k |
| 128 | 0.360±0.004 | 4k |
| 256 | 0.245±0.002 | 8k |
| 512 | 0.186±0.002 | 16k |
| **1024** | **0.013±0.002** | 33k |
| ≥2048 | 0.013±0.002 (flat) | 65k–524k |

Collisions bite hard below ~1024 bins; above that the big table **does not earn more quality** — footprint from 33k→524k is pure waste at this task. Efficiency is O(C) compute; memory only needs to clear the collision floor.

**Q4 online (phase 0 → π/2):**

| arm | pre (ph0) | mid-stream | post_online | post_frozen |
|-----|-----------|------------|-------------|-------------|
| cmac | 0.0149±0.0010 | 0.383±0.012 | **0.190±0.019** | 0.754±0.012 |
| mlp | 0.1578±0.0028 | 0.319±0.048 | 0.302±0.020 | 0.745±0.018 |

| # | Verdict | Evidence |
|---|---------|----------|
| **N3** online adaptation | **TRUE — CMAC adapts better at O(C), not “only CMAC adapts”** | Frozen CMAC collapses on new phase (0.75); one-pass local updates recover to **0.19**. MLP online (no replay) also recovers (0.30 vs frozen 0.75) but worse and from a weaker pre (0.16). Honest claim: CMAC adapts **better** and at **O(C) cost without replay** — not that MLP can’t. |

**Q3 param note (tempers Stage-2 cost story):** collisions clear by table=1024 (C·table ≈ 32k params); flat through 16k. Default Stage-2 table (262k) was ~8× oversized. Real memory penalty vs MLP (~300) is ~**130×**, not 1000× — efficiency is still compute-not-memory, but say the real number.

### Stop line

Stages 1–4 + N4 control complete and frozen. **No writeup yet** — N4 headline is now clean: sparse addressing avoids WiSARD’s interference-saturation; CMAC’s only sharp failure is coverage (curable with data). Broken no-bleach Stage 3 archived, not for citation.
