# cmac-study

Code and frozen results for **Forgotten Architectures #2 — CMAC on Modern Compute**.
Paper: <https://vulkgryph.com/research/nn-revival/papers/cmac>
Series: <https://vulkgryph.com/research/nn-revival/>

A controlled, honest re-run of Albus's **CMAC** (Cerebellar Model Articulation Controller, 1975) in its native domain — low-dimensional function approximation and open-loop control — plus a labeled, out-of-domain MNIST comparability arm. The central question, following #1: does CMAC's **sparse tiled addressing** avoid the capacity saturation vanilla WiSARD hit past its knee? Predictions are pre-registered and frozen; three seeds; results frozen and hashed.

## Arms

| arm | what it is |
|-----|-----------|
| `cmac` (native) | faithful CMAC — C overlapping offset quantization tilings, sum of C active cells, local delta rule (η/C), O(C) per example |
| `cmac` (MNIST) | out-of-domain hashing adaptation — C random bit-subsets → FNV hash → bounded table; a sparse code, **not** faithful tiling |
| `mlp` | small dense baseline (one hidden layer) |
| `wisard` | carried from #1 for the MNIST comparability arm (bleach-threshold search) |

## Key finding

Across the capacity range swept (C from 4 to 256), **adding tilings never degrades quality** — it improves then plateaus — where WiSARD's N-scaling reverses at N=10k. CMAC's one sharp failure (tiles too fine for the data budget) is **undersampling, not interference**: it fully recovers with more data (RMSE 0.123 → 0.0036 as n grows 4k → 32k). The costs, told straight: sample efficiency is *local* (a small MLP wins the ultra-low-data regime; CMAC crosses ahead at n≥1000), the table is large (~130× the MLP's parameters at the collision floor), and on 784-d MNIST — out of native domain — the MLP wins.

## Reproduce

Requires a recent Rust toolchain and Python 3 (standard library only).

**Native tasks** (fn-approx + open-loop IK) need no external data.

**MNIST comparability** requires the four standard IDX files in `data/mnist/`:

```
data/mnist/train-images-idx3-ubyte
data/mnist/train-labels-idx1-ubyte
data/mnist/t10k-images-idx3-ubyte
data/mnist/t10k-labels-idx1-ubyte
```

(Standard, widely-mirrored dataset. The MNIST runner requires real IDX and aborts rather than silently falling back.)

**Run the campaigns:**

```
cargo run --release --bin stage1_gate         # faithfulness gate: local-generalization probe
cargo run --release --bin stage2_full          # Q1 native sample-efficiency, 3 seeds (fn-approx + IK)
cargo run --release --bin stage3_mnist          # MNIST comparability (hash-CMAC / wisard / mlp)
cargo run --release --bin stage4_sweeps         # Q2 capacity / Q3 collisions / Q4 online
cargo run --release --bin stage4_n4_control     # coverage-vs-saturation control (fine-w × n, C-monotonicity)
```

(`stage2_native` is a single-seed sanity runner; `stage2_full` is the multi-seed campaign whose output is frozen.)

**Regenerate the paper's tables from the frozen results:**

```
python3 gen_tables.py
```

Every figure in the paper is *generated* from `results/*_frozen/*.json` by this script — no result numbers are hand-typed.

## Results & reproducibility

- Frozen results live in `results/*_frozen/` (JSON + Markdown). Their SHA-256 hashes are recorded in the paper and in `results/FROZEN_SHA256.txt`.
- Predictions were written in `SPEC.md` before the runs and are **append-only** — outcomes are added; predictions are never edited.
- `results/stage3_mnist_frozen_noblech_ARCHIVE/` is a **broken** no-bleach WiSARD run, kept for transparency and labeled **do not cite** — it documents why bleached WiSARD (0.875) is the honest full-data number.

## Scope

This studies CMAC in its **original** form under modern controls, in its native domain, plus a labeled out-of-domain MNIST arm. It is a toy-scale study with small baselines — **not a SOTA claim** (a CNN would beat every arm on MNIST; the native tasks are synthetic and chosen to be genuinely native, not hard). See the paper's Scope and Limitations sections.

## Contact

See [vulkgryph.com](https://vulkgryph.com).

## License

Apache-2.0 — see [LICENSE](LICENSE).

---

*Code and experiments produced with AI coding agents under the author's direction; figures are generated from the frozen results by `gen_tables.py` and audited by the author.*
