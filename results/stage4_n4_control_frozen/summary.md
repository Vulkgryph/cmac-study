# N4 control — coverage vs saturation + C-monotonicity

seeds: [0, 1, 2]

## A. Fine-w × n_train (C=64, w=0.025, table=16384)

If cliff is **coverage**: RMSE → ~0.01 as n↑. If **saturation**: stays high.

| n_train | test_rmse |
|---------|----------|
| 500 | 0.38283±0.00104 |
| 1000 | 0.31557±0.00656 |
| 2000 | 0.21439±0.00842 |
| 4000 | 0.11783±0.01302 |
| 8000 | 0.04031±0.00354 |
| 16000 | 0.01350±0.00200 |
| 32000 | 0.00356±0.00038 |

## B. C-monotonicity @ fixed w, n_train=4000

Does adding C ever *raise* test_rmse (hurt)?

### w=0.10

| C | test_rmse | active |
|---|-----------|--------|
| 4 | 0.03786±0.00124 | 4 |
| 8 | 0.02234±0.00065 | 8 |
| 16 | 0.01488±0.00033 | 16 |
| 32 | 0.01237±0.00035 | 32 |
| 64 | 0.01166±0.00040 | 64 |
| 128 | 0.01135±0.00033 | 128 |
| 256 | 0.01132±0.00035 | 256 |

_Adding C ever hurts (Δ>+0.003)? **NO**_

### w=0.05

| C | test_rmse | active |
|---|-----------|--------|
| 4 | 0.02375±0.00076 | 4 |
| 8 | 0.01887±0.00115 | 8 |
| 16 | 0.01727±0.00106 | 16 |
| 32 | 0.01695±0.00111 | 32 |
| 64 | 0.01697±0.00112 | 64 |
| 128 | 0.01684±0.00110 | 128 |
| 256 | 0.01680±0.00108 | 256 |

_Adding C ever hurts (Δ>+0.003)? **NO**_

## Verdict sketch (filled by numbers above)

- fine-w n=4000 RMSE = 0.1178; n=32000 RMSE = 0.0036  → RECOVERS with data → **coverage**, not saturation
