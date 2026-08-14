# cmac-study Stage 4 (stage4-full)

seeds: [0, 1, 2]

## Q2 — capacity sweep (fn_approx, n_train=4000, table=16384)

| C | tile_w | test_rmse | active | params |
|---|--------|-----------|--------|--------|
| 8 | 0.100 | 0.02197±0.00025 | 8 | 131072 |
| 16 | 0.100 | 0.01472±0.00027 | 16 | 262144 |
| 32 | 0.050 | 0.01701±0.00331 | 32 | 524288 |
| 32 | 0.100 | 0.01213±0.00024 | 32 | 524288 |
| 64 | 0.025 | 0.12317±0.00139 | 64 | 1048576 |
| 64 | 0.050 | 0.01700±0.00328 | 64 | 1048576 |
| 128 | 0.025 | 0.12305±0.00118 | 128 | 2097152 |

### #1 WiSARD N-scaling (cited from baseline_full_frozen — same framing)

| N | acc (mean±range) | source |
|---|------------------|--------|
| 100 | 0.8477±0.0064 | ramnet-study/results/baseline_full_frozen |
| 500 | 0.8713±0.0013 | ramnet-study/results/baseline_full_frozen |
| 1000 | 0.8769±0.0002 | ramnet-study/results/baseline_full_frozen |
| 5000 | 0.8759±0.0039 | ramnet-study/results/baseline_full_frozen |
| 10000 | 0.8474±0.0051 | ramnet-study/results/baseline_full_frozen |

_N4 test: does CMAC test_rmse soften/plateau as C↑ / w↓, vs WiSARD's N=10k dip?_

## Q3 — collision sweep (C=32, w=0.05, shrink table)

| table_size | test_rmse | params |
|------------|-----------|--------|
| 64 | 0.39272±0.00417 | 2048 |
| 128 | 0.35990±0.00419 | 4096 |
| 256 | 0.24533±0.00244 | 8192 |
| 512 | 0.18649±0.00217 | 16384 |
| 1024 | 0.01310±0.00151 | 32768 |
| 2048 | 0.01310±0.00151 | 65536 |
| 4096 | 0.01310±0.00151 | 131072 |
| 8192 | 0.01310±0.00151 | 262144 |
| 16384 | 0.01310±0.00151 | 524288 |

## Q4 — online adaptation (phase 0 → π/2)

| arm | pre (phase0) | mid-stream | post_online (phase π/2) | post_frozen |
|-----|--------------|------------|-------------------------|-------------|
| cmac | 0.01487±0.00097 | 0.38345±0.01242 | 0.19040±0.01913 | 0.75400±0.01156 |
| mlp | 0.15775±0.00281 | 0.31920±0.04751 | 0.30213±0.02030 | 0.74498±0.01836 |

_N3: online local updates should cut post RMSE vs frozen. MLP continues SGD without replay._
