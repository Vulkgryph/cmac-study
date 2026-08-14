# cmac-study Stage 2 native results (stage2-smoke-seed0)

| task | arm | n_train | test_rmse | val_rmse | best_ep | train_ms | update_us | active | params | notes |
|------|-----|---------|-----------|----------|---------|----------|-----------|--------|--------|-------|
| fn_approx | cmac | 50 | 0.459221 | 0.455950 | 27 | 44.6 | 0.82 | 32 | 262144 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| fn_approx | cmac | 200 | 0.355401 | 0.351428 | 79 | 75.9 | 0.75 | 32 | 262144 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| fn_approx | cmac | 1000 | 0.113585 | 0.108243 | 80 | 168.7 | 1.09 | 32 | 262144 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| fn_approx | cmac | 4000 | 0.013845 | 0.013019 | 78 | 508.5 | 0.80 | 32 | 262144 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| fn_approx | mlp | 50 | 0.493555 | 0.485084 | 2 | 4.5 | — | 64 | 257 | mlp_cont(h=64, lr=0.05) |
| fn_approx | mlp | 200 | 0.315684 | 0.317900 | 56 | 31.6 | — | 64 | 257 | mlp_cont(h=64, lr=0.05) |
| fn_approx | mlp | 1000 | 0.155304 | 0.159175 | 77 | 88.2 | — | 64 | 257 | mlp_cont(h=64, lr=0.05) |
| fn_approx | mlp | 4000 | 0.149293 | 0.149642 | 45 | 194.0 | — | 64 | 257 | mlp_cont(h=64, lr=0.05) |
| ik | cmac | 50 | 1.518675 | 1.595007 | 77 | 62.1 | 0.90 | 32 | 524288 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| ik | cmac | 200 | 1.134967 | 1.230640 | 80 | 80.8 | 0.85 | 32 | 524288 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| ik | cmac | 1000 | 0.356459 | 0.337485 | 79 | 177.2 | 0.85 | 32 | 524288 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| ik | cmac | 4000 | 0.132899 | 0.122127 | 45 | 365.9 | 0.87 | 32 | 524288 | tiling_cmac(C=32, w=0.0500, table=8192, η=0.35) |
| ik | mlp | 50 | 0.739864 | 0.728913 | 75 | 29.6 | — | 64 | 322 | mlp_cont(h=64, lr=0.05) |
| ik | mlp | 200 | 0.687455 | 0.685790 | 33 | 21.5 | — | 64 | 322 | mlp_cont(h=64, lr=0.05) |
| ik | mlp | 1000 | 0.499044 | 0.461146 | 30 | 46.2 | — | 64 | 322 | mlp_cont(h=64, lr=0.05) |
| ik | mlp | 4000 | 0.378098 | 0.339722 | 68 | 279.7 | — | 64 | 322 | mlp_cont(h=64, lr=0.05) |

_Stage 2 1-seed sanity. RMSE at best-val early-stop. active_cells must equal C for cmac._
