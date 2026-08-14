# cmac-study Stage 3 MNIST results (stage3-mnist-full)

seeds: [0, 1, 2]

**Label:** hashing-CMAC is out-of-domain / CMAC-inspired — not the faithful tiling claim.

## Q1 MNIST k-curve (test acc, mean±range)

| arm | k=1 | k=5 | k=10 | k=50 | k=100 | full | params | active |
|-----|-----|-----|------|------|-------|------|--------|--------|
| hash_cmac | 0.3552±0.0121 | 0.5872±0.0089 | 0.7002±0.0183 | 0.8370±0.0068 | 0.8731±0.0024 | 0.9496±0.0012 | 5242880 | 64 |
| wisard | 0.3979±0.0134 | 0.6728±0.0050 | 0.7706±0.0277 | 0.8629±0.0070 | 0.8696±0.0036 | 0.6707±0.0106 | 0 | 1000 |
| mlp | 0.3741±0.0328 | 0.6630±0.0030 | 0.7517±0.0249 | 0.8526±0.0091 | 0.8739±0.0025 | 0.9718±0.0013 | 50890 | 64 |

_mean±range over seeds. k∈{1,5} averaged over low_k_draws per seed before seed-agg._
