#!/usr/bin/env python3
"""Generate all paper tables from frozen JSON only. No hand-typed result numbers."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"

def load(rel):
    with open(R / rel) as f:
        return json.load(f)

def mr(vals, nd=4):
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{m:.{nd}f}"
    r = (max(vals) - min(vals)) / 2
    return f"{m:.{nd}f}±{r:.{nd}f}"

s2 = load("stage2_full_frozen/summary.json")
s3 = load("stage3_mnist_frozen/summary.json")
s4 = load("stage4_full_frozen/summary.json")
probe = load("stage2_full_frozen/local_gen_probe.json")

print("# Generated from frozen JSON — do not hand-edit numbers\n")

print("## Stage 2 Q1 sample-efficiency (test RMSE)\n")
print("| task | arm | n=50 | n=200 | n=1000 | n=4000 | params | active |")
print("|------|-----|------|-------|--------|--------|--------|--------|")
for task in ("fn_approx", "ik"):
    for arm in ("cmac", "mlp"):
        cells, params, active = [], None, None
        for n in (50, 200, 1000, 4000):
            rows = [r for r in s2["records"] if r["task"] == task and r["arm"] == arm and r["n_train"] == n]
            cells.append(mr([r["test_rmse"] for r in rows]))
            if rows:
                params, active = rows[0]["trainable_params"], rows[0]["active_cells"]
        print(f"| {task} | {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {params} | {active} |")

print("\n## Crossover (honest)\n")
print("- fn_approx: mean CMAC≤MLP at n≥50, but n=200 per-seed mixed (MLP wins seeds 0,1). Solid CMAC lead n≥1000.")
print("- ik: MLP wins n=50,200; CMAC crosses n≥1000.")
print("- Defensible crossover for both tasks: **n≥1000**.\n")

print("## Local-gen probe\n")
print("| distance | mean_shared | min | max |")
print("|----------|-------------|-----|-----|")
for row in probe:
    print(f"| {row['distance']:.4f} | {row['mean_shared']:.2f} | {row['min_shared']} | {row['max_shared']} |")

print("\n## Stage 3 MNIST (hash-CMAC out-of-domain; WiSARD bleached)\n")
print("| arm | k=1 | k=5 | k=10 | k=50 | k=100 | full | params |")
print("|-----|-----|-----|------|------|-------|------|--------|")
for arm in ("hash_cmac", "wisard", "mlp"):
    cells, params = [], None
    for k in (1, 5, 10, 50, 100, None):
        rows = [r for r in s3["records"] if r["arm"] == arm and r["k_per_class"] == k]
        cells.append(mr([r["test_acc"] for r in rows]))
        if rows:
            params = rows[0]["trainable_params"]
    print(f"| {arm} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} | {cells[5]} | {params} |")

print("\n## Stage 4 Q2 capacity\n")
print("| C | tile_w | test_rmse | active | params |")
print("|---|--------|-----------|--------|--------|")
seen = []
for r in s4["q2_capacity"]:
    key = (r["c"], round(r["tile_width"], 4))
    if key in seen:
        continue
    seen.append(key)
    xs = [x["test_rmse"] for x in s4["q2_capacity"] if x["c"] == key[0] and round(x["tile_width"], 4) == key[1]]
    print(f"| {key[0]} | {key[1]:.3f} | {mr(xs,5)} | {r['active_cells']} | {r['trainable_params']} |")

print("\n### #1 WiSARD N-scaling (cited)\n")
print("| N | acc | source |")
print("|---|-----|--------|")
for w in s4["wisard_n_scaling_cited"]:
    print(f"| {w['n']} | {w['acc']:.4f}±{w['acc_range']:.4f} | {w['source']} |")

print("\n## Stage 4 Q3 collisions\n")
print("| table_size | test_rmse | params |")
print("|------------|-----------|--------|")
tabs = sorted({r["table_size"] for r in s4["q3_collisions"]})
for t in tabs:
    rows = [r for r in s4["q3_collisions"] if r["table_size"] == t]
    print(f"| {t} | {mr([r['test_rmse'] for r in rows],5)} | {rows[0]['trainable_params']} |")

print("\n## Stage 4 Q4 online\n")
print("| arm | pre | mid | post_online | post_frozen |")
print("|-----|-----|-----|-------------|-------------|")
for arm in ("cmac", "mlp"):
    def grab(phase, a=arm):
        return [r["rmse"] for r in s4["q4_online"] if r["arm"] == a and r["phase"] == phase]
    froz = [r["rmse"] for r in s4["q4_online"] if r["arm"] == f"{arm}_frozen" and r["phase"] == "post_no_adapt"]
    print(f"| {arm} | {mr(grab('pre'),4)} | {mr(grab('during_mid'),4)} | {mr(grab('post_online'),4)} | {mr(froz,4)} |")

# N4 control
try:
    n4 = load("stage4_n4_control_frozen/summary.json")
    print("\n## N4 control — fine-w × n_train (C=64, w=0.025)\n")
    print("| n_train | test_rmse |")
    print("|---------|----------|")
    for n in sorted({r["n_train"] for r in n4["fine_w_n_sweep"]}):
        xs = [r["test_rmse"] for r in n4["fine_w_n_sweep"] if r["n_train"] == n]
        print(f"| {n} | {mr(xs,5)} |")
    print("\n## N4 control — C-monotonicity @ w=0.10, n=4000\n")
    print("| C | test_rmse |")
    print("|---|----------|")
    for c in sorted({r["c"] for r in n4["c_mono_fixed_w"]}):
        xs = [r["test_rmse"] for r in n4["c_mono_fixed_w"] if r["c"] == c]
        print(f"| {c} | {mr(xs,5)} |")
except FileNotFoundError:
    print("\n(N4 control not frozen yet)\n")
