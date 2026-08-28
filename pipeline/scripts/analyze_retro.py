"""Analyze the Vlasceanu retrodiction run: skill, tuning, ablations.

Run after scripts/run_retro.py --small/--full:
  uv run python scripts/analyze_retro.py
Writes tuned hyperparameters to runs/retro_tuned_params.json.
"""

import itertools
import json

import numpy as np
import pandas as pd

from silicon.config import RUNS_DIR
from silicon.retro_vlasceanu import vlasceanu_ground
from silicon.scoring import score
from silicon.tune import aggregate_with, grid_search

CSV = RUNS_DIR / "retro_vlasceanu_elicitations.csv"
g = vlasceanu_ground(CSV)
el = pd.read_csv(CSV)

print(f"elicitations: {len(el)} rows, {el.groupby(['model','variant','intervention']).ngroups} cells, "
      f"{el['model'].nunique()} models\n")
print("cells per model:")
print(el.groupby("model")["variant"].apply(lambda s: len(s) // 4).to_string(), "\n")


def quick_score(sub: pd.DataFrame, w=0.0, lam=1.0, trim=0.1) -> dict:
    final = aggregate_with(sub, g.priors, g.epsilon, w_prior=w, lam=lam, trim=trim)
    return score(final[["condition", "outcome", "ate"]], g.observed, g.ranges)


# 1. RAW crowd skill (no calibration: pure trimmed-mean crowd)
raw = quick_score(el)
print("=== RAW crowd (w_prior=0, lam=1, trim=0.1) ===")
print({k: round(v, 3) for k, v in raw.items()})

# 2. Per-model solo skill (each model's variants as its own crowd)
print("\n=== per-model solo skill (raw) ===")
rows = []
for m, sub in el.groupby("model"):
    s = quick_score(sub)
    rows.append({"model": m, **{k: s[k] for k in
                 ("pearson_r", "within_outcome_r", "directional_pct", "mae_pp", "rmse_pp", "calib_beta")}})
print(pd.DataFrame(rows).set_index("model").round(3).to_string())

# 3. Hyperparameter grid (single ground -> provisional until Hewitt confirms transfer)
gs = grid_search([g])
print("\n=== top 10 combos by pearson_r ===")
top_r = gs.sort_values("pearson_r", ascending=False).head(10)
print(top_r[["w_prior", "lam", "trim", "pearson_r", "within_outcome_r",
             "directional_pct", "mae_pp", "rmse_pp", "calib_beta"]].round(3).to_string(index=False))
print("\n=== top 10 combos by rmse_pp (lower better) ===")
top_rmse = gs.sort_values("rmse_pp").head(10)
print(top_rmse[["w_prior", "lam", "trim", "pearson_r", "within_outcome_r",
                "directional_pct", "mae_pp", "rmse_pp", "calib_beta"]].round(3).to_string(index=False))

# 4. leave-one-model-out at the best-r combo
best = gs.sort_values("pearson_r", ascending=False).iloc[0]
print(f"\n=== leave-one-model-out at (w={best['w_prior']}, lam={best['lam']}, trim={best['trim']}) ===")
rows = []
for drop in [None, *sorted(el["model"].unique())]:
    sub = el if drop is None else el[el["model"] != drop]
    final = aggregate_with(sub, g.priors, g.epsilon, best["w_prior"], best["lam"], best["trim"])
    s = score(final[["condition", "outcome", "ate"]], g.observed, g.ranges)
    rows.append({"dropped": drop or "-", "pearson_r": s["pearson_r"],
                 "within_outcome_r": s["within_outcome_r"], "rmse_pp": s["rmse_pp"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 5. Spearman-Brown: mean inter-variant correlation per model (pp units, pooled cells)
print("\n=== inter-variant agreement per model (mean pairwise r of prediction vectors) ===")
el_pp = el.assign(pp=el.apply(lambda r: r["ate"] * 100 / g.ranges[r["outcome"]], axis=1))
for m, sub in el_pp.groupby("model"):
    piv = sub.pivot_table(index=["intervention", "outcome"], columns="variant", values="pp")
    cors = [piv[a].corr(piv[b]) for a, b in itertools.combinations(piv.columns, 2)]
    rbar = float(np.mean(cors)) if cors else float("nan")
    k = len(piv.columns)
    sb12 = 12 * rbar / (1 + 11 * rbar) if rbar == rbar else float("nan")
    print(f"  {m:10s} K={k:2d}  mean r={rbar:.3f}  Spearman-Brown reliability at K=12: {sb12:.3f}")

# 6. persist tuned params (by-r winner and by-rmse winner)
out = {
    "ground": "vlasceanu_us",
    "n_cells": int(el.groupby(['model', 'variant', 'intervention']).ngroups),
    "raw": {k: round(float(v), 4) for k, v in raw.items()},
    "best_by_r": {k: float(best[k]) for k in ("w_prior", "lam", "trim")},
    "best_by_r_metrics": {k: round(float(best[k]), 4) for k in
                          ("pearson_r", "within_outcome_r", "rmse_pp", "calib_beta")},
    "best_by_rmse": {k: float(top_rmse.iloc[0][k]) for k in ("w_prior", "lam", "trim")},
}
(RUNS_DIR / "retro_tuned_params.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote runs/retro_tuned_params.json")
