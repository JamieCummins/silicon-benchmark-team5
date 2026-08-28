"""Error-metric-first evaluation + positive/negative observed-effect split.

Addresses: (1) MAE/RMSE alongside r everywhere; (2) the known silicon-sample
weakness on negative effects — how do raw and tuned predictions behave on
cells where the TRUE effect is negative?

  uv run python scripts/analyze_sign_split.py
"""

import numpy as np
import pandas as pd

from silicon.config import RUNS_DIR
from silicon.retro_vlasceanu import vlasceanu_ground
from silicon.scoring import score
from silicon.tune import aggregate_with, grid_search

pd.set_option("display.width", 140)

CSV = RUNS_DIR / "retro_vlasceanu_elicitations.csv"
g = vlasceanu_ground(CSV)
el = pd.read_csv(CSV)


def preds_for(sub: pd.DataFrame, priors=None, w=0.0, lam=1.0, trim=0.1) -> pd.DataFrame:
    final = aggregate_with(sub, priors or g.priors, g.epsilon, w_prior=w, lam=lam, trim=trim)
    return final[["condition", "outcome", "ate"]]


def merged_pp(pred: pd.DataFrame) -> pd.DataFrame:
    m = pred.merge(g.observed, on=["condition", "outcome"], suffixes=("_p", "_o"))
    m["p"] = m.apply(lambda r: r["ate_p"] * 100 / g.ranges[r["outcome"]], axis=1)
    m["o"] = m.apply(lambda r: r["ate_o"] * 100 / g.ranges[r["outcome"]], axis=1)
    return m


def metrics_row(pred: pd.DataFrame) -> dict:
    s = score(pred, g.observed, g.ranges)
    return {k: s.get(k, np.nan) for k in ("pearson_r", "within_outcome_r", "directional_pct",
                                          "mae_pp", "rmse_pp", "calib_beta")}


# ============ 1. MAE/RMSE-first metrics table ============
print("=== metrics with MAE/RMSE front and center (all in pp of scale range) ===")
rows = {"RAW crowd (all models)": metrics_row(preds_for(el))}
for m, sub in el.groupby("model"):
    rows[f"solo: {m}"] = metrics_row(preds_for(sub))
gs = grid_search([g])
best_r = gs.sort_values("pearson_r", ascending=False).iloc[0]
best_rmse = gs.sort_values("rmse_pp").iloc[0]
best_mae = gs.sort_values("mae_pp").iloc[0]
rows[f"TUNED best-r (w={best_r['w_prior']}, lam={best_r['lam']})"] = metrics_row(
    preds_for(el, w=best_r["w_prior"], lam=best_r["lam"], trim=best_r["trim"]))
rows[f"TUNED best-RMSE (w={best_rmse['w_prior']}, lam={best_rmse['lam']})"] = metrics_row(
    preds_for(el, w=best_rmse["w_prior"], lam=best_rmse["lam"], trim=best_rmse["trim"]))
rows[f"TUNED best-MAE (w={best_mae['w_prior']}, lam={best_mae['lam']})"] = metrics_row(
    preds_for(el, w=best_mae["w_prior"], lam=best_mae["lam"], trim=best_mae["trim"]))
# baseline: predict 0 everywhere (the all-null forecaster)
null_pred = g.observed[["condition", "outcome"]].assign(ate=0.0)
rows["baseline: all-zero forecaster"] = metrics_row(null_pred)
print(pd.DataFrame(rows).T.round(3).to_string())

# ============ 2. split by sign of the OBSERVED effect ============
raw = merged_pp(preds_for(el))
tuned = merged_pp(preds_for(el, w=best_r["w_prior"], lam=best_r["lam"], trim=best_r["trim"]))

print("\n=== split by sign of OBSERVED effect (raw crowd) ===")
for label, grp in raw.groupby(np.sign(raw["o"]).map({1.0: "observed POSITIVE", -1.0: "observed NEGATIVE"})):
    within = np.nan
    if len(grp) > 3:
        gg = grp.copy()
        gg["p_c"] = gg["p"] - gg.groupby("outcome")["p"].transform("mean")
        gg["o_c"] = gg["o"] - gg.groupby("outcome")["o"].transform("mean")
        if gg["p_c"].std() > 0 and gg["o_c"].std() > 0:
            within = np.corrcoef(gg["p_c"], gg["o_c"])[0, 1]
    print(f"{label}: n={len(grp)}  outcomes={dict(grp['outcome'].value_counts())}")
    print(f"   mean obs={grp['o'].mean():+.2f}pp  mean pred={grp['p'].mean():+.2f}pp  "
          f"directional={np.mean(np.sign(grp['p']) == np.sign(grp['o'])) * 100:.0f}%  "
          f"MAE={np.mean(np.abs(grp['p'] - grp['o'])):.2f}pp  "
          f"RMSE={np.sqrt(np.mean((grp['p'] - grp['o']) ** 2)):.2f}pp  within-r={within:+.2f}")

print("\nsign confusion matrix (raw crowd), cells:")
conf = pd.crosstab(np.sign(raw["o"]).map({1: "obs +", -1: "obs -"}),
                   np.sign(raw["p"]).map({1: "pred +", -1: "pred -", 0: "pred 0"}))
print(conf.to_string())

neg_rate_pred = (raw["p"] < 0).mean() * 100
neg_rate_obs = (raw["o"] < 0).mean() * 100
print(f"\npositivity bias: {neg_rate_obs:.0f}% of observed cells are negative, "
      f"but only {neg_rate_pred:.0f}% of predictions are negative")

print("\nper-model willingness to predict negative effects (share of that model's cells):")
el_pp = el.assign(pp=el.apply(lambda r: r["ate"] * 100 / g.ranges[r["outcome"]], axis=1))
for m, sub in el_pp.groupby("model"):
    wept = sub[sub["outcome"] == "wept"]
    print(f"  {m:10s} all cells: {(sub['pp'] < 0).mean() * 100:5.1f}%   "
          f"wept cells only: {(wept['pp'] < 0).mean() * 100:5.1f}%  (truth: 73% of wept cells negative)")

# ============ 3. the fix mechanism: outcome-level sign prior ============
print("\n=== demonstration: outcome-mean prior with the correct sign for wept ===")
print("(wept prior set to -0.25 on the 0-8 scale from the effort-licensing literature;")
print(" post-hoc on this ground -> mechanism demo, NOT a tuned result. Everything else unchanged.)")
fixed_priors = dict(g.priors) | {"wept": -0.25}
for label, priors in [("original priors (wept prior 0.0)", g.priors),
                      ("sign-informed priors (wept -0.25)", fixed_priors)]:
    pred = preds_for(el, priors=priors, w=best_r["w_prior"], lam=best_r["lam"], trim=best_r["trim"])
    mm = merged_pp(pred)
    neg = mm[mm["o"] < 0]
    s = score(pred, g.observed, g.ranges)
    print(f"{label}:")
    print(f"   overall: r={s['pearson_r']:.3f}  directional={s['directional_pct']:.0f}%  "
          f"MAE={s['mae_pp']:.2f}  RMSE={s['rmse_pp']:.2f}")
    print(f"   negative-obs cells: directional={np.mean(np.sign(neg['p']) == np.sign(neg['o'])) * 100:.0f}%  "
          f"MAE={np.mean(np.abs(neg['p'] - neg['o'])):.2f}pp")
