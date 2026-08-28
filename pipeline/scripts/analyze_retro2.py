"""Evaluate retrodiction run 2 per the pre-analysis note (frozen params first).

  uv run python scripts/analyze_retro2.py
"""

import numpy as np
import pandas as pd

from silicon.config import RUNS_DIR
from silicon.retro_hewitt import hewitt_ground, scope_elicitations
from silicon.retro_vlasceanu import vlasceanu_ground
from silicon.scoring import bootstrap_ci, score
from silicon.tune import aggregate_with, grid_search

pd.set_option("display.width", 150)

el2 = scope_elicitations(pd.read_csv(RUNS_DIR / "retro_hewitt_elicitations.csv"))
SCOPED_CSV = RUNS_DIR / "retro_hewitt_elicitations_scoped.csv"
el2.to_csv(SCOPED_CSV, index=False)  # grid_search re-reads from disk -> must be pre-scoped
g2 = hewitt_ground(SCOPED_CSV)

FROZEN = {  # from run 1 (see reference/retro2_preanalysis.md) — confirm, don't retune
    "frozen best-r (w=.75,lam=.3,trim=.2)": dict(w=0.75, lam=0.3, trim=0.2),
    "frozen best-RMSE (w=.75,lam=1,trim=0)": dict(w=0.75, lam=1.0, trim=0.0),
    "fallback conservative (w=.5,lam=1)": dict(w=0.5, lam=1.0, trim=0.1),
}


def preds(sub, priors=None, w=0.0, lam=1.0, trim=0.1):
    final = aggregate_with(sub, priors or g2.priors, g2.epsilon, w_prior=w, lam=lam, trim=trim)
    return final[["condition", "outcome", "ate"]]


def row(pred):
    s = score(pred, g2.observed, g2.ranges)
    return {k: s.get(k, np.nan) for k in ("n_pairs", "pearson_r", "within_outcome_r",
                                          "directional_pct", "mae_pp", "rmse_pp", "calib_beta")}


# ---- 1. CONFIRMATION: raw + frozen params + baseline ----
print("=== ground 2 (Hewitt slice): raw, FROZEN run-1 params, baseline ===")
tbl = {"RAW crowd": row(preds(el2))}
for label, p in FROZEN.items():
    tbl[label] = row(preds(el2, w=p["w"], lam=p["lam"], trim=p["trim"]))
tbl["all-zero baseline"] = row(g2.observed[["condition", "outcome"]].assign(ate=0.0))
print(pd.DataFrame(tbl).T.round(3).to_string())

# bootstrap CI (cluster by study) for the raw crowd
m = preds(el2).merge(g2.observed, on=["condition", "outcome"])
raw_pred = preds(el2)
obs_with_study = g2.observed.copy()
ci = bootstrap_ci(raw_pred, obs_with_study.drop(columns=["study_id"]), g2.ranges,
                  n_boot=1000)
print("\nraw-crowd bootstrap 95% CIs (over cells):")
print(ci.to_string())

# ---- 2. sign split on ground 2 (the H3 test) ----
mm = raw_pred.merge(g2.observed, on=["condition", "outcome"], suffixes=("_p", "_o"))
mm["p"] = mm.apply(lambda r: r["ate_p"] * 100 / g2.ranges[r["outcome"]], axis=1)
mm["o"] = mm.apply(lambda r: r["ate_o"] * 100 / g2.ranges[r["outcome"]], axis=1)
print("\n=== sign split by OBSERVED effect (raw crowd) ===")
for lab, grp in mm.groupby(np.sign(mm["o"]).map({1.0: "obs POSITIVE", -1.0: "obs NEGATIVE", 0.0: "obs ZERO"})):
    print(f"{lab}: n={len(grp)}  mean obs={grp['o'].mean():+.2f}pp  mean pred={grp['p'].mean():+.2f}pp  "
          f"directional={np.mean(np.sign(grp['p']) == np.sign(grp['o'])) * 100:.0f}%  "
          f"MAE={np.mean(np.abs(grp['p'] - grp['o'])):.2f}pp  "
          f"RMSE={np.sqrt(np.mean((grp['p'] - grp['o']) ** 2)):.2f}pp")
neg_pred = (mm["p"] < 0).mean() * 100
neg_obs = (mm["o"] < 0).mean() * 100
print(f"positivity bias: observed negative {neg_obs:.0f}% of cells; predicted negative {neg_pred:.0f}%")

# The clean positivity-bias test: genuine named controls only. The 45
# hypothesis-designated contrasts compare two ACTIVE messages (e.g. pro vs con),
# where a negative ATE is the predictable mirror of the design, not a backfire.
from silicon.retro_hewitt import SLICE, scoped

contrasts_meta = pd.read_csv(SLICE / "contrasts.csv", dtype={"study_id": str})
contrasts_meta["condition"] = [scoped(s, c) for s, c in
                              zip(contrasts_meta["study_id"], contrasts_meta["condition"])]
contrasts_meta["outcome"] = contrasts_meta["outcome_key"] + "@" + contrasts_meta["study_id"]
mm2 = mm.merge(contrasts_meta[["condition", "outcome", "reference_type"]],
               on=["condition", "outcome"], how="left")
nc = mm2[mm2["reference_type"] == "named_control"]
print(f"\n--- named_control contrasts only (n={len(nc)}) ---")
for lab, grp in nc.groupby(np.sign(nc["o"]).map({1.0: "obs POSITIVE", -1.0: "obs NEGATIVE", 0.0: "obs ZERO"})):
    print(f"{lab}: n={len(grp)}  mean obs={grp['o'].mean():+.2f}pp  mean pred={grp['p'].mean():+.2f}pp  "
          f"directional={np.mean(np.sign(grp['p']) == np.sign(grp['o'])) * 100:.0f}%  "
          f"MAE={np.mean(np.abs(grp['p'] - grp['o'])):.2f}pp")
print(f"named_control positivity: obs-neg {(nc['o'] < 0).mean() * 100:.0f}% vs pred-neg {(nc['p'] < 0).mean() * 100:.0f}%")
two_sided = mm2[mm2["reference_type"] != "named_control"]
if len(two_sided):
    print(f"--- two-sided/hypothesis-designated contrasts (n={len(two_sided)}): "
          f"directional={np.mean(np.sign(two_sided['p']) == np.sign(two_sided['o'])) * 100:.0f}%  "
          f"(negatives here are design mirrors, not backfires)")

# H4 NOTE: the archive's hypothesized_direction is contrast ORIENTATION (93/96
# coded rows are +1), not a verified sign prediction -> the prospective sign-
# prior source assumed in the pre-analysis note does not exist for this ground.
# The comparison below is therefore near-vacuous; H4 is NOT TESTABLE AS DESIGNED.
flat_priors = {k: abs(v) for k, v in g2.priors.items()}
for lab, pri in [("flat-positive priors", flat_priors), ("hypothesis-sign priors", g2.priors)]:
    pr = preds(el2, priors=pri, w=0.75, lam=1.0, trim=0.0)
    x = pr.merge(g2.observed, on=["condition", "outcome"], suffixes=("_p", "_o"))
    x["pp_p"] = x.apply(lambda r: r["ate_p"] * 100 / g2.ranges[r["outcome"]], axis=1)
    x["pp_o"] = x.apply(lambda r: r["ate_o"] * 100 / g2.ranges[r["outcome"]], axis=1)
    neg = x[x["pp_o"] < 0]
    print(f"H4 {lab}: negative-cell directional = "
          f"{np.mean(np.sign(neg['pp_p']) == np.sign(neg['pp_o'])) * 100:.0f}%  (n={len(neg)})")

# ---- 3. extended grid on ground 2 + cross-ground transfer ----
print("\n=== extended grid on ground 2, top 8 by RMSE ===")
gs2 = grid_search([g2])
print(gs2.sort_values("rmse_pp").head(8)[
    ["w_prior", "lam", "trim", "pearson_r", "directional_pct", "mae_pp", "rmse_pp", "calib_beta"]
].round(3).to_string(index=False))

print("\n=== cross-ground: combos near-top on BOTH grounds (mean RMSE rank) ===")
g1 = vlasceanu_ground(RUNS_DIR / "retro_vlasceanu_elicitations.csv")
gs_both = grid_search([g1, g2])
gs_both["rmse_rank"] = gs_both.groupby("ground")["rmse_pp"].rank()
agg = gs_both.groupby(["w_prior", "lam", "trim"]).agg(
    mean_rmse_rank=("rmse_rank", "mean"),
    min_r=("pearson_r", "min"),
    mean_dir=("directional_pct", "mean"),
).reset_index().sort_values("mean_rmse_rank")
print(agg.head(8).round(3).to_string(index=False))

# ---- 4. hypothesis verdicts ----
print("\n=== pre-analysis hypothesis verdicts (see reference/retro2_preanalysis.md) ===")
raw_s = row(preds(el2))
froz = tbl["frozen best-r (w=.75,lam=.3,trim=.2)"]
neg = mm[mm["o"] < 0]
pos = mm[mm["o"] > 0]
dir_neg = np.mean(np.sign(neg["p"]) == np.sign(neg["o"])) * 100 if len(neg) else np.nan
dir_pos = np.mean(np.sign(pos["p"]) == np.sign(pos["o"])) * 100 if len(pos) else np.nan
best_rmse2 = gs2.sort_values("rmse_pp").iloc[0]
print(f"H1 transfer (frozen-r beats raw on pooled r): {froz['pearson_r']:.3f} vs {raw_s['pearson_r']:.3f} "
      f"-> {'SUPPORTED' if froz['pearson_r'] > raw_s['pearson_r'] else 'NOT SUPPORTED'}")
print(f"H2 raw pooled r in 0.35-0.65: {raw_s['pearson_r']:.3f} "
      f"-> {'SUPPORTED' if 0.35 <= raw_s['pearson_r'] <= 0.65 else 'NOT SUPPORTED'}")
print(f"H3 positivity bias (neg dir <40%, pos dir >85%, pred-neg rate < half obs-neg rate): "
      f"neg {dir_neg:.0f}%, pos {dir_pos:.0f}%, pred-neg {neg_pred:.0f}% vs obs-neg {neg_obs:.0f}% "
      f"-> {'SUPPORTED' if (dir_neg < 40 and dir_pos > 85 and neg_pred < neg_obs / 2) else 'NOT SUPPORTED'}"
      f"  [check the named_control cut above: two-sided designs make pooled negatives easier]")
print("H4 -> NOT TESTABLE AS DESIGNED: hypothesized_direction is contrast orientation "
      "(93/96 rows +1), not a sign prediction; no prospective sign source exists for this ground.")
print(f"H5 calib beta in 1.3-3.0: {raw_s['calib_beta']:.2f} "
      f"-> {'SUPPORTED' if 1.3 <= raw_s['calib_beta'] <= 3.0 else 'NOT SUPPORTED'}")
print(f"H6 ground-2 RMSE optimum has lam >= 1: lam={best_rmse2['lam']} "
      f"-> {'SUPPORTED' if best_rmse2['lam'] >= 1.0 else 'NOT SUPPORTED'}")
