"""T3-A aggregation: crowd elicitations -> calibrated ATE vector.

Pipeline: per-cell trimmed mean over model x variant predictions
        -> outcome-level centering toward priors (blend weight w_prior)
        -> within-outcome shrinkage of deviations (factor lam)
        -> signed-epsilon rule (never submit an exact zero: zeros score half
           credit on directional agreement, a signed epsilon scores full/none)
        -> template-format CSV (condition,outcome,ate).

Hyperparameters and priors are PLACEHOLDERS pending retrodiction tuning; they
encode the strategy's directional priors (attitudinal small, behavioral ~zero)
at conservative magnitudes. Pearson r (the leaderboard metric) is unaffected by
centering/scaling within outcomes only if applied globally — outcome-level
centering DOES change pooled r, which is intended: it injects the between-
outcome effect profile we believe more than the raw crowd's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from .materials import scored_labels
from .outcomes import OUTCOME_KEYS

# Prior mean ATE per outcome, native units. Literature-synthesized and
# signed off by Jamie 2026-08-18 — see reference/outcome_priors.md for the
# full derivation, anchors, and baseline table. Do not edit casually.
PRIORS: dict[str, float] = {
    "trust_multidimensional": 1.5,
    "trust_post": 1.7,
    "distrust_post": -1.1,
    "funding_perceptions": 0.8,
    "policy_role_mean": 1.0,
    "inst_trust_mean": 0.7,
    "belief_post": 0.8,
    "concern_mean": 0.7,
    "policy_general": 0.6,
    "policy_specific_mean": 0.4,
    "behavior_mean": 0.4,
    "donation_ams": 0.04,
    "newsletter_signup": 0.006,
}

# Smallest submitted magnitude per outcome (signed-epsilon rule).
EPSILON: dict[str, float] = {
    **{k: 0.02 for k in OUTCOME_KEYS},
    "donation_ams": 0.002,
    "newsletter_signup": 0.0002,
}

# Adopted from cross-ground retrodiction (near-top on BOTH grounds by RMSE;
# see runs/retro_tuned_params.json and reference/retro2_preanalysis.md):
DEFAULT_W_PRIOR = 0.5  # weight on prior vs crowd for the outcome-level mean
DEFAULT_LAM = 1.0      # within-outcome deviations pass through unshrunk
DEFAULT_TRIM = 0.2     # trimmed-mean fraction per side across crowd members


def cell_estimates(elicitations: pd.DataFrame, trim: float = DEFAULT_TRIM) -> pd.DataFrame:
    """(intervention, outcome) -> trimmed mean + spread across model x variant."""
    g = elicitations.groupby(["intervention", "outcome"])["ate"]
    out = g.apply(lambda s: trim_mean(s, trim)).rename("crowd").reset_index()
    out["crowd_sd"] = g.std().values
    out["n_preds"] = g.count().values
    return out


def calibrate(
    cells: pd.DataFrame,
    w_prior: float = DEFAULT_W_PRIOR,
    lam: float = DEFAULT_LAM,
) -> pd.DataFrame:
    rows = []
    for outcome, grp in cells.groupby("outcome"):
        crowd_mean = grp["crowd"].mean()
        center = w_prior * PRIORS[outcome] + (1 - w_prior) * crowd_mean
        for _, r in grp.iterrows():
            ate = center + lam * (r["crowd"] - crowd_mean)
            eps = EPSILON[outcome]
            if abs(ate) < eps:
                sign = np.sign(r["crowd"]) or np.sign(PRIORS[outcome]) or 1.0
                ate = sign * eps
            rows.append({"condition": r["intervention"], "outcome": outcome,
                         "ate": float(ate), "crowd_raw": r["crowd"],
                         "n_preds": r["n_preds"]})
    return pd.DataFrame(rows)


def write_t3_csv(df: pd.DataFrame, path: Path, allow_partial: bool = False) -> None:
    """Write the template-format Tier-3 file (condition,outcome,ate)."""
    expected = {(c, o) for c in scored_labels() for o in OUTCOME_KEYS}
    have = set(zip(df["condition"], df["outcome"]))
    missing = expected - have
    if missing and not allow_partial:
        raise ValueError(f"incomplete grid: {len(missing)} cells missing, e.g. {sorted(missing)[:3]}")
    out = df[df.apply(lambda r: (r["condition"], r["outcome"]) in expected, axis=1)]
    out = out.sort_values(["condition", "outcome"])[["condition", "outcome", "ate"]]
    out["ate"] = out["ate"].round(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} rows -> {path}" + (f"  (PARTIAL: {len(missing)} missing)" if missing else ""))


def diagnostics(cells: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    a = cells.groupby("outcome")["crowd"].agg(crowd_mean="mean", crowd_spread="std")
    b = final.groupby("outcome")["ate"].agg(final_mean="mean", final_spread="std")
    d = a.join(b)
    d["prior"] = pd.Series(PRIORS)
    return d[["prior", "crowd_mean", "final_mean", "crowd_spread", "final_spread"]].round(3)


if __name__ == "__main__":
    from .config import RUNS_DIR

    src = RUNS_DIR / "t3a_elicitations.csv"
    cells = cell_estimates(pd.read_csv(src))
    final = calibrate(cells)
    print(diagnostics(cells, final).to_string())
    write_t3_csv(final, RUNS_DIR / "demo_T3_from_pilot.csv", allow_partial=True)
