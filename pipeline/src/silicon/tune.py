"""Retrodiction hyperparameter tuning: grid-search the aggregation layer on
public studies where observed ATEs are known, with leave-one-study-out checks.

Elicitation happens once per retrodiction study (cached in its CSV); tuning
re-aggregates the cached predictions offline, so the grid costs nothing.

A retrodiction ground is registered as:
    Ground(study_id, elicitations_csv, observed (condition,outcome,ate[,se]),
           ranges (outcome -> scale range), priors (outcome -> prior mean ATE))
Study-specific priors are needed because PRIORS in aggregate.py is
benchmark-specific; the *hyperparameters* (w_prior, lam, trim) are what must
transfer across grounds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import aggregate as agg
from .scoring import score


@dataclass
class Ground:
    study_id: str
    elicitations_csv: Path
    observed: pd.DataFrame  # condition, outcome, ate [, se]
    ranges: dict[str, float]
    priors: dict[str, float]
    epsilon: dict[str, float]


def aggregate_with(
    elicitations: pd.DataFrame,
    priors: dict[str, float],
    epsilon: dict[str, float],
    w_prior: float,
    lam: float,
    trim: float,
) -> pd.DataFrame:
    """aggregate.calibrate with ground-specific priors (monkeypatch-free)."""
    saved_p, saved_e = agg.PRIORS, agg.EPSILON
    try:
        agg.PRIORS, agg.EPSILON = priors, epsilon
        cells = agg.cell_estimates(elicitations, trim=trim)
        return agg.calibrate(cells, w_prior=w_prior, lam=lam)
    finally:
        agg.PRIORS, agg.EPSILON = saved_p, saved_e


GRID = {
    "w_prior": [0.0, 0.25, 0.5, 0.75],
    # lam > 1 = amplification: run 1 showed the raw crowd UNDERpredicts (~3x),
    # so the error-optimal direction is amplify, not shrink.
    "lam": [0.3, 0.6, 1.0, 1.5, 2.0, 3.0],
    "trim": [0.0, 0.1, 0.2],
}


def grid_search(
    grounds: list[Ground],
    grid: dict[str, list[float]] | None = None,
    metric: str = "pearson_r",
) -> pd.DataFrame:
    """Score every hyperparameter combo on every ground. Returns long dataframe;
    look for combos that are near-top on ALL grounds (transfer), not the argmax
    of a single one."""
    grid = grid or GRID
    rows = []
    cached = {g.study_id: pd.read_csv(g.elicitations_csv) for g in grounds}
    for w, lam, trim in itertools.product(grid["w_prior"], grid["lam"], grid["trim"]):
        for g in grounds:
            final = aggregate_with(cached[g.study_id], g.priors, g.epsilon, w, lam, trim)
            s = score(final[["condition", "outcome", "ate"]], g.observed, g.ranges)
            rows.append({"ground": g.study_id, "w_prior": w, "lam": lam, "trim": trim, **s})
    df = pd.DataFrame(rows)
    # rank within ground by the target metric; low mean rank across grounds = transfers
    df["rank"] = df.groupby("ground")[metric].rank(ascending=False)
    return df


def model_subset_ablation(
    grounds: list[Ground], w_prior: float, lam: float, trim: float
) -> pd.DataFrame:
    """Leave-one-model-out: which crowd members earn their place?"""
    rows = []
    for g in grounds:
        el = pd.read_csv(g.elicitations_csv)
        models = sorted(el["model"].unique())
        for drop in [None, *models]:
            sub = el if drop is None else el[el["model"] != drop]
            final = aggregate_with(sub, g.priors, g.epsilon, w_prior, lam, trim)
            s = score(final[["condition", "outcome", "ate"]], g.observed, g.ranges)
            rows.append({"ground": g.study_id, "dropped": drop or "-", **s})
    return pd.DataFrame(rows)
