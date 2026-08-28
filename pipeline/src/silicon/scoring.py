"""Self-evaluation metrics mirroring the benchmark preregistration's Section 1.

Everything is converted to percentage points of each outcome's scale range
before pooling (the prereg's pp convention: sliders /1, donation /10 * ... —
i.e. pp = ate * 100 / range). Used to score our pipeline on retrodiction
studies and for internal ablations; the organizers' pipeline is authoritative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def to_pp(df: pd.DataFrame, ranges: dict[str, float], col: str) -> pd.Series:
    """ate -> percentage points of scale range: ate * 100 / range."""
    return df.apply(lambda r: r[col] * 100.0 / ranges[r["outcome"]], axis=1)


def score(
    pred: pd.DataFrame,  # columns: condition, outcome, ate
    obs: pd.DataFrame,   # columns: condition, outcome, ate [, se]
    ranges: dict[str, float],  # outcome -> scale range (100 for 0-100 sliders, 10 for $0-10, 1 for 0/1)
) -> dict[str, float]:
    m = pred.merge(obs, on=["condition", "outcome"], suffixes=("_p", "_o"))
    if m.empty:
        raise ValueError("no overlapping (condition, outcome) cells")
    p = to_pp(m, ranges, "ate_p")
    o = to_pp(m, ranges, "ate_o")

    # directional agreement with the prereg's half-credit rule for exact zeros
    sign_scores = np.where(
        (p == 0) | (o == 0), 0.5, (np.sign(p) == np.sign(o)).astype(float)
    )

    # within-outcome r: demean both sides per outcome, then pool (message-level skill)
    dm = m.assign(p=p, o=o)
    dm["p_c"] = dm["p"] - dm.groupby("outcome")["p"].transform("mean")
    dm["o_c"] = dm["o"] - dm.groupby("outcome")["o"].transform("mean")

    out = {
        "n_pairs": int(len(m)),
        "pearson_r": float(pearsonr(p, o)[0]) if len(m) > 2 else np.nan,
        "spearman_rho": float(spearmanr(p, o)[0]) if len(m) > 2 else np.nan,
        "within_outcome_r": float(pearsonr(dm["p_c"], dm["o_c"])[0]) if len(m) > 2 else np.nan,
        "directional_pct": float(np.mean(sign_scores) * 100),
        "rmse_pp": float(np.sqrt(np.mean((p - o) ** 2))),
        "mae_pp": float(np.mean(np.abs(p - o))),
    }
    # calibration regression obs = a + b * pred (b < 1 means predictions exaggerate)
    if len(m) > 2 and p.std() > 0:
        b, a = np.polyfit(p, o, 1)
        out["calib_beta"] = float(b)
        out["calib_alpha_pp"] = float(a)
    # disattenuated r when observed SEs are available
    if "se" in obs.columns and len(m) > 2:
        se_pp = to_pp(m.rename(columns={"se": "ate_se"}).assign(ate_se=m["se"]), ranges, "ate_se")
        var_true = o.var(ddof=1) - float(np.mean(se_pp**2))
        if var_true > 0:
            out["r_adj"] = float(np.clip(
                np.cov(p, o, ddof=1)[0, 1] / (p.std(ddof=1) * np.sqrt(var_true)), -1, 1
            ))
    return out


def score_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.round(3)


def bootstrap_ci(
    pred: pd.DataFrame,
    obs: pd.DataFrame,
    ranges: dict[str, float],
    metrics: tuple[str, ...] = ("pearson_r", "mae_pp", "rmse_pp", "directional_pct"),
    n_boot: int = 2000,
    cluster_col: str | None = None,  # e.g. "study" for cluster bootstrap over studies
    seed: int = 7,
) -> pd.DataFrame:
    """Percentile bootstrap CIs over cells (or clusters of cells)."""
    m = pred.merge(obs, on=["condition", "outcome"], suffixes=("_p", "_o"))
    rng = np.random.default_rng(seed)
    clusters = m[cluster_col].unique() if cluster_col else m.index.to_numpy()
    draws: dict[str, list[float]] = {k: [] for k in metrics}
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        if cluster_col:
            sample = pd.concat([m[m[cluster_col] == c] for c in pick])
        else:
            sample = m.loc[pick]
        try:
            s = score(
                sample[["condition", "outcome", "ate_p"]].rename(columns={"ate_p": "ate"}),
                sample[["condition", "outcome", "ate_o"]].rename(columns={"ate_o": "ate"}),
                ranges,
            )
        except ValueError:
            continue
        for k in metrics:
            if k in s and s[k] == s[k]:
                draws[k].append(s[k])
    rows = {
        k: {"lo95": float(np.percentile(v, 2.5)), "hi95": float(np.percentile(v, 97.5))}
        for k, v in draws.items() if v
    }
    return pd.DataFrame(rows).T.round(3)
