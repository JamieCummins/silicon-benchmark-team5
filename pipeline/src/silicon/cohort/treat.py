"""T1 stage S4: inject the T3-A effect vector as latent shifts.

Treatment enters as per-outcome-channel additive shifts on the render-stage
latent inputs (baseline correlations come from the shared z; treatment only
moves means). Channel shift sizes are auto-calibrated by simulation: a probe
shift on the control group measures each channel's observed-scale slope
(points per latent unit, through heaping/clipping/two-part models), then
dz(condition, outcome) = ATE / slope. Within-arm heterogeneity: headroom
weighting (people far from ceiling move more) + person noise, mean-preserving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .latents import TRUST_ITEM_COLS
from .render import render_all

# channel -> (T3 outcome key, direction where +ATE means channel latent up)
CHANNELS = {
    "trust": "trust_multidimensional",
    "trust_post": "trust_post",
    "distrust": "distrust_post",
    "funding": "funding_perceptions",
    "policy_role": "policy_role_mean",
    "inst_trust": "inst_trust_mean",
    "belief": "belief_post",
    "concern": "concern_mean",
    "policy_general": "policy_general",
    "policy_specific": "policy_specific_mean",
    "behavior": "behavior_mean",
    "donation": "donation_ams",
    "newsletter": "newsletter_signup",
}

PROBE = 0.15
HETERO_NOISE = 0.8  # person-level noise SD as multiple of |dz|


def _apply_shifts(z: pd.DataFrame, items_z: pd.DataFrame,
                  shifts: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    z2 = z.copy()
    items2 = items_z.copy()
    for ch, dz in shifts.items():
        if ch == "trust":
            for col in TRUST_ITEM_COLS:
                items2[col] = items2[col] + dz
        elif ch == "trust_post":
            z2["trust"] = z2["trust"]  # trust_post shift handled via its own channel below
        if ch in z2.columns:
            z2[ch] = z2[ch] + dz
    return z2, items2


def calibrate_slopes(profiles: pd.DataFrame, z: pd.DataFrame, items_z: pd.DataFrame,
                     render_seed: int = 13) -> dict[str, float]:
    """Observed-scale points per latent unit, per channel, via probe shifts."""
    base = render_all(profiles, z, items_z, seed=render_seed)
    slopes: dict[str, float] = {}
    for ch, key in CHANNELS.items():
        dz = np.full(len(z), PROBE)
        if ch == "trust_post":
            z_p = z.copy()
            z_p["trust"] = z_p["trust"] + dz  # trust_post renders from z['trust']
            r = render_all(profiles, z_p, items_z, seed=render_seed)
        else:
            z_p, items_p = _apply_shifts(z, items_z, {ch: dz})
            r = render_all(profiles, z_p, items_p, seed=render_seed)
        slopes[ch] = float((r[key].mean() - base[key].mean()) / PROBE)
    return slopes


def treatment_shifts(profiles: pd.DataFrame, z: pd.DataFrame, ates: pd.DataFrame,
                     slopes: dict[str, float], seed: int = 17) -> dict[str, np.ndarray]:
    """Per-channel person-level dz arrays implementing the T3-A vector."""
    rng = np.random.default_rng(seed)
    ate = ates.set_index(["condition", "outcome"])["ate"]
    shifts: dict[str, np.ndarray] = {}
    is_treated = (profiles["condition"] != "control").values

    for ch, key in CHANNELS.items():
        dz = np.zeros(len(profiles))
        zsrc = z["trust"] if ch == "trust_post" else z[ch if ch in z.columns else "trust"]
        # headroom: high-latent people are near ceiling -> move less (reverse for distrust)
        cdf = norm.cdf(zsrc.values)
        head = 2 * (cdf if ch == "distrust" else (1 - cdf))
        for cond in profiles["condition"].unique():
            if cond == "control":
                continue
            mask = (profiles["condition"] == cond).values
            target = ate.get((cond, key), 0.0)
            base_dz = target / slopes[ch]
            h = head[mask] / head[mask].mean()
            noise = rng.standard_normal(mask.sum()) * abs(base_dz) * HETERO_NOISE
            noise -= noise.mean()
            dz[mask] = base_dz * h + noise
        shifts[ch] = dz
    assert not shifts["trust"][~is_treated].any()
    return shifts


def render_with_treatment(profiles: pd.DataFrame, z: pd.DataFrame, items_z: pd.DataFrame,
                          shifts: dict[str, np.ndarray], render_seed: int = 13) -> pd.DataFrame:
    z2 = z.copy()
    items2 = items_z.copy()
    for ch, dz in shifts.items():
        if ch == "trust":
            for col in TRUST_ITEM_COLS:
                items2[col] = items2[col] + dz
        elif ch == "trust_post":
            z2["trust"] = z2["trust"] + dz
        else:
            z2[ch] = z2[ch] + dz
    return render_all(profiles, z2, items2, seed=render_seed)
