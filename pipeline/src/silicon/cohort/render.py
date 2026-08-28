"""T1 stage S3: map latent scores to observed responses (0-100 heaped sliders,
composite means with correct support, discrete donation, binary signup).

Baseline targets (control-group means) come from reference/outcome_priors.md;
SDs anchored to ANES thermometer (~20, widened for climate polarization since
partisan offsets already inject much of the variance). Heaping calibrated to
the ANES 2016 regime (softer label-heaping; benchmark sliders are unlabeled):
target ~80% on multiples of 5, ~25% on {0,50,100}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (baseline mean, marginal SD) on 0-100; from outcome_priors.md baseline table
SCALE_TARGETS: dict[str, tuple[float, float]] = {
    "trust": (65.0, 21.0),          # 12 trust items family + trust_post uses own mean below
    "distrust": (32.0, 26.0),
    "funding": (58.0, 24.0),
    "policy_role": (58.0, 23.0),
    "inst_trust": (55.0, 21.0),
    "belief": (66.0, 30.0),         # US belief SD large (Voelkel control SD 22.5-32)
    "concern": (58.0, 29.0),
    "policy_general": (66.0, 28.0),
    "policy_specific": (63.0, 24.0),
    "behavior": (48.0, 25.0),
}
TRUST_POST_MEAN = 64.0
TRUST_ITEM_MEAN = 66.0  # single items slightly above 12-item composite mean

# Heaping kernel (2016 regime), tuned in build loop: probabilities that a raw
# value snaps to nearest 10 / nearest 5 / stays integer; plus label attraction.
HEAP_P10 = 0.42
HEAP_P5 = 0.38
LABEL_PULL = 0.13   # extra chance of snapping to nearest of {0,50,100} if within 6
ITEM_NOISE_SD = 6.0  # extra integer-item roughness for composite items

# per-item mean offsets inside composites (points on 0-100)
INST_ITEM_OFFSETS = [3.0, 10.0, 8.0, 4.0, -22.0]  # EPA, NASA, NOAA, universities, federal gov
POLICY_SPEC_OFFSETS = [-10.0, 2.0, 8.0, 12.0, -14.0, 6.0, 14.0]  # taxes..waterways
BEHAVIOR_OFFSETS = [-6.0, -2.0, -16.0, -8.0, 8.0, -4.0]  # meat, transport, solar, fly, talk, donateNGO
CONCERN_OFFSETS = [2.0, 4.0, -6.0]
POLICY_ROLE_OFFSETS = [4.0, -10.0, 8.0, -2.0]  # advocate item lower (Kotcher)


def heap(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Integer-snap with 2016-regime heaping."""
    v = np.clip(values, 0, 100)
    u = rng.random(v.shape)
    out = np.rint(v)
    out = np.where(u < HEAP_P10, np.rint(v / 10) * 10, out)
    both = (u >= HEAP_P10) & (u < HEAP_P10 + HEAP_P5)
    out = np.where(both, np.rint(v / 5) * 5, out)
    # label attraction to 0/50/100
    for lab in (0.0, 50.0, 100.0):
        near = np.abs(v - lab) <= 6
        pull = rng.random(v.shape) < LABEL_PULL
        out = np.where(near & pull, lab, out)
    return np.clip(out, 0, 100)


def render_slider(z: np.ndarray, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    return heap(mean + sd * z, rng)


def render_composite(z: np.ndarray, mean: float, sd: float, offsets: list[float],
                     rng: np.random.Generator) -> np.ndarray:
    """k integer items from one latent -> composite mean with correct support."""
    items = []
    for off in offsets:
        raw = mean + off + sd * z + rng.standard_normal(len(z)) * ITEM_NOISE_SD
        items.append(heap(raw, rng))
    return np.mean(items, axis=0)


def render_all(profiles: pd.DataFrame, z: pd.DataFrame, trust_items_z: pd.DataFrame,
               seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=profiles.index)

    # 12 trust items (0-100 each, heaped), composite = their mean
    t_mean, t_sd = TRUST_ITEM_MEAN, SCALE_TARGETS["trust"][1]
    for col in trust_items_z.columns:
        out[col] = render_slider(trust_items_z[col].values, t_mean, t_sd, rng)
    subs = ["competence", "integrity", "benevolence", "openness"]
    sub_means = {s: out[[f"trust_{s}_{i}" for i in (1, 2, 3)]].mean(axis=1) for s in subs}
    out["trust_multidimensional"] = np.mean([sub_means[s] for s in subs], axis=0)

    out["trust_post"] = render_slider(z["trust"].values * 0.9 + rng.standard_normal(len(z)) * 0.45,
                                      TRUST_POST_MEAN, 22.0, rng)
    out["distrust_post"] = render_slider(z["distrust"].values, *SCALE_TARGETS["distrust"], rng)

    # funding: generate raw funding_5 reversed (higher=too much), heap at 50, recode
    fund_raw = 100 - (SCALE_TARGETS["funding"][0] + SCALE_TARGETS["funding"][1] * z["funding"].values)
    fund5 = heap(fund_raw + 0, rng)
    # extra 'about right' heap at 50
    mid_pull = rng.random(len(z)) < 0.12
    fund5 = np.where(mid_pull & (np.abs(fund5 - 50) < 15), 50, fund5)
    out["funding_perceptions"] = 100 - fund5

    out["policy_role_mean"] = render_composite(z["policy_role"].values, *SCALE_TARGETS["policy_role"],
                                               POLICY_ROLE_OFFSETS, rng)
    out["inst_trust_mean"] = render_composite(z["inst_trust"].values, *SCALE_TARGETS["inst_trust"],
                                              INST_ITEM_OFFSETS, rng)
    out["belief_post"] = render_slider(z["belief"].values, *SCALE_TARGETS["belief"], rng)
    out["concern_mean"] = render_composite(z["concern"].values, *SCALE_TARGETS["concern"],
                                           CONCERN_OFFSETS, rng)
    out["policy_general"] = render_slider(z["policy_general"].values, *SCALE_TARGETS["policy_general"], rng)
    out["policy_specific_mean"] = render_composite(z["policy_specific"].values,
                                                   *SCALE_TARGETS["policy_specific"],
                                                   POLICY_SPEC_OFFSETS, rng)
    out["behavior_mean"] = render_composite(z["behavior"].values, *SCALE_TARGETS["behavior"],
                                            BEHAVIOR_OFFSETS, rng)

    # donation: two-part; P(0)~45%, positive part modes at 5/10, mean ~$1.85
    zd = z["donation"].values
    p_zero = 1 / (1 + np.exp(0.20 + 0.9 * zd))     # higher generosity -> less zero
    is_zero = rng.random(len(zd)) < p_zero
    pos_raw = np.clip(2.4 + 2.2 * zd + rng.standard_normal(len(zd)) * 1.8, 1, 10)
    pos = np.rint(pos_raw)
    five_pull = rng.random(len(zd)) < 0.28
    pos = np.where(five_pull & (np.abs(pos_raw - 5) <= 2.4), 5, pos)
    ten_pull = rng.random(len(zd)) < 0.5
    pos = np.where((pos_raw > 8.4) & ten_pull, 10, pos)
    out["donation_ams"] = np.where(is_zero, 0, pos).astype(int)

    # newsletter: Bernoulli, marginal ~9%
    zn = z["newsletter"].values
    p = 1 / (1 + np.exp(-(-2.45 + 0.75 * zn)))
    out["newsletter_signup"] = (rng.random(len(zn)) < p).astype(int)
    return out
