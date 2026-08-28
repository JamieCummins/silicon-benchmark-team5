"""T1 stage S2: person-level latent construct scores.

Each person gets a z-vector over 13 constructs (one per outcome; donation and
newsletter get propensity latents), drawn from MVN(0, R) and shifted by
party/demographic offsets. The 12 trust items additionally get item-level
structure (shared trust factor -> subscale -> item noise) calibrated to TISP.

PROVISIONAL_* values are placeholders wired to real calibration targets
(data/derived/calibration_targets/) as they land; each is annotated with its
target source.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import DATA_DIR

TARGETS = DATA_DIR / "derived" / "calibration_targets"

CONSTRUCTS = [
    "trust",            # drives 12 items + trust_post
    "distrust",         # separate construct, strongly negative r with trust
    "funding",
    "policy_role",
    "inst_trust",
    "belief",
    "concern",
    "policy_general",
    "policy_specific",
    "behavior",
    "donation",         # generosity propensity
    "newsletter",       # interest/click propensity
]

# Construct correlations. Sources: TISP US (trust x clim items .72, trust x
# policy .45), CCAM (belief x worry .63, worry x policy .64), plausibility
# fill elsewhere (flagged: refine against ccam_corr.csv / tisp corr on arrival).
_R_SPEC: dict[tuple[str, str], float] = {
    ("trust", "distrust"): -0.70,
    ("trust", "funding"): 0.50,
    ("trust", "policy_role"): 0.55,
    ("trust", "inst_trust"): 0.60,
    ("trust", "belief"): 0.65,       # TISP trust12 x CLIM_TRUST = .72; belief a step out
    ("trust", "concern"): 0.55,
    ("trust", "policy_general"): 0.50,
    ("trust", "policy_specific"): 0.45,  # TISP trust x policy support = .45
    ("trust", "behavior"): 0.35,
    ("trust", "donation"): 0.25,
    ("trust", "newsletter"): 0.25,
    ("belief", "concern"): 0.70,     # CCAM happening x worry .63, inflate for slider vs cat
    ("concern", "policy_general"): 0.65,  # CCAM worry x policy .64
    ("belief", "policy_general"): 0.55,
    ("policy_general", "policy_specific"): 0.75,
    ("concern", "behavior"): 0.50,
    ("behavior", "donation"): 0.30,
    ("concern", "newsletter"): 0.30,
    ("distrust", "belief"): -0.50,
    ("distrust", "inst_trust"): -0.50,
}
_DEFAULT_R = 0.35  # generic attitude-attitude fill
_BEHAV = {"donation", "newsletter", "behavior"}


def correlation_matrix() -> pd.DataFrame:
    k = len(CONSTRUCTS)
    R = np.full((k, k), np.nan)
    np.fill_diagonal(R, 1.0)
    for i, a in enumerate(CONSTRUCTS):
        for j, b in enumerate(CONSTRUCTS):
            if i >= j:
                continue
            v = _R_SPEC.get((a, b), _R_SPEC.get((b, a)))
            if v is None:
                v = 0.2 if (a in _BEHAV or b in _BEHAV) else _DEFAULT_R
                if "distrust" in (a, b):
                    v = -v
            R[i, j] = R[j, i] = v
    R = pd.DataFrame(R, index=CONSTRUCTS, columns=CONSTRUCTS)
    # nearest PSD via eigenvalue clipping
    w, V = np.linalg.eigh(R.values)
    if w.min() < 1e-8:
        w = np.clip(w, 1e-8, None)
        M = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(M))
        R = pd.DataFrame(M / np.outer(d, d), index=CONSTRUCTS, columns=CONSTRUCTS)
    return R


# Party offsets in SD units (Dem/Rep/Ind/Other), per construct. Anchors: ANES
# 2020 scientists-thermometer gap 16pts/SD20 = 0.8 SD, widened for climate
# specificity; CCAM party gaps on belief/worry/policy ~1.0-1.2 SD.
# Refine against thermometer_anes.json + ccam_targets.csv on arrival.
PROVISIONAL_PARTY_OFFSETS: dict[str, dict[str, float]] = {
    # construct: {Democrat, Republican, Independent, Other} (mean-zero-ish after mixing)
    "trust":          {"Democrat": 0.62, "Republican": -0.73, "Independent": -0.05, "Other": -0.15},
    "distrust":       {"Democrat": -0.56, "Republican": 0.67, "Independent": 0.05, "Other": 0.15},
    "funding":        {"Democrat": 0.55, "Republican": -0.60, "Independent": -0.05, "Other": -0.10},
    "policy_role":    {"Democrat": 0.55, "Republican": -0.60, "Independent": -0.05, "Other": -0.10},
    "inst_trust":     {"Democrat": 0.45, "Republican": -0.50, "Independent": -0.05, "Other": -0.10},
    "belief":         {"Democrat": 0.60, "Republican": -0.70, "Independent": -0.05, "Other": -0.10},
    "concern":        {"Democrat": 0.60, "Republican": -0.70, "Independent": -0.05, "Other": -0.10},
    "policy_general": {"Democrat": 0.60, "Republican": -0.70, "Independent": -0.05, "Other": -0.10},
    "policy_specific":{"Democrat": 0.55, "Republican": -0.60, "Independent": -0.05, "Other": -0.10},
    "behavior":       {"Democrat": 0.40, "Republican": -0.45, "Independent": -0.05, "Other": -0.05},
    "donation":       {"Democrat": 0.20, "Republican": -0.20, "Independent": 0.0, "Other": 0.0},
    "newsletter":     {"Democrat": 0.25, "Republican": -0.25, "Independent": 0.0, "Other": 0.0},
}

# Secondary demographic offsets (SD units), smaller: education gradient on
# trust/belief, age on behavior, gender on concern. Refine vs targets.
EDU_TRUST_GRADIENT = {  # applies to trust/inst_trust/belief at half strength elsewhere
    "Less than high school": -0.15, "High school diploma / GED": -0.10,
    "Some college or Associate's degree": 0.0, "Bachelor's degree": 0.10,
    "Master's degree / Professional degree": 0.15, "Doctorate degree / Ph.D.": 0.20,
}
GENDER_CONCERN = {"Male": -0.10, "Female": 0.10, "Other": 0.10}
AGE_BEHAVIOR = {"18-29": 0.15, "30-44": 0.05, "45-59": -0.05, "60+": -0.10}


def draw_latents(profiles: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(profiles)
    R = correlation_matrix()
    L = np.linalg.cholesky(R.values)
    z = rng.standard_normal((n, len(CONSTRUCTS))) @ L.T
    z = pd.DataFrame(z, columns=CONSTRUCTS, index=profiles.index)

    for c in CONSTRUCTS:
        z[c] += profiles["party"].map(PROVISIONAL_PARTY_OFFSETS[c]).values
    for c in ("trust", "inst_trust", "belief"):
        z[c] += profiles["education"].map(EDU_TRUST_GRADIENT).values
    z["concern"] += profiles["gender"].map(GENDER_CONCERN).values
    z["behavior"] += profiles["age_band"].map(AGE_BEHAVIOR).values
    return z


# --- 12 trust items ------------------------------------------------------------

TRUST_ITEM_COLS = [
    f"trust_{sub}_{i}" for sub in ("competence", "integrity", "benevolence", "openness")
    for i in (1, 2, 3)
]

# Item structure targets: within-battery mean inter-item r from TISP US
# (refine vs trust_items_tisp_corr.csv). loading^2 + sub^2 + noise^2 = 1.
PROVISIONAL_ITEM_R_WITHIN_SUB = 0.65
PROVISIONAL_ITEM_R_BETWEEN_SUB = 0.50


def draw_trust_items(z_trust: pd.Series, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(z_trust)
    lam2 = PROVISIONAL_ITEM_R_BETWEEN_SUB          # shared-factor variance share
    sub2 = PROVISIONAL_ITEM_R_WITHIN_SUB - lam2    # subscale-specific share
    noise2 = 1 - lam2 - sub2
    lam, sub_sd, noise_sd = np.sqrt(lam2), np.sqrt(sub2), np.sqrt(noise2)

    out = {}
    for s, sub in enumerate(("competence", "integrity", "benevolence", "openness")):
        sub_f = rng.standard_normal(n) * sub_sd
        for i in (1, 2, 3):
            out[f"trust_{sub}_{i}"] = lam * z_trust.values + sub_f + rng.standard_normal(n) * noise_sd
    return pd.DataFrame(out, index=z_trust.index)
