"""Build T2 (cells_main + cells_moderator) from the T1 generator components.

Cell values are model-implied and smooth: baselines are FULL-cohort (n=18k)
group means of the deterministic baseline render; treatment cells add the T3-A
ATE scaled by the group's mean headroom factor (slider outcomes; behavioral
outcomes get flat effects per strategy). Moderator cells therefore carry REAL
demographic level differences (reference-calibrated) + principled near-
proportional effect moderation — never exact zeros, never NA.

  uv run python scripts/build_t2.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from silicon.aggregate import calibrate, cell_estimates
from silicon.config import RUNS_DIR
from silicon.cohort.latents import draw_latents, draw_trust_items
from silicon.cohort.profiles import build_profiles
from silicon.cohort.render import render_all
from silicon.cohort.treat import CHANNELS

MODERATORS = ["gender", "age_band", "race", "education", "income", "party"]
FLAT_EFFECT = {"donation_ams", "newsletter_signup"}  # no headroom moderation

# identical seeds/order to build_t1.py -> same cohort
el = pd.read_csv(RUNS_DIR / "t3a_elicitations.csv")
ates = calibrate(cell_estimates(el))[["condition", "outcome", "ate"]]
ate_idx = ates.set_index(["condition", "outcome"])["ate"]
conditions = ["control"] + sorted(ates["condition"].unique())

profiles = build_profiles()
z = draw_latents(profiles)
items_z = draw_trust_items(z["trust"])
cond = profiles["condition"].values
for frame in (z, items_z):
    for col in frame.columns:
        grp = frame.groupby(cond)[col].transform("mean")
        frame[col] = frame[col] - grp + frame[col].mean()

baseline = render_all(profiles, z, items_z)  # deterministic, no treatment

# headroom factor per person x channel (mean 1.0 overall)
head = {}
for ch, key in CHANNELS.items():
    zsrc = z["trust"] if ch == "trust_post" else z[ch]
    cdf = norm.cdf(zsrc.values)
    h = 2 * (cdf if ch == "distrust" else (1 - cdf))
    head[key] = h / h.mean()

outcome_keys = list(CHANNELS.values())

# ---- cells_main: full-cohort baseline mean + ATE ----
rows = []
for c in conditions:
    for key in outcome_keys:
        mean = baseline[key].mean() + (0.0 if c == "control" else ate_idx.get((c, key), 0.0))
        rows.append({"condition": c, "outcome": key, "mean": round(float(mean), 3)})
main = pd.DataFrame(rows)

# ---- cells_moderator ----
rows = []
for mod in MODERATORS:
    for level, idx in profiles.groupby(mod).groups.items():
        base_g = baseline.loc[idx, outcome_keys].mean()
        for c in conditions:
            for key in outcome_keys:
                v = float(base_g[key])
                if c != "control":
                    hr = 1.0 if key in FLAT_EFFECT else float(np.mean(head[key][profiles.index.isin(idx)]))
                    v += float(ate_idx.get((c, key), 0.0)) * hr
                rows.append({"condition": c, "moderator": mod, "moderator_level": level,
                             "outcome": key, "mean": round(v, 3)})
mod_df = pd.DataFrame(rows)

# ---- verify + write ----
assert len(main) == 17 * 13, len(main)
n_levels = sum(profiles[m].nunique() for m in MODERATORS)
assert len(mod_df) == 17 * 13 * n_levels, (len(mod_df), n_levels)
assert not main["mean"].isna().any() and not mod_df["mean"].isna().any()
# sliders in [0,100], donation [0,10], newsletter [0,1]
for df in (main, mod_df):
    sl = df[~df.outcome.isin(FLAT_EFFECT)]
    assert sl["mean"].between(0, 100).all()
    assert df[df.outcome == "donation_ams"]["mean"].between(0, 10).all()
    assert df[df.outcome == "newsletter_signup"]["mean"].between(0, 1).all()

main.to_csv(RUNS_DIR / "T2_cells_main_v0.csv", index=False)
mod_df.to_csv(RUNS_DIR / "T2_cells_moderator_v0.csv", index=False)
print(f"main: {len(main)} rows; moderator: {len(mod_df)} rows ({n_levels} levels)")

print("\nsample moderation profile (Consensus, trust_multidimensional, party):")
s = mod_df[(mod_df.condition == "Consensus") & (mod_df.outcome == "trust_multidimensional")
           & (mod_df.moderator == "party")]
ctl = mod_df[(mod_df.condition == "control") & (mod_df.outcome == "trust_multidimensional")
             & (mod_df.moderator == "party")].set_index("moderator_level")["mean"]
for _, r in s.iterrows():
    print(f"  {r.moderator_level:12s} baseline {ctl[r.moderator_level]:6.2f}  "
          f"treated {r['mean']:6.2f}  effect {r['mean'] - ctl[r.moderator_level]:+.2f}")
print(f"  (overall ATE {ate_idx.get(('Consensus', 'trust_multidimensional')):+.2f})")
