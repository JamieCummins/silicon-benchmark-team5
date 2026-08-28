"""Build the T1 cohort end-to-end and verify it. No API calls.

  uv run python scripts/build_t1.py
Writes runs/T1_cohort_v0.csv + prints the verification report.
"""

import numpy as np
import pandas as pd

from silicon.aggregate import calibrate, cell_estimates
from silicon.config import RUNS_DIR
from silicon.cohort.latents import TRUST_ITEM_COLS, draw_latents, draw_trust_items
from silicon.cohort.profiles import build_profiles
from silicon.cohort.treat import CHANNELS, calibrate_slopes, render_with_treatment, treatment_shifts
from silicon.cohort.render import render_all

# 1. fresh T3-A vector from the current (backfilled) crowd
el = pd.read_csv(RUNS_DIR / "t3a_elicitations.csv")
ates = calibrate(cell_estimates(el))[["condition", "outcome", "ate"]]
print(f"T3-A vector: {len(ates)} cells from {el.groupby(['model','variant','intervention']).ngroups} crowd cells")

# 2. cohort skeleton
profiles = build_profiles()
z = draw_latents(profiles)
items_z = draw_trust_items(z["trust"])

# Balance-correct baseline latents across arms: remove between-condition noise
# in baseline means (SE ~0.6-0.9 points on SD~25 sliders would otherwise swamp
# the injected +/-0.5-1.5 effects; the shared control mean makes it common-mode).
# Arm contrasts then carry only the injected vector + render noise.
cond = profiles["condition"].values
for frame in (z, items_z):
    for col in frame.columns:
        grp = frame.groupby(cond)[col].transform("mean")
        frame[col] = frame[col] - grp + frame[col].mean()

# 3. slope calibration + treatment
slopes = calibrate_slopes(profiles, z, items_z)
print("\nchannel slopes (points per latent unit):")
print({k: round(v, 2) for k, v in slopes.items()})
shifts = treatment_shifts(profiles, z, ates, slopes)
data = render_with_treatment(profiles, z, items_z, shifts)

# feedback passes: probe slopes are biased by render nonlinearity x headroom.
# The render map is deterministic (fixed seeds), so per-cell target/realized
# rescaling converges to near-exact realized ATEs — protecting the within-
# outcome arm ranking (spread ~0.2 pts) from render noise.
ate_idx = ates.set_index(["condition", "outcome"])["ate"]
cond_arr = profiles.condition.values
for _ in range(3):
    ctrl_means = {key: data.loc[cond_arr == "control", key].mean()
                  for key in CHANNELS.values()}
    for ch, key in CHANNELS.items():
        for c in profiles["condition"].unique():
            if c == "control":
                continue
            t = ate_idx.get((c, key), 0.0)
            r = data.loc[cond_arr == c, key].mean() - ctrl_means[key]
            mask = cond_arr == c
            shifts[ch][mask] = shifts[ch][mask] + (t - r) / slopes[ch]
    data = render_with_treatment(profiles, z, items_z, shifts)

# 4. assemble in exact template column order
front = ["profile_id", "condition", "gender", "age_band", "race", "education", "income", "party"]
outcome_cols = (
    ["trust_multidimensional"] + TRUST_ITEM_COLS
    + ["trust_post", "distrust_post", "funding_perceptions", "policy_role_mean",
       "inst_trust_mean", "belief_post", "concern_mean", "policy_general",
       "policy_specific_mean", "behavior_mean", "donation_ams", "newsletter_signup"]
)
t1 = pd.concat([profiles[front].reset_index(drop=True),
                data[outcome_cols].reset_index(drop=True)], axis=1)

# 5. verification
print("\n=== realized vs target ATEs (native units) ===")
ctrl = t1[t1.condition == "control"]
rows = []
for key in [CHANNELS[ch] for ch in CHANNELS]:
    tgt = ates[ates.outcome == key].set_index("condition")["ate"]
    for cond in tgt.index:
        realized = t1.loc[t1.condition == cond, key].mean() - ctrl[key].mean()
        rows.append({"outcome": key, "cond": cond, "target": tgt[cond], "realized": realized})
ver = pd.DataFrame(rows)
ver["abs_err"] = (ver.realized - ver.target).abs()
print(ver.groupby("outcome")[["target", "realized", "abs_err"]].mean().round(3).to_string())
print(f"max |err| = {ver.abs_err.max():.3f}")

print("\n=== control-group baselines (target from outcome_priors.md) ===")
targets = {"trust_multidimensional": 66, "trust_post": 64, "distrust_post": 32,
           "funding_perceptions": 58, "policy_role_mean": 58, "inst_trust_mean": 55,
           "belief_post": 66, "concern_mean": 58, "policy_general": 66,
           "policy_specific_mean": 63, "behavior_mean": 48,
           "donation_ams": 1.85, "newsletter_signup": 0.09}
for k, tv in targets.items():
    print(f"  {k:24s} target {tv:>6}  realized {ctrl[k].mean():7.2f}  sd {ctrl[k].std():6.2f}")

print("\n=== shape diagnostics (control) ===")
for col in ("trust_competence_1", "belief_post"):
    v = ctrl[col].values
    on5 = np.mean(np.mod(v, 5) == 0) * 100
    lab = np.mean(np.isin(v, [0, 50, 100])) * 100
    print(f"  {col}: {on5:.0f}% on multiples of 5 (target ~80), {lab:.0f}% on {{0,50,100}} (target ~25)")
print(f"  donation: P(0)={np.mean(ctrl.donation_ams == 0)*100:.0f}% (target 40-55), "
      f"P(5)={np.mean(ctrl.donation_ams == 5)*100:.0f}%, P(10)={np.mean(ctrl.donation_ams == 10)*100:.0f}%")
print(f"  newsletter rate: {ctrl.newsletter_signup.mean()*100:.1f}% (target ~9)")
items = ctrl[TRUST_ITEM_COLS]
cm = items.corr().values
print(f"  trust item mean inter-r: {cm[np.triu_indices(12, 1)].mean():.2f} (TISP target ~.5-.65)")
gap = (ctrl[ctrl.party == 'Democrat'].trust_multidimensional.mean()
       - ctrl[ctrl.party == 'Republican'].trust_multidimensional.mean())
print(f"  Dem-Rep gap on trust_multidimensional: {gap:.1f} points (target ~20-25, climate-widened)")

# composite consistency check the validator runs
dev = (t1["trust_multidimensional"]
       - t1[TRUST_ITEM_COLS].mean(axis=1)).abs().max()
print(f"  composite consistency max dev: {dev:.4f} (validator limit 0.5)")

out = RUNS_DIR / "T1_cohort_v0.csv"
t1.to_csv(out, index=False)
print(f"\nwrote {len(t1)} rows x {len(t1.columns)} cols -> {out}")
