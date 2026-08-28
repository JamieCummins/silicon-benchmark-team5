"""Compute simple per-intervention ATEs vs control for the Vlasceanu et al. (2024)
climate intervention tournament (Sci. Adv. 10.1126/sciadv.adj5778).

Input:  data63.xlsx (Zenodo 10.5281/zenodo.10345806) -- participant-level data,
        63 countries, 59,440 rows.
Output: us_ates.csv  -- one row per (sample x intervention x outcome).

Method (documented per project instructions):
- Outcome scores follow the paper's preregistered outcome definitions collapsed
  to participant level:
    belief  = mean of non-missing Belief1..Belief4   (0-100 sliders)
    policy  = mean of non-missing Policy1..Policy9   (0-100 sliders)
    share   = SHAREcc (1 = agreed to share climate info on social media, else 0)
    wept    = WEPTcc  (0-8 pages completed in the Work for Environmental
              Protection Task; each page ~= trees planted via checked boxes)
- ATE = mean(treatment) - mean(control), computed separately within each sample
  ("US" = Country == "Usa"; "ALL" = all 63 countries pooled).
- SE  = Welch standard error sqrt(s_t^2/n_t + s_c^2/n_c).
- No covariates, no multilevel structure (the paper's preregistered models use
  item/participant/country random effects; here we use simple difference in
  means by design, for benchmark ground truth).
"""

import numpy as np
import pandas as pd

DATA = "data63.xlsx"
OUT = "us_ates.csv"

df = pd.read_excel(DATA, sheet_name=0, na_values=["NA"])

belief_items = [f"Belief{i}" for i in range(1, 5)]
policy_items = [f"Policy{i}" for i in range(1, 10)]
for c in belief_items + policy_items + ["SHAREcc", "WEPTcc"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["belief"] = df[belief_items].mean(axis=1)   # NaN only if all 4 missing
df["policy"] = df[policy_items].mean(axis=1)   # NaN only if all 9 missing
df["share"] = df["SHAREcc"]
df["wept"] = df["WEPTcc"]

OUTCOMES = {
    "belief": "0-100; mean of 4 belief sliders",
    "policy": "0-100; mean of 9 policy-support sliders",
    "share": "0/1; shared climate info on social media",
    "wept": "0-8; WEPT pages completed",
}

rows = []
for sample_name, d in [("US", df[df["Country"] == "Usa"]), ("ALL", df)]:
    ctrl = d[d["condName"] == "Control"]
    for cond in sorted(d["condName"].dropna().unique()):
        if cond == "Control":
            continue
        tr = d[d["condName"] == cond]
        for oc, desc in OUTCOMES.items():
            t = tr[oc].dropna()
            c = ctrl[oc].dropna()
            if len(t) < 2 or len(c) < 2:
                continue
            ate = t.mean() - c.mean()
            se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
            rows.append({
                "sample": sample_name,
                "intervention": cond,
                "outcome": oc,
                "outcome_scale": desc,
                "ate_vs_control": ate,
                "se_welch": se,
                "n_treat": len(t),
                "n_control": len(c),
                "mean_treat": t.mean(),
                "sd_treat": t.std(ddof=1),
                "mean_control": c.mean(),
                "sd_control": c.std(ddof=1),
            })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False, float_format="%.5f")
print(out[out["sample"] == "US"].to_string(index=False, max_colwidth=40))
print(f"\nWrote {len(out)} rows to {OUT}")
