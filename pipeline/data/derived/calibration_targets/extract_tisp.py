"""Extract TISP US calibration targets.

Run from pipeline/:  uv run python data/derived/calibration_targets/extract_tisp.py

Input : data/raw/reference/tisp/ds_final.csv (semicolon-delimited)
Output: trust_items_tisp.csv, trust_items_tisp_corr.csv

All statistics are UNWEIGHTED (matches the DIGEST quick-look sanity numbers;
WEIGHT_CNTRY exists but is stored with comma decimal separators and the
quota-sampled US n=2,559 is close to self-weighting).

Ideology terciles use DEM_POL_conservative (1 strongly liberal ... 5 strongly
conservative; "don't know"=99 arrives as NaN in ds_final):
  left = 1-2, center = 3, right = 4-5.  DEM_POL_right (1-5 left-right) exists
but has more missing (322 vs 196) and liberal-conservative is the standard US
framing, so it is the primary split. Note the scale is 1-5, NOT the 1-7 in the
task spec; terciles were adapted accordingly.
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "../../raw/reference/tisp/ds_final.csv"

# Item -> our subscale mapping, from core-questionnaire_english.pdf wordings
ITEMS = {
    "TRUST_SCI_expert":     ("competence",  "How expert or inexpert are most scientists?"),
    "TRUST_SCI_intellig":   ("competence",  "How intelligent or unintelligent are most scientists?"),
    "TRUST_SCI_qualified":  ("competence",  "How qualified or unqualified are most scientists when it comes to conducting high-quality research?"),
    "TRUST_SCI_honest":     ("integrity",   "How honest or dishonest are most scientists?"),
    "TRUST_SCI_ethical":    ("integrity",   "How ethical or unethical are most scientists?"),
    "TRUST_SCI_sincere":    ("integrity",   "How sincere or insincere are most scientists?"),
    "TRUST_SCI_concerned":  ("benevolence", "How concerned or not concerned are most scientists about people's wellbeing?"),
    "TRUST_SCI_improve":    ("benevolence", "How eager or uneager are most scientists to improve others' lives?"),
    "TRUST_SCI_otherint":   ("benevolence", "How considerate or inconsiderate are most scientists of others' interests?"),
    "TRUST_SCI_open":       ("openness",    "How open are most scientists to feedback?"),
    "TRUST_SCI_trans":      ("openness",    "How willing or unwilling are most scientists to be transparent?"),
    "TRUST_SCI_otherviews": ("openness",    "How much or little attention do scientists pay to others' views?"),
}
EXTRA = {
    "CLIM_TRUST": ("single_item_climate", "To what extent do you trust scientists in your country who work on climate change? (1 not at all - 5 very strongly)"),
    "TRUST_PEW":  ("single_item_pew",     "How much confidence do you have in scientists to act in the best interests of the public? (1 none at all - 5 a great deal)"),
}

def main():
    df = pd.read_csv(RAW, sep=";", low_memory=False)
    us = df[df["COUNTRY_NAME"] == "United States"].copy()
    assert len(us) == 2559, len(us)

    pol = us["DEM_POL_conservative"]
    us["ideo3"] = pd.cut(pol, bins=[0.5, 2.5, 3.5, 5.5],
                         labels=["left", "center", "right"])

    all_vars = {**ITEMS, **EXTRA}
    rows = []
    for var, (sub, wording) in all_vars.items():
        for grp, sdf in [("overall", us)] + [(g, us[us["ideo3"] == g])
                                             for g in ["left", "center", "right"]]:
            x = sdf[var].dropna()
            dist = x.value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0)
            rows.append({
                "item": var, "subscale": sub, "group": grp,
                "n": len(x), "mean": round(x.mean(), 4), "sd": round(x.std(), 4),
                **{f"p{int(k)}": round(v, 4) for k, v in dist.items()},
                "wording": wording if grp == "overall" else "",
            })
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "trust_items_tisp.csv", index=False)

    corr_vars = list(ITEMS) + list(EXTRA)
    corr = us[corr_vars].corr(method="pearson")  # pairwise complete
    corr.round(4).to_csv(HERE / "trust_items_tisp_corr.csv")

    # sanity
    trust12 = us[list(ITEMS)].mean(axis=1)
    print("US n:", len(us))
    print("trust12 composite mean/sd:", round(trust12.mean(), 3), round(trust12.std(), 3))
    print("item means range:", out[out.group == "overall"].set_index("item")["mean"].agg(["min", "max"]).to_dict())
    print("ideology group ns:", us["ideo3"].value_counts().to_dict(), "missing:", pol.isna().sum())

if __name__ == "__main__":
    main()
