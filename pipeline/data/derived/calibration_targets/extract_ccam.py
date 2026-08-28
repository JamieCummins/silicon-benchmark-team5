"""Extract CCAM (Yale/GMU Climate Change in the American Mind) calibration targets.

Run from pipeline/:  uv run --with pyreadstat python data/derived/calibration_targets/extract_ccam.py

Input : data/raw/reference/ccam/CCAM_SPSS_Data_2008-2024.sav
Output: ccam_targets.csv, ccam_corr.csv, party_by_demo.csv

Sample: waves 24-31 (Mar 2021 - Dec 2024) pooled, n = 8,234.
Weight: weight_aggregate (YPCCC's weight for pooled multi-wave analysis).

Items (numeric coding for means/correlations; -1 Refused always -> missing):
  happening   1 No / 2 Don't know / 3 Yes            (DK kept as middle = YPCCC coding)
  worry       1 Not at all ... 4 Very worried
  harm_personally, harm_US, harm_future_gen
              1 Not at all ... 4 A great deal; 0 = Don't know -> missing for
              mean/SD/corr, but reported as dist_0 in the categorical distribution
  reg_CO2_pollutant, fund_research, reduce_tax
              1 Strongly oppose ... 4 Strongly support
              (fund_research asked in 5 of 8 waves, n~6,170; reg_CO2/reduce_tax in 7)

Party allocation (leaners allocated via party_w_leaners):
  Republican  party_w_leaners == 1 (Republicans incl. leaners)
  Democrat    party_w_leaners == 2 (Democrats incl. leaners)
  Independent party_w_leaners == 3 AND party == 3 (Independent, no lean)
  Other       party_w_leaners == 3 AND party == 4 (Other party), OR
              party_w_leaners == 4 (No party/not interested; incl. party == 5)
  Refused (-1 on either) -> excluded from party groups.

party_by_demo.csv: weighted party proportions within gender x age band
(18-29/30-44/45-59/60+, from continuous `age`) x education (educ collapsed:
1-9 HS-or-less, 10-11 Some college [incl. Associate's], 12 Bachelor,
13-14 Postgrad), plus unweighted cell n and weighted n.
"""
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

HERE = Path(__file__).parent
SAV = HERE / "../../raw/reference/ccam/CCAM_SPSS_Data_2008-2024.sav"

ITEMS = ["happening", "worry", "harm_personally", "harm_US", "harm_future_gen",
         "reg_CO2_pollutant", "fund_research", "reduce_tax"]
LABELS = {
    "happening":  "1=No 2=Don't know 3=Yes (Do you think global warming is happening?)",
    "worry":      "1=Not at all worried 2=Not very 3=Somewhat 4=Very worried",
    "harm_personally": "0=Don't know(excl from mean) 1=Not at all 2=Only a little 3=Moderate amount 4=A great deal",
    "harm_US":         "0=Don't know(excl from mean) 1=Not at all 2=Only a little 3=Moderate amount 4=A great deal",
    "harm_future_gen": "0=Don't know(excl from mean) 1=Not at all 2=Only a little 3=Moderate amount 4=A great deal",
    "reg_CO2_pollutant": "1=Strongly oppose 2=Somewhat oppose 3=Somewhat support 4=Strongly support (regulate CO2 as a pollutant)",
    "fund_research":     "1=Strongly oppose 2=Somewhat oppose 3=Somewhat support 4=Strongly support (fund research into renewable energy sources)",
    "reduce_tax":        "1=Strongly oppose 2=Somewhat oppose 3=Somewhat support 4=Strongly support (give tax rebates for efficient vehicles/solar)",
}
HARM = {"harm_personally", "harm_US", "harm_future_gen"}


def wmean_sd(x, w):
    if len(x) < 2:
        return np.nan, np.nan
    m = np.average(x, weights=w)
    sd = np.sqrt(np.average((x - m) ** 2, weights=w) * len(x) / (len(x) - 1))
    return m, sd


def wcorr(a, b, w):
    ok = a.notna() & b.notna()
    a, b, w = a[ok].values, b[ok].values, w[ok].values
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    return cov / np.sqrt(np.average((a - ma) ** 2, weights=w) *
                         np.average((b - mb) ** 2, weights=w))


def main():
    df, meta = pyreadstat.read_sav(str(SAV))
    d = df[df.wave >= 24].copy()
    print("pooled waves 24-31 n:", len(d))
    w = "weight_aggregate"

    # party allocation
    pwl, p = d["party_w_leaners"], d["party"]
    d["party4"] = np.select(
        [pwl == 1, pwl == 2, (pwl == 3) & (p == 3),
         ((pwl == 3) & (p == 4)) | (pwl == 4)],
        ["Republican", "Democrat", "Independent", "Other"], default="Refused")

    # numeric versions (refused -> nan; harm DK(0) -> nan)
    for v in ITEMS:
        d[v + "_num"] = d[v].where(d[v] != -1)
        if v in HARM:
            d[v + "_num"] = d[v + "_num"].where(d[v + "_num"] != 0)

    # (a) targets: per item x party group (+ overall)
    rows = []
    groups = [("overall", d)] + [(g, d[d.party4 == g])
                                 for g in ["Republican", "Democrat", "Independent", "Other"]]
    for v in ITEMS:
        cats = sorted(c for c in d[v].dropna().unique() if c != -1)
        for gname, g in groups:
            valid = g[g[v].notna() & (g[v] != -1)]
            num = g[v + "_num"].dropna()
            wnum = g.loc[num.index, w]
            m, sd = wmean_sd(num.values, wnum.values)
            dist = valid.groupby(v)[w].sum().reindex(cats, fill_value=0)
            dist = dist / dist.sum()
            rows.append({
                "item": v, "party_group": gname, "n_unweighted": len(valid),
                "wmean": round(m, 4), "wsd": round(sd, 4),
                **{f"dist_{int(c)}": round(dist[c], 4) for c in cats},
                "coding": LABELS[v] if gname == "overall" else "",
            })
    pd.DataFrame(rows).to_csv(HERE / "ccam_targets.csv", index=False)

    # (b) weighted pairwise Pearson correlations
    corr = pd.DataFrame(index=ITEMS, columns=ITEMS, dtype=float)
    for i, a in enumerate(ITEMS):
        for b in ITEMS[i:]:
            r = wcorr(d[a + "_num"], d[b + "_num"], d[w])
            corr.loc[a, b] = corr.loc[b, a] = round(r, 4)
    corr.to_csv(HERE / "ccam_corr.csv")

    # (c) party x demographics
    d["age_band"] = pd.cut(d.age, [17, 29, 44, 59, 200],
                           labels=["18-29", "30-44", "45-59", "60+"])
    d["educ4"] = pd.cut(d.educ, [0, 9, 11, 12, 14],
                        labels=["HS or less", "Some college", "Bachelor", "Postgrad"])
    d["gender_lab"] = d.gender.map({1: "Male", 2: "Female"})
    dd = d[d.party4 != "Refused"]
    out = []
    for (g_, a_, e_), cell in dd.groupby(["gender_lab", "age_band", "educ4"],
                                         observed=False):
        tot = cell[w].sum()
        pr = cell.groupby("party4")[w].sum() / tot if tot > 0 else {}
        out.append({
            "gender": g_, "age_band": a_, "educ": e_,
            "n_unweighted": len(cell), "weighted_n": round(tot, 1),
            **{f"p_{k.lower()}": round(float(pr.get(k, 0)), 4) if tot > 0 else np.nan
               for k in ["Republican", "Democrat", "Independent", "Other"]},
        })
    pbd = pd.DataFrame(out)
    pbd.to_csv(HERE / "party_by_demo.csv", index=False)

    print("party4 (weighted):", (dd.groupby('party4')[w].sum() / dd[w].sum()).round(3).to_dict())
    print("min/median cell n:", pbd.n_unweighted.min(), pbd.n_unweighted.median())
    print("cells with n<30:", (pbd.n_unweighted < 30).sum(), "of", len(pbd))


if __name__ == "__main__":
    main()
