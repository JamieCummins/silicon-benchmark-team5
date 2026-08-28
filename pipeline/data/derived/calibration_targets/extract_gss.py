"""Extract GSS 2024 CONSCI x PARTYID sanity anchor.

Run from pipeline/:  uv run --with pyreadstat python data/derived/calibration_targets/extract_gss.py

Input : data/raw/reference/gss/2024/GSS2024.dta
Output: gss_consci.csv

Variables: consci (1 a great deal / 2 only some / 3 hardly any confidence in
the scientific community), partyid (0 strong Dem ... 6 strong Rep, 7 other),
weight wtssnrps (2024 post-stratified nonresponse-adjusted weight; missing for
677 of 3,986 cases, which are dropped). String missing codes (.d/.n/.s etc.)
arrive as NaN and are excluded. Proportions are weighted row proportions of
consci within each partyid level; n is unweighted.
"""
import pandas as pd
import pyreadstat
from pathlib import Path

HERE = Path(__file__).parent
DTA = HERE / "../../raw/reference/gss/2024/GSS2024.dta"

PARTY = {0: "strong democrat", 1: "not very strong democrat",
         2: "independent, near democrat", 3: "independent",
         4: "independent, near republican", 5: "not very strong republican",
         6: "strong republican", 7: "other party"}
CONSCI = {1: "great_deal", 2: "only_some", 3: "hardly_any"}


def main():
    df, _ = pyreadstat.read_dta(str(DTA), usecols=["consci", "partyid", "wtssnrps"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    d = df.dropna(subset=["consci", "partyid", "wtssnrps"])
    d = d[d.wtssnrps > 0]
    print("valid n:", len(d))

    rows = []
    for pid, g in [("overall", d)] + [(k, d[d.partyid == k]) for k in sorted(PARTY)]:
        wt = g.groupby("consci")["wtssnrps"].sum()
        tot = wt.sum()
        rows.append({
            "partyid": pid if pid == "overall" else int(pid),
            "partyid_label": "overall" if pid == "overall" else PARTY[pid],
            "n_unweighted": len(g),
            **{f"p_{CONSCI[k]}": round(float(wt.get(k, 0) / tot), 4) for k in CONSCI},
        })
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "gss_consci.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
