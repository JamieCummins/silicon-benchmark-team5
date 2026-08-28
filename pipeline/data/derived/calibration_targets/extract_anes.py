"""Extract ANES scientists feeling-thermometer targets (2016 + 2020).

Run from pipeline/:  uv run --with pyreadstat python data/derived/calibration_targets/extract_anes.py

Inputs : data/raw/reference/anes/anes_timeseries_2020_stata_20220210.dta
         data/raw/reference/anes/anes_timeseries_2016.dta
Output : thermometer_anes.json

Variables
  2020: V202173 scientists FT (0-100), V201231x party ID summary (1-7),
        V200010b post-election full-sample weight
  2016: V162112 scientists FT (0-100), V161158x party ID summary (1-7),
        V160102 post-election weight
Missing handling: FT values outside [0, 100] are excluded (ANES negative codes
-9 refused / -7 no post data / -6 no post interview / -5 breakoff / -4 error,
and any 998/999 DK codes). Party values outside [1, 7] excluded. Rows with
non-positive/missing weight excluded. Statistics are WEIGHTED; n is the
unweighted count of valid cases. Histogram = weighted proportion per observed
integer value.
"""
import json
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

HERE = Path(__file__).parent
ANES = HERE / "../../raw/reference/anes"

SPECS = {
    "2020": dict(file="anes_timeseries_2020_stata_20220210.dta",
                 ft="V202173", pid="V201231x", wt="V200010b"),
    "2016": dict(file="anes_timeseries_2016.dta",
                 ft="V162112", pid="V161158x", wt="V160102"),
}
PARTY_GROUPS = {"dem_incl_lean": (1, 2, 3), "ind": (4,), "rep_incl_lean": (5, 6, 7)}


def wstats(x, w):
    m = np.average(x, weights=w)
    sd = np.sqrt(np.average((x - m) ** 2, weights=w) * len(x) / (len(x) - 1))
    hist = pd.Series(w, index=x.astype(int)).groupby(level=0).sum() / w.sum()
    return dict(
        n=int(len(x)), mean=round(float(m), 3), sd=round(float(sd), 3),
        pct_multiple_of_5=round(float(w[(x % 5 == 0)].sum() / w.sum()), 4),
        pct_on_0_50_100=round(float(w[np.isin(x, [0, 50, 100])].sum() / w.sum()), 4),
        histogram={str(int(k)): round(float(v), 5) for k, v in hist.items()},
    )


def main():
    out = {"_notes": __doc__.strip()}
    for year, s in SPECS.items():
        df, _ = pyreadstat.read_dta(str(ANES / s["file"]),
                                    usecols=[s["ft"], s["pid"], s["wt"]])
        df.columns = ["ft", "pid", "wt"] if list(df.columns) == [s["ft"], s["pid"], s["wt"]] else df.columns
        df = df.rename(columns={s["ft"]: "ft", s["pid"]: "pid", s["wt"]: "wt"})
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        valid = df[(df.ft.between(0, 100)) & (df.wt > 0)].copy()
        res = {"variables": s, "overall": wstats(valid.ft.values, valid.wt.values),
               "by_party": {}}
        for g, codes in PARTY_GROUPS.items():
            sub = valid[valid.pid.isin(codes)]
            res["by_party"][g] = wstats(sub.ft.values, sub.wt.values)
        out[year] = res
        print(year, "valid n:", res["overall"]["n"], "mean:", res["overall"]["mean"],
              "| party means:", {g: res["by_party"][g]["mean"] for g in PARTY_GROUPS})

    (HERE / "thermometer_anes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
