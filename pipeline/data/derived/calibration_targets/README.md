# Calibration targets

Exact reference targets for the synthetic-cohort generator, extracted 2026-08-18
from `pipeline/data/raw/reference/` (see `DIGEST.md` there for provenance).
Every file is reproducible from the `extract_*.py` script sitting next to it;
run each from `pipeline/` with `uv run --with pyreadstat python <script>`
(pyreadstat only needed for the ANES/CCAM/GSS scripts).

## 1. TISP (US, n = 2,559) — `trust_items_tisp.csv`, `trust_items_tisp_corr.csv`
Script: `extract_tisp.py`. Source: `tisp/ds_final.csv` (semicolon-delimited),
filter `COUNTRY_NAME == "United States"`. **Unweighted** (matches DIGEST
quick-look; WEIGHT_CNTRY stored with comma decimals, US sample near
self-weighting).

Item -> subscale mapping (from `core-questionnaire_english.pdf`; all items
1-5, higher = more trusting; full wordings in the CSV `wording` column):

| Subscale    | Items |
|-------------|-------|
| competence  | TRUST_SCI_expert, TRUST_SCI_intellig, TRUST_SCI_qualified |
| integrity   | TRUST_SCI_honest, TRUST_SCI_ethical, TRUST_SCI_sincere |
| benevolence | TRUST_SCI_concerned, TRUST_SCI_improve, TRUST_SCI_otherint |
| openness    | TRUST_SCI_open, TRUST_SCI_trans, TRUST_SCI_otherviews |

Plus CLIM_TRUST (trust in climate scientists, 1-5) and TRUST_PEW (Pew-style
confidence, 1-5) as extra rows and in the 14x14 Pearson corr matrix (pairwise
complete deletion; per-item missingness is 0-2 cases).

Ideology split: **DEM_POL_conservative** (1 strongly liberal ... 5 strongly
conservative; "don't know" = NaN, 196 missing). NOTE: the scale is **1-5, not
1-7** as in the task spec; terciles adapted as left = 1-2 (n=667),
center = 3 (n=718), right = 4-5 (n=978). DEM_POL_right (left-right, also 1-5)
exists but has more missing (322).

Sanity: trust-12 composite mean 3.868 (SD 0.824) — matches the ≈3.87 target.
Item means 3.62 (open) to 4.32 (intellig). CLIM_TRUST 3.69 (SD 1.29),
TRUST_PEW 3.84 (SD 1.12). trust12 x CLIM_TRUST r = .72 block reproduced in
corr file.

## 2. ANES — `thermometer_anes.json`
Script: `extract_anes.py`. Scientists feeling thermometer, 0-100.

| Year | FT | Party ID | Weight |
|------|----|----------|--------|
| 2020 | V202173 | V201231x (7-pt summary) | V200010b (post, full sample) |
| 2016 | V162112 | V161158x (7-pt summary) | V160102 (post) |

Missing handling: FT outside [0,100] excluded (negative ANES codes -9/-7/-6/-5/-4
and 998/999 DK); party outside [1,7] excluded; non-positive/missing weight
excluded. All stats **weighted**; `n` = unweighted valid count. Party groups:
Dem incl lean = 1-3, Ind = 4, Rep incl lean = 5-7. JSON carries per group:
n, weighted mean/SD, % on multiples of 5, % on {0,50,100}, and the full
{integer value: weighted proportion} histogram.

Key weighted numbers: 2020 overall 78.0 (SD 20.9, n 7,367), D 85.5 / I 74.8 /
R 71.0 (D-R gap 14.6); heaping 98.8% on 5s, 44.1% on {0,50,100}.
2016 overall 76.5 (SD 19.8, n 3,615), D 80.7 / I 72.8 / R 72.7 (gap 8.0);
heaping 80.4% on 5s, 32.4% on {0,50,100}.

## 3. CCAM — `ccam_targets.csv`, `ccam_corr.csv`, `party_by_demo.csv`
Script: `extract_ccam.py`. Source: `ccam/CCAM_SPSS_Data_2008-2024.sav`,
**waves 24-31 pooled (Mar 2021 - Dec 2024), n = 8,234**, weight
`weight_aggregate` (YPCCC's pooled-analysis weight).

Items + numeric codings (full response labels in the CSV `coding` column;
-1 Refused always dropped):
- `happening` 1 No / 2 Don't know / 3 Yes (DK kept as middle category)
- `worry` 1-4 (not at all ... very worried)
- `harm_personally`, `harm_US`, `harm_future_gen` 1-4; **0 = Don't know**,
  excluded from means/SD/correlations but reported as `dist_0`
- policy: `reg_CO2_pollutant`, `fund_research`, `reduce_tax` 1-4
  (strongly oppose ... strongly support); fund_research asked in only 5 of the
  8 waves (n≈6,170), reg_CO2/reduce_tax in 7 (n≈7,200)

Party allocation (leaners allocated via `party_w_leaners`):
Republican = party_w_leaners 1; Democrat = 2;
Independent (no lean) = party_w_leaners 3 & party 3;
Other = (party_w_leaners 3 & party 4) or party_w_leaners 4 (no party/not
interested, incl. party 5); refused dropped. Weighted shares:
D .437 / R .354 / I .102 / Other .107.

`ccam_corr.csv`: weighted pairwise Pearson (numeric codings above).
Anchors: worry x reg_CO2 .65, worry x fund_research .64, happening x worry .65.

`party_by_demo.csv`: weighted party proportions within gender (1 Male/2 Female)
x age band (continuous `age`: 18-29/30-44/45-59/60+) x education (`educ`
collapsed: 1-9 HS-or-less, 10-11 Some college incl. Associate's, 12 Bachelor,
13-14 Postgrad) = 32 cells; all cells n >= 43 unweighted (median 241).

## 4. GSS 2024 — `gss_consci.csv`
Script: `extract_gss.py`. Source: `gss/2024/GSS2024.dta`. `consci`
(1 great deal / 2 only some / 3 hardly any) x `partyid` (0 strong Dem ...
6 strong Rep, 7 other). Weight `wtssnrps` (missing for 677 of 3,986 cases —
dropped); string missing codes arrive as NaN and are excluded; valid analytic
n = 2,104. Weighted consci proportions within party + overall.
Anchor: overall 36.6 / 51.3 / 12.1; strong Dem 56.3% great deal; strong Rep
21.0% great deal, 22.4% hardly any.

## Caveats to double-check
1. TISP ideology terciles use a 1-5 liberal-conservative item, not 1-7.
2. TISP is unweighted by design here.
3. CCAM `happening` treats "Don't know" as the numeric middle (YPCCC coding);
   if you want belief-strength only, drop code 2 instead.
4. ANES 2020 heaping (98.8% on 5s) partly reflects labeled response points;
   2016 (80.4%) is the better anchor for an unlabeled slider (DIGEST note).
5. Benevolence/openness assignment of `otherint` (considerate of others'
   interests -> benevolence) vs `otherviews` (attention to others' views ->
   openness) follows the questionnaire's Mayer-style dimensions; these two are
   the most swappable items in the mapping.
