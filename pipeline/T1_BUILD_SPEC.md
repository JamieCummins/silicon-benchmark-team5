# T1 "calibrated cohort" — build specification

Target: N=18,000 synthetic individual-level dataset in the template's T1 schema
(template/predictions/example_T1_primary_v1.csv), `make check` clean, no
precision-floor warnings (>=1,000/intervention arm, 2,000 control).

## Inputs (all on disk)

| input | path | role |
|---|---|---|
| T3-A effect vector | pipeline/runs/benchmark_T3A_v0.csv | treatment shifts (re-aggregate from 847-cell crowd before freeze) |
| ACS PUMS adults | data/raw/reference/census/acs2023_pums_adults.csv.gz | joint demographics (n=2.77M, weighted 262M) |
| TISP US | data/raw/reference/tisp/ds_final.csv (US subset n=2,559) | 12-item trust battery shapes (1-5) + construct correlations |
| ANES 2016/2020 | data/raw/reference/anes/ | 0-100 thermometer shape, heaping, partisan gaps |
| CCAM | data/raw/reference/ccam/ | climate belief/worry/policy dists + party x demographics |
| GSS 2024 | data/raw/reference/gss/ | confidence-in-science categorical check |
| Pew 2026 topline | data/raw/reference/pew/ | 2026 level drift anchor |
| priors + baselines | reference/outcome_priors.md | target control-group means |

## Locked calibration decisions

1. Thermometer SHAPE from ANES, 2016 regime (~81% on multiples of 5, softer
   label-heaping) — benchmark sliders are unlabeled empty-default boxes.
2. Partisan gap widened from general-science to climate-science using
   CCAM/TISP climate items (climate trust more polarized).
3. Party raked onto ACS joint via CCAM party x demographics + NPORS 2025
   margins (47D/43R lean split).
4. Levels drifted to 2026 via Pew trend (87->73->77% confidence path).
5. Baseline targets from outcome_priors.md table (trust ~66/64, distrust ~32,
   belief ~66, concern ~58, policy_gen ~66, donation ~$1.85, newsletter ~9%).

## Pipeline stages (module: src/silicon/cohort/)

### S1 profiles.py — sample 18,000 demographic profiles
- Weighted draw from ACS PUMS adults; band AGEP->age_band, SCHL->6 education
  levels, HINCP(ADJINC)->5 income bands, RAC1P/HISP->5 race levels, SEX->
  gender (add small Other share ~1.5%, benchmark panels include it).
- Rake to the benchmark's stated quotas: census-matched gender x age,
  gender x race (prereg quota table); education/income/party free-floating
  but panel-skew-adjusted (opt-in panels: more educated, more Democratic).
- Assign party per profile: P(party | age, race, gender, education, income)
  fit on CCAM microdata (multinomial logit or raked cells), margins matched
  to NPORS-with-panel-skew. Assign state ~ ACS state weights (feeds weather
  case assignment; 'Prefer not to say' ~1%).
- profile_id, condition assignment: random, 1000/arm + 2000 control
  (validator: exact per-condition N; strings from condition_codenames.csv
  titles + 'control').

### S2 baselines.py — latent baseline scores per person x outcome
- Latent scale: z-scores on a multivariate normal with correlation matrix
  assembled from TISP (trust items <-> climate trust <-> policy), CCAM
  (belief <-> worry <-> policy), cross-block filled by Hommel/Arslan-style
  semantic-similarity prior where unmeasured; make PSD (nearest-corr).
- Group means: party/demographic offsets per outcome estimated from
  reference data (TISP/CCAM/ANES), scaled so the marginal matches the
  outcome_priors baseline table + ANES-derived SDs (thermometer SD ~20;
  widen for climate polarization: bimodal by party).
- 12 trust items: person-level trust factor + item noise calibrated to TISP
  item intercorrelations; items on 0-100 via S3 mapping.

### S3 render.py — map latents to observed response scales
- Latent -> 0-100: quantile-map each outcome's latent marginal onto a target
  histogram built from ANES-2016-regime heaping: (i) continuous target CDF
  per group (party-specific), (ii) snap to integers with heaping kernel:
  P(multiple of 10) >> P(multiple of 5) >> other integers; extra mass at
  0/50/100; calibrate snap shares to ANES 2016 (~81% on 5s incl 10s,
  ~25-30% on {0,50,100}).
- donation_ams: two-part model — P(zero) ~ 45%, then discrete $1-10 with
  modes at 5 and 10, mean matching $1.85 overall; correlate with trust/
  concern latent (generosity gradient).
- newsletter_signup: Bernoulli(p_i), p_i = logistic in latent interest
  (concern/trust), marginal 9%.
- Composites: trust_multidimensional = mean of 12 items (validator checks
  consistency <=0.5); other composites built from their items analogously
  (policy_role 4 items, inst_trust 5, concern 3, policy_specific 7,
  behavior 6 — generate item-level then average; item wordings' means offset
  per outcome_priors notes, e.g. fed-gov trust ~35-40 vs NASA ~65).
  CHECK template T1 schema: which item columns are required vs composite-only
  — read codebook.csv target labels for the exact required columns.
- funding_perceptions: generate funding_5 raw (0-100, 'about right' heap at
  50) then recode 100-x.

### S4 treat.py — inject treatment effects
- Person-level effect for condition c, outcome o:
  delta_i = ATE(c,o) * headroom_i / mean(headroom) + noise
  headroom_i = (100 - baseline_i)/100 for positive-effect sliders (mirror
  for distrust); noise SD ~ 1.5x |ATE| (individual heterogeneity, keeps
  arm-level ATE exact in expectation, adds realism).
- Extreme weather arm: person's state -> case (prereg mapping); ATE applied
  uniformly (case-level variation not identified; note in registration).
- Behavioral: donation shift via P(zero) and mean shift matching ATE in $;
  newsletter via logit shift matching ATE in probability.
- Post-injection re-render (snap again) WITHOUT destroying the ATE: apply
  effects on the latent BEFORE S3 rendering (cleanest) — i.e., order is
  S2 latents -> S4 latent shifts -> S3 render. Verify realized ATEs vs
  target vector (tolerance 0.1x); iterate snap calibration if heaping
  attenuates effects.

### S5 validate.py — self-diagnostics before make check
- Realized vs target: per-cell ATEs, control means vs baseline table.
- Section-3 mimicry: variance ratio vs ANES/TISP-derived targets (~1.0),
  heaping shares, partisan bimodality (dip test / group means), item
  intercorrelations vs TISP, demographic-gap table vs reference (stereotype
  metric: our group gaps should MATCH reference surveys, not exaggerate).
- Run template's make check; zero fails, zero warnings.

## Schema notes (from template example + codebook)
- Columns: profile_id, condition, gender, age_band, race, education, income,
  party + outcome columns (exact set per codebook target_label rows marked
  T1-required — VERIFY against scripts/check before building S3).
- Exact strings: lowercase 'control'; race 'Black / African American' etc.;
  income bands with exact $ formatting from codebook.

## Order of work
1. S1 (0.5d) -> 2. correlation matrix + S2 (1d) -> 3. S3 rendering +
   calibration loop vs ANES/TISP (1d) -> 4. S4 + realized-ATE verification
   (0.5d) -> 5. S5 + make check + fix cycle (0.5d). Fallback date: if not
   make-check-clean by Aug 26, primary moves to T3-A.

## API needs
None mandatory (fully statistical given acquired data). Optional: spot
readouts for uncovered item-level baselines (~$5-10) if reference mapping
leaves gaps (e.g., 'scientists should advocate' items absent from TISP/CCAM
-> readout those item baselines per party on the 0-9 digit scale).
