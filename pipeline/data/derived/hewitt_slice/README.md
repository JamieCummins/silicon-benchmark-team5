# hewitt_slice — retrodiction slice of the Hewitt et al. (2026) experiment archive

Built 2026-08-18 from `pipeline/data/raw/hewitt2026_llm_prediction/` (Code Ocean capsule
9843791, data license CC0-1.0; Nature SI files). Local wrangling only — **no LLM API calls**.
Nothing here is committed to git.

Purpose: machine-usable inputs for a benchmark that feeds each study's verbatim stimuli to
LLMs, asks them to predict treatment effects, and compares against observed effects computed
from the participant-level data.

## Files

| file | contents |
|---|---|
| `contrasts.csv` | 129 treatment-vs-reference contrasts (one row per study x treatment condition x outcome) across 33 studies, with observed means/SDs/Ns, ATE, Welch SE |
| `stimuli.csv` | 97 rows, one per (study, condition) **including reference conditions**, with verbatim stimulus text and any shared `study_context` |
| `scripts/01_extract_from_rds.R` | RDS -> intermediate CSVs (cell stats, hypotheses, prompt pages) |
| `scripts/02_process_pages.py` | prompt-page normalization; stimulus/outcome-page tables; per-study digest |
| `scripts/03_build_slice.py` | selection rules, sampling, contrast computation, MOESM4 validation |
| `intermediate/` | all intermediate tables + `study_summary.txt` (the digest used for classification) + `build_log.txt` (counts printed by the build) |

Reproduce: `LC_ALL=en_US.UTF-8 Rscript scripts/01_extract_from_rds.R`, then
`uv run python scripts/02_process_pages.py`, then `uv run --with openpyxl python scripts/03_build_slice.py`
(R 4.6.0; Python 3.12 via `uv run` from the `pipeline/` project; the UTF-8 locale matters —
under the default C locale R escapes non-ASCII characters as `<U+2019>`).

## Where things come from

- **Observed effects**: participant-level `y` in `codeocean_capsule/data/rct_responses.RDS`
  (71 studies, 134 study-outcomes, 616 condition cells, 233,601 participant-outcome rows).
  Cell n / mean / SD are computed on non-missing `y` in the original response coding;
  `ate = mean_treat − mean_ref`; `se` is the Welch standard error
  `sqrt(sd_t²/n_t + sd_r²/n_r)`. No covariate adjustment, no weights (matches the paper's own
  analysis, which regresses `y ~ condition` — see validation below).
- **Stimuli and outcome wording**: parsed from the `prompt` field of `llm_responses.RDS`
  (gpt-4 rows; 72,120 prompts). Prompts are survey transcripts with a fixed 6-item
  demographic persona block (dropped), one or more stimulus "pages", and a final
  outcome-question page. All 72,120 prompts parsed; after normalizing two prompt-template
  voices ("You choose" vs "Participant X chooses"), every (study, condition, outcome) cell
  had exactly one page-sequence variant; stimulus pages are identical across outcomes within
  condition, and outcome pages identical across conditions within outcome (0 exceptions).
- **Hypotheses**: `RA_hypotheses.RDS` (165 study-outcome-hypothesis rows; each maps
  conditions to a `t_hypothesis` 1 ("treatment") vs 0 ("comparison") side).
- **Study metadata**: `RA_study_features.csv` (`study_label` = study_title; `study_is_tess`
  identifies the 20 Coppock meta-analysis experiments as `FALSE`).

## Selection rules and counts

Applied mechanically in `scripts/03_build_slice.py`; full log in `intermediate/build_log.txt`.

- **Step 0 — universe**: 71 studies / 134 study-outcomes / 616 cells / 482 cell contrasts.
- **Step 1 — prompt availability**: `willer845` absent from `llm_responses.RDS` (no stimuli;
  dropped). Two outcomes lack prompts and are dropped: `Howard823/predictedexpense_nextweek`,
  `Melin1066/salary`. Leaves 70 studies / 129 study-outcomes / 305 conditions.
- **Step 2 — text-stimulus filter** (study-level, hand-coded from
  `intermediate/study_summary.txt`; classification dict with per-study reasons is in
  `03_build_slice.py`): kept 58 studies. Excluded:
  - `question_wording` (manipulation is the wording of the outcome item itself; no
    substantive stimulus): Enos700, KlarBRIEF70, KrupnikovS34, RyanS14.
  - `task_induction` (writing/recall/priming tasks rather than reading a text):
    Howard823, Howat1039, Levendusky741, Merolla843, Rifkin1073, wayne841,
    caprarielloS2, converseS16 (the last two are Coppock studies).
  - `no_prompts`: willer845.
  No study in this archive had image/video/monetary treatments as the primary manipulation
  (wayne841 references a photo and is excluded anyway; see caveats for chart-description cases).
- **Step 3 — reference designation** (per study):
  - **Rule A `named_control`** (16 studies): a unique condition named
    Control/Baseline/Placebo/No narrative/No Prompt. (Regex hits that are *not* controls,
    e.g. Braman751's "Gun Control, ..." topic cells, were excluded by hand; multi-control
    factorials — Bougher893, Harbridge-Yong1032, Iles1294, KlarBRIEF70 — were not given a
    reference by this rule.)
  - **Rule B `hypothesis_t0`** (21 studies): every RA hypothesis for the study has the same
    single-cell t0 side -> that cell is the study-wide reference (all other conditions
    contrast against it). This covers 16 of the 18 kept Coppock studies (the other two are
    Rule A: system_threat's `craig_placebo`, gashS5's `4 [No Prompt]`).
  - **Rule C `hypothesis_pair`** (4 studies: Kennedy1017, Melin1066, Mezzapelle1439,
    Terman1029): hypotheses with different (single-cell) t0 sides -> only the
    hypothesis-defined pairs become contrasts; the reference varies by row.
  - **Ambiguous -> excluded** (17 studies, mostly factorials with no designated baseline and
    no single-cell hypothesis side): Blair1131, Bougher893, Braman751, CalarcoS38, Cohen1099,
    FaheyS78, HamiltonS31, Harbridge-Yong1032, Iles1294, KuruB67, MunschS68, SchaadS62,
    Schnabel903, ShannonS28, Stoker1063, ThorsonS42, relihan1399.
  The paper itself designates **no** reference conditions ("most experiments did not include
  a designated control condition ... effect sizes relative to a randomly-chosen reference
  condition", SI §2.2); the rules above use the only designations that exist in the archive
  (condition names and the RA hypothesis coding) rather than inventing choices.
- **Step 4 — condition-level exclusion**: McCabeS19 `INSURANCE+GOVERNMENT CUE` dropped —
  its prompts in the source archive are corrupted (see caveats).
- **Step 5 — eligible set**: 179 contrasts across 41 studies (18 Coppock + 23 TESS).
- **Step 6 — sampling to target** (target ≤130 contrasts, 25–45 studies): keep **all 18
  Coppock studies** (18 contrasts); shuffle the 23 TESS studies with `random.Random(42)` and
  add each study (all its contrasts) while the running total stays ≤130. Kept 15 TESS
  studies; skipped by budget: Krupnikov719, bucci1408, Craig735, WallaceS9, Kennedy1017,
  AnsonBRIEF60, CorstangeBRIEF69, bolsenM6. **Final: 33 studies, 129 contrasts.**
  Sampling never looked at effect sizes, signs, or significance; observed effects come out
  64 positive / 65 negative, and 57% have |ate/se| < 1.96.

## Column notes (`contrasts.csv`)

- `outcome_text`, `scale_min`, `scale_max`, `scale_labels`: verbatim from the final prompt
  page (the RA transcription of the original survey item). `scale_labels` holds the two
  anchor labels ("1=Strongly Disapprove; 7=Strongly Approve").
- **Orientation**: all means and `ate` are reported on the scale **as displayed in
  `outcome_text`**. For 52 of 129 rows the original data were coded in the opposite
  direction and the archive flipped the scale for prompting (`scale_flip=TRUE`); for those
  rows we transformed `m -> (scale_min+scale_max) − m`, `ate -> −ate` so each row is
  internally consistent. To recover original-coding values, apply the same transform again.
- `hypothesized_direction`: **orientation of the RA-coded hypothesis, not a verified sign
  prediction.** +1 = this row's `condition` is on the hypothesis's t1 ("treatment") side and
  `reference` on its t0 side; −1 = reversed; blank = pair not covered by any hypothesis
  (33 rows). We checked empirically whether t1-minus-t0 differences are predominantly
  positive on either coding and they are not (49/48 split on original coding, 52/36 on the
  displayed coding across the 97 single-cell hypothesis pairs), so the t1/t0 coding cannot
  be interpreted as "hypothesized higher on this scale" in general. Treat it as "which side
  the authors' hypothesis singled out as the treatment of interest".
- `reference_type`: named_control (84 rows) | hypothesis_t0 (18) | hypothesis_pair (27).
  For `hypothesis_t0`/`hypothesis_pair` rows the reference is the RA-designated comparison
  side, which for two-sided persuasion designs (e.g. `immigration` positive vs negative,
  `patriot_act` Pro vs Con) is itself an active message, **not** a pure no-treatment control.
  Filter to `reference_type=="named_control"` if you need strict treatment-vs-control.
- `outcome_text_is_scale_only` (8 rows: ShannonS2, HankinsonS22, dennyS17, gashS5): the
  outcome page contains only the scale instruction; the substantive question stem is at the
  end of `stimulus_text` (faithful to how the survey embedded the question in the vignette).
- Ns are non-missing-`y` counts; the 33 slice studies comprise 57,040 participants in total
  (per-study maximum across that study's outcomes).

## Column notes (`stimuli.csv`)

- One row per (study, condition) of the 33 kept studies, references included (97 rows;
  the corrupted McCabeS19 cue condition is omitted). `source` = `llm_prompts` for all rows
  (the socsci210 fallback was not needed).
- `stimulus_text` = the condition's survey pages before the outcome question, with the
  demographic persona block and prompt-template chooser lines removed, `> ` quoting
  stripped, and dangling "You answer the question"/"You choose" scaffold lines replaced by
  `[participant responds]`. Pages are separated by blank lines.
- `study_context` = pages shown identically to **every** condition in that study (shared
  preamble/vignette stem), in display order; `stimulus_text` then contains only
  condition-specific pages. 6 reference conditions have an empty `stimulus_text` because
  they truly saw nothing beyond the shared context (Campbell1308, Connors1226, McCabeS19,
  McGinty730, senS81 controls; turagaS11 condition `3 [3]`, whose page is a strict subset of
  condition `1 [1]`'s).
- `flags` (comma-separated, condition-level):
  - `bracket_alternation` — unresolved `[A]/[B]` piped-text alternatives kept by the RA
    transcription (e.g. `immigration`: "[Nikolai Vandisnky]/[Jose Sanchez]"); the
    sub-manipulation those brackets encode was not varied in the archived conditions.
  - `html_tags` — literal `<u>`/`<i>` emphasis tags from the survey transcription.
  - `layout_placeholder` — markers like `[SPACE]`, `[TEXT BOX; ...]`, `[BLANK SPACE]`.
  - `embedded_response_prompt` — the stimulus contains an interim question the participant
    answered mid-treatment (Haaland874's incentivized guess, marked `[participant responds]`;
    McCabeS19's "Have you had this experience?" item). The static text cannot carry those
    responses.
  - `chart_or_image_reference` — the text describes or refers to a chart/plot that
    participants saw (Williamson859, senS81 — the numeric content is present in the text
    itself; jacobsenS7's report cards and Silverman1035's risk table are rendered as plain
    text tables and carry no flag).

## Validation

1. **Against Nature source data** (`nature_si/MOESM4.xlsx`, sheet `panelA` = Fig. source
   data with one `estimate` per (study, reference_condition, condition, outcome) in
   rescaled [0,1] units of the original coding): converting our `ate` back to original
   coding and dividing by the scale range reproduces their estimate for **all 129 slice
   contrasts** (all present in panelA); max |difference| = 4.5e-7 (float round-trip only).
   Spot checks printed by the build: Campbell1308 Democratic story/Gender Roles
   (0.004152 = theirs), Connors1226 Friends Reminder/rate_democrats (−0.037493),
   FarrowS6 (−0.008282), Haaland874 (−0.005047), HankinsonS22 (−0.106192).
   **Their published per-contrast estimates are raw differences in means** (their
   `run_regression` fits `lm(y ~ condition)` with no covariates and no weights), so no
   adjustment discrepancy exists — agreement is exact up to floating point.
2. **Stimulus spot-checks** (full texts read): `immigration/negative` (complete AP-style
   article; clean; bracket-alternation flag), `Silverman1035/Treatment 2` (complete
   correction text incl. risk table; `[SPACE]`/`</u>` artifacts flagged),
   `McGinty730/Successful Treatment Engagement & Recovery` (complete narrative; clean),
   plus `Haaland874/TREATMENT` (embedded guess interaction rendered with
   `[participant responds]`). No demographic scaffolding, no "Interviewer:" artifacts, no
   response-option lists glued onto stimuli in any checked text.

## Caveats a downstream user must know

1. **Upstream data error found**: in the source archive, *all* 1,680 `llm_responses.RDS`
   prompts for McCabeS19 `INSURANCE+GOVERNMENT CUE` (every model) contain the stimulus of a
   different study (Howat1039's stereotype-listing task) instead of the insurance+cue
   vignette — i.e., the paper's LLM predictions for that cell were generated from the wrong
   stimulus. The condition is excluded here (McCabeS19's other 4 conditions are kept).
2. **References are not always pure controls** — see `reference_type` above.
3. **`hypothesized_direction` is an orientation**, not a sign prediction (details above).
4. **Scale orientation**: 52/129 rows are sign-flipped relative to the original data coding
   (`scale_flip=TRUE`); everything in `contrasts.csv` is consistent with `outcome_text`,
   but comparisons against the raw RDS must apply the flip.
5. **Transcription quirks kept verbatim** (flagged): bracket alternations, `<u>`/`[SPACE]`
   placeholders, chart descriptions. One Silverman1035 treatment page lacks the headline
   that its sibling conditions carry (source-archive inconsistency). Krupnikov719 (excluded
   by sampling, not by rules) is a procedural/disclosure manipulation; if it ever re-enters
   a larger slice, note its "income" outcome is a demographic self-report.
6. **Melin1066 and Howard823 were excluded from the paper's own prediction analyses**
   (`load_archive1_results.R` drops them); Melin1066 is nevertheless in this slice because
   its prompts and participant data are complete for 2 of 3 outcomes. Its published-paper
   MOESM4 rows exist, and our estimates match them.
7. **Coverage vs the paper**: the paper analyzes 469 effects under randomly sampled
   references; this slice is a curated, reference-designated subset (129 contrasts) and is
   **not** a reproduction of their headline analysis sample.
8. Melin1066 condition names contain curly quotes (`"Cosmetics/Automotive"` etc.) — join on
   exact UTF-8 strings.
9. TESS studies excluded only by the seed-42 budget (not by any rule) are listed in Step 6;
   to enlarge the slice, raise `TARGET_MAX` and re-run — no other choices change.
