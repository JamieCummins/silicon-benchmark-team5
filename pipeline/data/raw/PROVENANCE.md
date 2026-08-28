# PROVENANCE — raw datasets for retrodiction/calibration benchmark

Download date for everything below: **2026-08-17** (Europe/Zurich).
All downloads were anonymous HTTP/git — no registration or authentication was used.
This directory is gitignored; nothing here is committed.

---

## 1. `vlasceanu2024_climate/` — Vlasceanu et al. (2024) climate intervention tournament

**Citation:** Vlasceanu, M., Doell, K. C., Bak-Coleman, J. B., et al. (2024). Addressing
climate change with behavioral science: A global intervention tournament in 63 countries.
*Science Advances*, 10(6), eadj5778. https://doi.org/10.1126/sciadv.adj5778

**Sources used (exact URLs):**
- Analysis code (Bak-Coleman): `git clone https://github.com/josephbb/ManyLabsClimate` → `ManyLabsClimate/`
- Analysis code (Vlasceanu, partial mirror): `git clone https://github.com/mvlasceanu/ClimateTournament` → `ClimateTournament/`
- Participant-level data: Zenodo record 10345806 (DOI 10.5281/zenodo.10345806),
  file `data63.xlsx` via https://zenodo.org/api/records/10345806/files/data63.xlsx/content
  → `data63.xlsx` (9.4 MB). **License: CC-BY-4.0** (per Zenodo record).
- OSF project "International Collaboration to Understand Climate Action", node `ytf89`
  (https://osf.io/ytf89/), files fetched via `https://files.osf.io/v1/resources/ytf89/providers/osfstorage/<id>`:
  - `osf_materials/usa_1.qsf` (id 658c19be7094e94023a17470) — US Qualtrics survey ("USA Climate master survey MSI")
  - `osf_materials/usa_2.qsf` (id 658c19c003446142a568f546) — US Qualtrics survey ("2022_08_22_ClimateCollab-MASTER")
  - `osf_materials/master_survey.pdf` (id 657389d9666e1008b91a7c15) — 80-page master survey (note: embedded fonts make text extraction garbled; use QSFs for machine-readable text)
  - `osf_materials/intervention_adaptation_manual.pdf` (id 66c5f10b212dbbf5870b5426)
  - `osf_materials/codebook.xlsx` (id 659559f17094e97327a1755d) — codebook for the cleaned datapaper dataset
  - `data_countries.csv` (id 65956e327094e973a0a174bd, 160 MB) — full cleaned dataset from the companion data paper (more variables than data63.xlsx, incl. many timers/covariates)
- Open-access paper PDF: https://pure.uva.nl/ws/files/168311163/sciadv.adj5778.pdf → `sciadv.adj5778.pdf`

**Not obtained:** journal supplementary PDF `sciadv.adj5778_sm.pdf`
(https://www.science.org/doi/suppl/10.1126/sciadv.adj5778/suppl_file/sciadv.adj5778_sm.pdf
is behind a Cloudflare browser challenge; the PMC mirror
https://pmc.ncbi.nlm.nih.gov/articles/instance/10849597/bin/sciadv.adj5778_sm.pdf is behind a
JS proof-of-work challenge). Non-blocking: verbatim stimuli come from the QSFs; intervention
summaries are in the main paper.

**Derived files created here (documented computations):**
- `compute_us_ates.py` → `us_ates.csv` — 88 rows: {US, ALL} × 11 interventions × 4 outcomes.
  Simple difference in means vs `condName=="Control"` on participant-level scores
  (belief = mean Belief1–4, policy = mean Policy1–9, share = SHAREcc 0/1, wept = WEPTcc 0–8),
  Welch SE, Ns, and condition means/SDs. US = `Country=="Usa"` (N=8,253; control n=669).
- `extract_us_stimuli.py` → `stimuli_us_extracted/` — 30 plain-text files, one per Qualtrics
  block of `usa_2.qsf` (HTML stripped, includes response options). Block→condName mapping is
  in the script header; verify before final use (mechanical extraction, includes timers/probes).

**Usability assessment:**
- (a) Verbatim stimuli: **YES** — full English intervention texts in `usa_1.qsf`/`usa_2.qsf`
  (11 interventions + control distracter (Great Expectations passage) + outcome item wording);
  also extracted to `stimuli_us_extracted/`.
- (b) Computable ATEs with SEs: **YES** — participant-level `data63.xlsx` (N=59,440, 63 countries);
  computed in `us_ates.csv`.
- (c) US sample: **YES** — N=8,253, 669 control; per-cell n ≈ 530–730.
- (d) Usable contrasts: **44 US contrasts** (11 interventions × 4 outcomes: belief, policy,
  sharing (binary), WEPT behavioral task). Same again for the 63-country pooled sample.
- Caveats: paper's preregistered estimates use mixed models (item/participant/country REs), so
  our simple ATEs differ slightly from published coefficients; SHAREcc has more missingness
  (item shown conditionally); WEPT effects are near zero/negative (consistent with the paper).

---

## 2. `hewitt2026_llm_prediction/` — Hewitt, Ashokkumar, Ghezae & Willer experiment archive

**Citation:** Hewitt, L., Ashokkumar, A., Ghezae, I., & Willer, R. (2026). Large language
models can predict the results of social science experiments. *Nature*, 656, 115–122.
https://doi.org/10.1038/s41586-026-10742-x (earlier 2024 working paper: "Predicting results
of social science experiments using large language models"; preprint https://osf.io/preprints/psyarxiv/3svep_v1).
Demo: https://www.treatmenteffect.app/

**Sources used (exact URLs):**
- **Code Ocean capsule** (per the paper's Data/Code availability statements:
  https://codeocean.com/capsule/9843791/tree/v1). The capsule's git remote is anonymous-clonable:
  `git clone https://git.codeocean.com/capsule-9843791.git` → `codeocean_capsule/`
  (the Code Ocean web UI itself is a JS app and its REST endpoints 403 for scripts;
  the git clone contains the full `code/` and `data/` trees).
  **Licenses:** `data/LICENSE` = **CC0-1.0**; `code/LICENSE` = **MIT** (© 2026 Luke Hewitt).
- **Nature supplementary files** (freely served by Springer): 
  `https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10742-x/MediaObjects/41586_2026_10742_MOESM{1..8}_ESM.{pdf,xlsx}`
  → `nature_si/MOESM1.pdf` (48-pp. Supplementary Information; full text extracted to
  `MOESM1_supplementary_information.txt`), `MOESM2.pdf`, `MOESM3.pdf` (103-pp. peer-review file),
  `MOESM4–8.xlsx` (source data for Fig. 1, Fig. 2, Fig. 4, ED Fig. 2, ED Fig. 3).

**Contents of `codeocean_capsule/data/` (formats):**
- `rct_responses.RDS` — nested tibble, 134 study×outcome rows over **71 studies**; the `data`
  list-column holds participant-level rows: `y` (numeric outcome), `condition.name`, and
  demographics (GENDER, race_4, pid_3, age_5, EDUC4, ideo_3). 233,601 participant-outcome rows;
  616 conditions total → **482 treatment-vs-reference cell contrasts** (paper analyzes 469
  effects after exclusions). This is the primary archive: 50 TESS + 20 Coppock-meta-analysis
  experiments, all **pre-registered, nationally representative US** survey experiments.
- `llm_responses.RDS` — 504,840 rows × 12 cols (21 MB compressed / ~1.2 GB in memory): model
  (gpt-4, gpt-3.5-turbo, davinci-002, babbage-002, deepseek-v3, gemma-3-27b, gpt-oss-120b),
  study, condition.name, outcome.name, **`prompt`** (full simulation prompt embedding the
  **verbatim condition stimulus text and outcome question wording/scale**), samples, expectation,
  weight, scale metadata. Covers the 70 archive-1 studies.
- `RA_study_features.csv` — 70 studies: publication-date flags, discipline, title, authors, links.
- `RA_outcome_features.csv` — outcome-level flags (e.g. `outcome_existing_attitude`).
- `RA_hypotheses.RDS` — 165 author-hypothesized contrasts (study, outcome, hypothesis).
- `megastudies.RDS` — secondary archive: 64 dataset×outcome rows (15 megastudies incl.
  SDC/Voelkel2025, Milkman2021/2022/2023, Broockman persuasion (per policy × direction),
  Tappin (immigration/UBI × direction), DellaVigna & Pope, Allen2023, Doell, Zickfeld,
  Saccardo2024, Mason, Goldwert); nested df per row with `condition.name`, `estimate.rct`,
  `prediction.expert`, `prediction.gpt-4`, plus V (covariance) and N columns. ~600 effects.
- `individual_expert_predictions.rds`, `forecasting_responses.RDS` — expert/layperson forecasts.
- `gpt_author_recognition.csv` — robustness check data.
- `survey_data/` — survey of 460 social scientists (4 CSVs).
- `code/` — R replication pipeline (`00_run_all_analyses.R`, `0_minimal_example.R`, etc.) and
  `PLOT_DATA_CODEBOOK.docx` describing per-figure source-data columns.

**Usability assessment:**
- (a) Verbatim stimuli: **YES for the 70-study primary archive** — embedded verbatim in the
  `prompt` field of `llm_responses.RDS` (condition text + outcome wording + response scale,
  formatted as a survey transcript; demographic preamble varies per simulated respondent).
  There is no separate clean "stimuli.csv"; stimuli must be parsed out of prompts (structure is
  regular: "The next page of the survey says:" blocks), or re-derived from TESS originals via
  links in `RA_study_features.csv`. **NO for the megastudies archive** (only condition *names*,
  some renamed to protect unpublished work).
- (b) Computable ATEs with SEs: **YES** — participant-level `y` per condition in
  `rct_responses.RDS` → condition means/SDs/Ns and any contrast; megastudies provide point
  estimates + variance info but not participant data.
- (c) US sample: **YES** — all primary-archive experiments are US nationally representative
  (TESS/NORC etc.); demographics included.
- (d) Usable contrasts: 482 cell contrasts (469 analyzed effects) across 134 study-outcomes;
  165 author-hypothesized contrasts flagged in `RA_hypotheses.RDS`. The message/persuasion
  subset: the 20 Coppock meta-analysis experiments (classic text-message persuasion → policy
  attitude; identifiable via `RA_study_features.csv: study_is_tess==FALSE`) plus many TESS
  vignette experiments; `RA_outcome_features.csv: outcome_existing_attitude` helps filter.
- GPT-4/expert/layperson predictions ship alongside → useful baselines for calibration.

---

## 3. `socsci210/` — SocSci210 (secondary evaluation set, Stanford "Socrates" release)

Note: the task brief associated "SocSci210" with source 2; it is in fact a separate release
by a Stanford HCI team, built from the same TESS repository (so it overlaps archive 1).

**Citation:** Suh, J., et al. (2025). Finetuning LLMs for Human Behavior Prediction in Social
Science Experiments. arXiv:2509.05830. Project: https://stanfordhci.github.io/socrates.

**Source used:** Hugging Face dataset `socratesft/SocSci210`
(https://huggingface.co/datasets/socratesft/SocSci210), files via
`https://huggingface.co/datasets/socratesft/SocSci210/resolve/main/...`:
- `data/train-00000..00016-of-00017.parquet` (17 shards, 1.4 GB total)
- `metadata/{participant,task,condition}_mapping.json` (train/eval split definitions)
- `README.md`
**License:** none stated on the HF dataset card (README has no license field) — check before
redistribution; underlying TESS data are public.

**Format (parquet schema):** sample_id, participant, demographic (struct of 16 fields),
**`stimuli` (verbatim condition text)**, response (int), condition_num, task_num,
`prompt` (full simulation prompt), `reasoning` (model-generated), study_id (TESS id, e.g. "9nphm").
~2.9M individual responses, 400,491 participants, 210 studies, 1,194 conditions, 1,197 outcomes.

**Usability assessment:** (a) verbatim stimuli **YES** (`stimuli` column); (b) ATEs+SEs **YES,
computable** from individual `response` by condition (means/SDs/Ns; response scales must be
inferred per task from the prompt text); (c) US samples **YES** (TESS = US nationally
representative); (d) ~1,000+ condition-vs-reference contrasts across 210 studies. Overlaps the
Hewitt primary archive (both draw on TESS) — deduplicate by TESS study id/title when combining.

---

## 4. `voelkel2024_sdc/` — Strengthening Democracy Challenge megastudy (optional source)

**Citation:** Voelkel, J. G., Stagnaro, M. N., Chu, J., et al. (2024). Megastudy testing 25
treatments to reduce antidemocratic attitudes and partisan animosity. *Science*, 386, eadh4764.
https://doi.org/10.1126/science.adh4764

**Sources used:** OSF project "The Strengthening Democracy Challenge" (node `jzbnt`,
https://osf.io/jzbnt/), component "Main Survey" (node `2sv7p`), files via
`https://files.osf.io/v1/resources/{jzbnt|2sv7p}/providers/osfstorage/<id>`:
- `SDC_Data_Anonymized.csv` (9.2 MB; 35,252 rows × 70 cols) and `SDC_Data_Recoded.csv` (25 MB)
- `SDC_Data_Intervention_Names.csv` (27 conditions: Null Control, Alternative Control, 25 treatments)
- `SDC_Data_Outcome_Names.csv` (outcomes incl. PA = partisan animosity, ADA = support for
  undemocratic practices, + weights variable names)
- `SDC_Questionnaire.pdf` (374 pp. — includes intervention content/scripts) and
  `SDC_Questionnaire_Qualtrics.qsf` (machine-readable)
- `ReadMe.pdf`, `SDC_Supplementary_Materials.pdf`
- R analysis scripts were listed under `2sv7p` Code/ but not downloaded (names/ids documented
  in the OSF folder listing; fetch on demand).
**License:** no explicit license shown via API for these files; OSF default terms apply — check
the project wiki before redistribution.

**Usability assessment:** (a) stimuli **PARTIAL** — many of the 25 treatments are videos/
interactive flashcards; text/scripts are in the questionnaire PDF/QSF, but pure-text stimuli
exist only for a subset; (b) ATEs+SEs **YES** — participant-level anonymized data with
condition and survey weights; (c) US sample **YES**; (d) up to 25 treatments × 2 primary
outcomes (+ secondary outcomes) vs two controls. Note: preprocessed SDC effects + GPT-4/expert
predictions are also in `hewitt2026_llm_prediction/codeocean_capsule/data/megastudies.RDS`.

**Milkman flu-vaccine text-message megastudy (noted, NOT downloaded):**
Milkman et al. (2021) PNAS 118(20):e2101165118 (patient megastudy, 19 messages) and
(2022) PNAS 119(6):e2115126119 (Walmart pharmacy megastudy, 22 messages).
Message texts are in the SI appendices (e.g.
https://www.pnas.org/doi/suppl/10.1073/pnas.2101165118/suppl_file/pnas.2101165118.sapp.pdf —
Cloudflare-blocked for scripted download; fetch via browser). Participant-level data are NOT
public (health-system/Walmart records; available on request per the papers). Effect estimates
incl. GPT-4/expert predictions for Milkman2021/2022/2023 are in the Hewitt `megastudies.RDS`.

---

## Cross-cutting notes / gaps

1. **No LLM APIs were called.** All artifacts came from web search, HTTP(S) downloads, OSF/Zenodo/
   Hugging Face APIs, git clones, and one headless-browser session to read the Code Ocean UI
   (the actual data came via the capsule's public git remote).
2. Gaps: (i) Science Advances SM PDF for source 1 (blocked; not needed — stimuli in QSFs);
   (ii) verbatim stimuli for the Hewitt *megastudies* archive are not public in that release —
   for SDC we recovered materials from OSF directly, for Milkman only via the (blocked) PNAS SI;
   (iii) SocSci210 has no stated license; (iv) Code Ocean "Published Result" outputs (456 MB,
   incl. `results/plot_data/`) were not pulled — equivalent figure source data already obtained
   via Nature MOESM4–8, and outputs are regenerable from code+data.
3. Suggested benchmark subsets: Vlasceanu US (44 clean text-intervention contrasts, this dir);
   Hewitt archive 1 (482 contrasts, stimuli parseable from prompts; message/persuasion subset =
   20 Coppock studies + TESS vignette studies); SocSci210 for scale (dedupe vs Hewitt TESS).
