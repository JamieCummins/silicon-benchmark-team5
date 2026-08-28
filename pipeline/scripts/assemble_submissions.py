"""Assemble the four submission repos locally (NO publishing — Zenodo/GitHub
happen only in the Aug 28-31 window).

  uv run python scripts/assemble_submissions.py

Creates submissions/{t1_primary,t2_primary,t3_primary,t3_secondary1}/ as full
template copies with our predictions, metadata, registration forms, and raw
logs; runs make check + manifest + zenodo_citation in each.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
REPO = PIPE.parent
TEMPLATE = REPO / "template"
SUBS = REPO / "submissions"
RUNS = PIPE / "runs"

TEAM = "team_5"
CONTACT = "jamie.cummins@unibe.ch"
CREATORS = [{"name": "Cummins, Jamie", "affiliation": "University of Bern", "orcid": ""}]
CODE_REPO = "https://github.com/jamie-cummins/silicon-benchmark-team5 (made public at deposit; see DEPOSIT_CHECKLIST)"

MODELS_T3A = ["gpt-5.6-terra (OpenAI API)", "gpt-5.6-luna (OpenAI API)",
              "openai/gpt-oss-120b (Groq API)", "deepseek/deepseek-v3.2 (OpenRouter)",
              "meta-llama/llama-4-maverick (OpenRouter)"]
MODELS_T3B = ["deepseek/deepseek-v3.2 (OpenRouter, logprobs)",
              "openai/gpt-oss-120b (OpenRouter, logprobs)"]

# ---------------- registration form text ----------------

COMMON = {
    "0.1": f"Team 5 — solo: Jamie Cummins, University of Bern. Contact: {CONTACT}.",
    "A.2": "Fully automated at prediction time; no human edited, selected, or overrode any "
           "model output or predicted value. Human decisions were limited to pre-specified "
           "pipeline design and literature-derived prior values, all fixed before the target "
           "elicitation runs.",
    "I.1": "No funding or in-kind support from LLM-interested entities. API usage paid "
           "personally at standard public rates (OpenAI, Groq, OpenRouter). No relationships "
           "with model providers beyond ordinary paid API access.",
    "I.3": "I attest that no team member accessed, solicited, or was shown any human outcome "
           "data from this study, including pilots and the three disclosed talks, before the "
           "prediction lock. A signed declaration accompanies the deposit email.",
    "J.1": "Design space was explored exclusively by retrodiction on public studies (never on "
           "the target): a 72-combination grid over three aggregation hyperparameters "
           "(prior weight x shrinkage x trim) scored on two grounds — the Vlasceanu et al. "
           "2024 US arm (11 interventions x 4 outcomes; effects computed from public Zenodo "
           "microdata) and a 129-contrast slice of the Hewitt et al. 2026 archive (Code "
           "Ocean, CC0). Selection criterion: mean RMSE rank across BOTH grounds "
           "(pre-registered internally before ground-2 elicitation; frozen-parameter "
           "confirmation run first). Model roster and prompt-variant count were fixed from "
           "leave-one-model-out and split-half reliability on ground 1. Approximately 2 "
           "small pilots (~30 calls) preceded each elicitation run.",
    "K.3": "Whole project (all four entries + retrodiction tuning): ~5,500 API calls, "
           "~30M tokens, ~$14 total, on laptop + public APIs; no GPUs.",
    "I.4": "Model training cutoffs (2025 for most roster members; gpt-5.6 family 2026) "
           "predate any release of this benchmark's results (unreleased). The benchmark's "
           "public materials (call, template) appeared mid-2026 and could in principle be in "
           "the newest models' training data; they contain no outcome information. The "
           "target study's preliminary-results talks were never searched for or accessed; "
           "web access was disabled for all prediction calls (plain chat completions).",
}


def reg_form(entries: dict[str, str]) -> str:
    d = {**COMMON, **entries}
    order = ["0.1", "0.2", "0.3", "0.4", "0.5", "A.1", "A.2", "B.1", "B.2", "B.3", "B.4",
             "B.5", "B.6", "B.7", "C.1", "C.2", "C.3", "D.1", "D.2", "D.3", "E.1", "E.2",
             "E.3", "F.1", "F.2", "G.1", "G.2", "G.3", "H.1", "H.2", "I.1", "I.2", "I.3",
             "I.4", "J.1", "K.1", "K.2", "K.3", "L"]
    lines = ["# Silicon Sample Benchmark — method registration form (completed)",
             f"\nTeam: {TEAM} · Disclosure class: A (all items public)\n"]
    for k in order:
        if k in d:
            lines.append(f"**{k}** — {d[k]}\n")
    return "\n".join(lines)


T3A_TEXT = {
    "0.2": "A 'calibrated silicon crowd': five LLMs from five model families each forecast, "
           "in a superforecaster frame with reference-class anchors, the average treatment "
           "effect of every intervention on every outcome in native units. Forecasts are "
           "pooled by trimmed mean, blended with literature-derived per-outcome prior mean "
           "effects, and passed through unshrunk — hyperparameters fixed in advance by "
           "retrodiction on two public experiment corpora.",
    "0.3": "Tier 3, primary. Family: direct effect forecast; multi-model + multi-prompt "
           "ensemble; literature-conditioned via in-prompt reference-class anchors and "
           "prior blending.",
    "0.4": "(1) parse benchmark survey materials -> (2) build 12 prompt variants (3 framings "
           "x 2 anchor placements x 2 outcome orders) per intervention -> (3) elicit "
           "per-intervention ATE JSON from 5 models (12 variants each; the reasoning model 6) "
           "-> (4) parse/validate -> (5) per-cell 20%-trimmed mean -> (6) outcome-level "
           "centering: 0.5x literature prior + 0.5x crowd mean -> (7) within-outcome "
           "deviations retained at lambda=1.0 -> (8) signed-epsilon floor (no exact zeros) "
           "-> (9) write 208-cell file.",
    "0.5": "Full coverage confirmed: 16 interventions x 13 outcomes = 208 effect estimates.",
    "A.1": "LLMs used only for stage (3), effect forecasting. Parsing, aggregation, "
           "calibration are deterministic code.",
    "B.1": "See metadata.json models list; exact API identifiers: gpt-5.6-terra, "
           "gpt-5.6-luna (OpenAI); openai/gpt-oss-120b (Groq); deepseek/deepseek-v3.2, "
           "meta-llama/llama-4-maverick (OpenRouter).",
    "B.2": "Provider HTTP APIs (OpenAI-compatible chat completions), stateless single-turn "
           "calls, 2026-08-18 (retrodiction runs 2026-08-17/18).",
    "B.3": "Provider-default sampling (no temperature/top-p passed); max output tokens "
           "8192; one completion per (model x variant x intervention) cell, one resample "
           "retry on parse failure; no seeds (hosted APIs).",
    "B.4": "No fine-tuning, retrieval, tools, or web access. Literature conditioning occurs "
           "only through fixed reference-class anchor text in prompts and the prior blend "
           "in aggregation.",
    "B.5": "None; every call independent.",
    "B.6": "N/A (all hosted APIs).",
    "B.7": "Ensemble = 5 models x 12 prompt variants (6 for gpt-5.6-terra), pooled per cell "
           "by 20%-per-side trimmed mean, then the aggregation rule in G.3/F.2.",
    "C.1": "Verbatim prompt templates and variant generator deposited in the code repo "
           "(src/silicon/prompts_t3.py); design pre-specified, refined only during "
           "retrodiction pilots, frozen before target elicitation.",
    "C.2": "Three system framings (forecaster / methodologist / meta-analyst), verbatim in "
           "the deposited code.",
    "C.3": "Multi-prompt variation implements the measurement view of silicon sampling "
           "(prompt-specific bias cancels across variants; Cummins & Wulff): framings, "
           "anchor placement, and outcome order are deliberately varied and averaged.",
    "D.1": "N/A (no personas; direct effect forecasting).",
    "D.2": "N/A.", "D.3": "N/A.",
    "E.1": "Stimuli presented verbatim, including control filler texts and multi-page "
           "sequences (page breaks marked). State-contingent arm: all four case texts "
           "shown with approximate population exposure shares (56/24/19/1%), model asked "
           "for the mixture ATE.",
    "E.2": "All 13 outcomes defined (verbatim item wordings + scales + polarity notes, e.g. "
           "distrust reverse-scored) in one call per intervention; outcome order randomized "
           "in half the variants.",
    "E.3": "Structured JSON output: point ATE per outcome in native units.",
    "F.1": "One generation per cell (+<=1 retry); aggregation seeds fixed in code; API "
           "generations not reproducible bit-for-bit (documented).",
    "F.2": "Submitted value = 0.5*prior_o + 0.5*mean_o(crowd) + 1.0*(trimmed cell mean - "
           "mean_o(crowd)); |value| floored at signed epsilon (0.02 slider pts / $0.002 / "
           "0.0002 probability) with the crowd's sign.",
    "G.1": "None.",
    "G.2": "JSON parsing with think-tag stripping and a glitch-token sanitizer; per-outcome "
           "sanity bounds; failed cells resampled once then dropped (847/864 cells "
           "retained; median 52 forecasts per cell).",
    "G.3": "Aggregation hyperparameters (prior weight 0.5, shrink 1.0, trim 0.2) fit by "
           "cross-ground retrodiction RMSE rank (J.1). Per-outcome priors are "
           "literature-derived point values (deposited: reference/outcome_priors.md) from "
           "published meta-analyses and megastudies (Rode 2021; van Stekelenburg 2022; "
           "Većkalov 2024; Vlasceanu 2024; Voelkel 2026; trust-intervention and "
           "charitable-giving literatures).",
    "H.1": "None.",
    "H.2": "In-context: only the benchmark's own public materials (verbatim) and the fixed "
           "anchor paragraph. No retrieval.",
    "I.2": "Human data used for hyperparameter fitting and priors only (never respondent "
           "simulation): Vlasceanu et al. 2024 microdata (Zenodo, CC-BY), Hewitt et al. "
           "2026 archive (Code Ocean, CC0), plus published summary statistics cited in the "
           "priors document. All public; none from the target study.",
    "K.1": "Full pipeline code deposited (code_repository in metadata.json); seeds and "
           "deterministic aggregation documented; API keys excluded.",
    "K.2": "Complete raw model responses (JSONL, every call with full prompt + response + "
           "usage) in raw_data_deposit/, public.",
    "L": "Class A — every item public.",
}

T3B_TEXT = {**T3A_TEXT,
    "0.2": "A 'readout crowd' (mechanical contrast entry): open-weight models are asked, for "
           "each condition x outcome, for the predicted average response as a single digit, "
           "and the full answer distribution is READ from token log-probabilities in one "
           "call — no sampling, no personas, no priors, no calibration. ATE = readout "
           "expectation under condition minus pooled control. Deliberately uncorrected to "
           "quantify what calibration adds relative to the primary Tier-3 entry.",
    "0.3": "Tier 3, secondary-1. Family: direct forecast via log-probability readout; "
           "2-model x 4-framing ensemble; zero-shot, no literature conditioning.",
    "0.4": "(1) 22 text units (16 interventions, weather split into 4 state cases, 3 control "
           "fillers) -> (2) per unit x outcome x framing x model, ask for the average "
           "response as one digit 0-9 -> (3) read top-20 first-token log-probabilities, "
           "renormalize over digit tokens, take the expectation -> (4) map digits to native "
           "scales (tens digit for 0-100; dollars; per-10 signup) -> (5) average over "
           "models x framings -> (6) weather cases recombined by population shares; ATE vs "
           "mean of the three control fillers.",
    "B.1": "deepseek/deepseek-v3.2 and openai/gpt-oss-120b via OpenRouter with "
           "require_parameters (hosts returning full top-20 logprobs).",
    "B.3": "Single call per cell, max 8 output tokens (2048 for the reasoning model), "
           "logprobs=true top_logprobs=20; no sampling used for the estimate (readout of "
           "the answer distribution).",
    "B.7": "2 models x 4 framings averaged with equal weights; no trimming.",
    "C.3": "Framings vary the observer role (simulator / pollster / analyst / minimal); "
           "readout-not-sample design follows Cummins & Wulff: one call's answer "
           "distribution is the quantity of interest.",
    "E.2": "One outcome per call (single-item context); scales mapped to single digits "
           "(multi-digit answers break token readout).",
    "E.3": "Token log-probability readout; renormalized over in-scale digit tokens; "
           "digit expectation mapped linearly to native units.",
    "F.2": "Equal-weight mean of per-call expectations across models and framings; no "
           "priors, no shrinkage, no epsilon rule (values continuous).",
    "G.2": "Cells with <0.5 digit probability mass would be dropped and retried; none "
           "occurred. ~95% cell coverage after retries; missing cells absorbed by "
           "model/framing averaging.",
    "G.3": "None — deliberately uncalibrated (the entry's purpose).",
    "I.2": "None used in this entry's pipeline (no priors, no tuning).",
}

T1_TEXT = {**T3A_TEXT,
    "0.2": "A 'calibrated cohort': 18,000 synthetic respondents whose demographics come from "
           "census microdata, whose baseline response distributions are calibrated to public "
           "reference surveys (ANES thermometer shapes incl. integer heaping, TISP trust-item "
           "structure, CCAM partisan geometry), and whose treatment effects implement the "
           "team's Tier-3 crowd-forecast vector with headroom-based heterogeneity. LLMs "
           "forecast effects; individuals are synthesized statistically — measurement plus "
           "design calculation, not roleplay.",
    "0.3": "Tier 1, primary. Family: measurement-calibrated statistical synthesis; LLM "
           "ensemble for treatment effects (no per-respondent LLM simulation); "
           "literature- and reference-survey-conditioned.",
    "0.4": "(1) sample 18,000 profiles from ACS 2023 PUMS (weighted joint draw; panel "
           "education skew; party assigned via CCAM party x demographics raked to NPORS "
           "margins; state kept for the state-adaptive arm) -> (2) draw latent construct "
           "scores from a multivariate normal whose correlations come from TISP/CCAM; "
           "party/education/age/gender offsets from reference surveys -> (3) balance-correct "
           "latents across arms -> (4) inject the Tier-3 ATE vector as latent shifts with "
           "headroom weighting + person-level heterogeneity; per-cell Newton feedback makes "
           "realized arm contrasts match the vector (max dev 0.03 pts) -> (5) render to "
           "observed scales: ANES-2016-regime heaped 0-100 sliders, item-level simulation "
           "for all composites, two-part donation model, Bernoulli signup -> (6) validate "
           "(distributions, heaping, correlations, group gaps, composite consistency).",
    "0.5": "Full coverage: 18,000 respondents; 2,000 control + 16 x 1,000; all 13 outcomes "
           "(12 trust items + composites per schema).",
    "A.1": "LLMs only in the Tier-3 effect-forecast stage feeding step (4) (see the team's "
           "T3 primary registration; same crowd, same logs, deposited here too). Profile "
           "construction, baselines, rendering: deterministic statistical code, no LLM.",
    "D.1": "Profiles: weighted draw from ACS 2023 1-year PUMS (18+, n=2.77M records), "
           "banded to the benchmark's exact levels; documented opt-in-panel education "
           "skew; benchmark quotas (gender x age, gender x race) hold by construction of "
           "the weighted draw; party from a CCAM-microdata x NPORS-margin raked table; "
           "condition assignment random, exactly 1,000/arm and 2,000 control.",
    "D.2": "No verbalization — respondents are never shown to an LLM.",
    "D.3": "18,000 profiles, no reuse, no weighting beyond the sampling design.",
    "E.1": "N/A at respondent level. State-adaptive arm: each treated profile's ACS state "
           "maps to its preregistered case; effects applied uniformly across cases.",
    "E.2": "N/A (no survey walk-through; response structure comes from the calibrated "
           "generative model).",
    "E.3": "N/A.",
    "F.1": "Fully deterministic given documented numpy seeds (in deposited code); "
           "regeneration reproduces the file bit-for-bit.",
    "F.2": "N/A at respondent level; Tier-3 aggregation rule as in the T3 primary form.",
    "G.2": "Generator-level validation: realized arm contrasts within 0.03 native units of "
           "the target vector; control baselines within ~2 pts of literature anchors; "
           "heaping 85-87% on multiples of 5; trust inter-item r=.59 (TISP band); D-R trust "
           "gap 20.8 pts; composite consistency exact.",
    "G.3": "Baseline levels/shapes calibrated to ANES/TISP/CCAM/GSS/Pew (public); effect "
           "vector calibrated as per the T3 primary form (retrodiction).",
    "I.2": "Reference surveys used for calibration: TISP (OSF), ANES 2016/2020 (public "
           "Dataverse replication copy), CCAM (OSF), GSS 2024, ACS 2023 PUMS, Pew/NPORS "
           "published toplines; plus the two retrodiction corpora (see T3 form). None from "
           "the target study.",
    "K.2": "Raw logs of every LLM call feeding the effect vector (JSONL) in "
           "raw_data_deposit/, plus the deterministic generator code; public.",
}

T2_TEXT = {**T1_TEXT,
    "0.2": "Cell-level statistics derived from the same generator as the team's Tier-1 "
           "cohort: control-cell means are reference-survey-calibrated baselines; treatment "
           "cells add the Tier-3 crowd-forecast ATEs; demographic-moderator cells carry "
           "real group-level baseline differences (reference-calibrated) plus "
           "headroom-proportional effect moderation (groups far from ceiling move more); "
           "behavioral outcomes get level differences but flat effects.",
    "0.3": "Tier 2, primary. Family: measurement-calibrated statistical synthesis; LLM "
           "ensemble for effects; reference-survey-conditioned.",
    "0.4": "(1) same generator as Tier-1 (see that form) -> (2) main cells: full-cohort "
           "baseline means + ATE vector -> (3) moderator cells: group baseline means from "
           "the full 18,000-profile cohort render + ATE x group mean headroom factor "
           "(flat for donation/signup) -> (4) verify ranges, completeness, no NAs, no "
           "exact zeros.",
    "0.5": "Full coverage: 221 main cells (17 conditions x 13 outcomes) + 5,967 moderator "
           "cells (x 27 demographic levels), complete, no NA.",
    "D.1": T1_TEXT["D.1"] + " (Moderator cells use the full cohort as the group baseline "
           "population for smoothness.)",
    "G.2": "All cells verified in range with exact level strings; moderation profile "
           "example: near-ceiling groups receive proportionally smaller effects "
           "(headroom mechanism), never exact zeros.",
}

ENTRIES = [
    dict(dir="t3_primary", tier=3, entry="primary",
         family="direct effect forecast; multi-model multi-prompt ensemble; literature-calibrated",
         models=MODELS_T3A, text=T3A_TEXT,
         abstract="Calibrated silicon crowd: five-model, multi-prompt direct forecasts of all 208 "
                  "treatment effects, pooled by trimmed mean and calibrated with literature-derived "
                  "outcome priors; hyperparameters fixed by retrodiction on two public corpora.",
         files={"benchmark_T3A_v0.csv": "predictions/team_5_T3_primary_v1.csv"},
         logs=["t3a_elicitations.csv"], log_glob="*7c2ec9*.jsonl"),
    dict(dir="t3_secondary1", tier=3, entry="secondary-1",
         family="direct forecast via log-probability readout; two-model ensemble; zero-shot",
         models=MODELS_T3B, text=T3B_TEXT,
         abstract="Readout crowd: uncalibrated token-logprob readout of predicted average responses "
                  "per condition and outcome from open-weight models; the deliberate no-priors "
                  "contrast to the team's primary Tier-3 entry.",
         files={"__T3B__": "predictions/team_5_T3_secondary-1_v1.csv"},
         logs=["t3b_readouts.csv"], log_glob="*b79bfb*.jsonl"),
    dict(dir="t1_primary", tier=1, entry="primary",
         family="measurement-calibrated statistical synthesis; LLM ensemble for effects only",
         models=MODELS_T3A, text=T1_TEXT,
         abstract="Calibrated cohort: 18,000 synthetic respondents with census demographics, "
                  "reference-survey-calibrated response distributions (heaping, partisan structure, "
                  "item correlations), and the team's crowd-forecast effect vector injected with "
                  "headroom-based heterogeneity. No per-respondent LLM simulation.",
         files={"T1_cohort_v0.csv": "predictions/team_5_T1_primary_v1.csv"},
         logs=["t3a_elicitations.csv"], log_glob="*7c2ec9*.jsonl"),
    dict(dir="t2_primary", tier=2, entry="primary",
         family="measurement-calibrated statistical synthesis; LLM ensemble for effects only",
         models=MODELS_T3A, text=T2_TEXT,
         abstract="Cell-level means derived from the team's calibrated-cohort generator: "
                  "reference-calibrated baselines, crowd-forecast effects, real demographic level "
                  "differences with headroom-proportional moderation.",
         files={"T2_cells_main_v0.csv": "predictions/team_5_T2_primary_v1_cells_main.csv",
                "T2_cells_moderator_v0.csv": "predictions/team_5_T2_primary_v1_cells_moderator.csv"},
         logs=["t3a_elicitations.csv"], log_glob="*7c2ec9*.jsonl"),
]


def build_t3b_csv(dest: Path) -> None:
    from silicon.readout import t3b_ates
    df = t3b_ates(RUNS / "t3b_readouts.csv")
    df = df.sort_values(["condition", "outcome"])
    df["ate"] = df["ate"].round(4)
    df.to_csv(dest, index=False)


def assemble(e: dict) -> str:
    dst = SUBS / e["dir"]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(TEMPLATE, dst)
    # clear example predictions + example raw deposit
    for f in (dst / "predictions").glob("example_*"):
        f.unlink()
    raw = dst / "raw_data_deposit"
    for f in raw.glob("*"):
        f.unlink() if f.is_file() else shutil.rmtree(f)

    for src, rel in e["files"].items():
        if src == "__T3B__":
            build_t3b_csv(dst / rel)
        else:
            shutil.copy(RUNS / src, dst / rel)

    # raw logs
    for name in e["logs"]:
        shutil.copy(RUNS / name, raw / name)
    for f in RUNS.glob(e["log_glob"]):
        shutil.copy(f, raw / f.name)
    (raw / "README.md").write_text(
        "Raw model outputs: JSONL files contain every API call (full prompt, response, "
        "usage, timing) for the runs feeding this entry; the CSV is the parsed/tidied "
        "version. Generator code and seeds: see code_repository in metadata.json.\n")

    meta = json.loads((dst / "metadata.json").read_text())
    meta.update({
        "team_id": TEAM, "team_name": "Team 5 (Cummins)", "contact": CONTACT,
        "creators": CREATORS, "abstract": e["abstract"], "tier": e["tier"],
        "entry": e["entry"], "approach_family": e["family"], "models": e["models"],
        "code_repository": CODE_REPO, "disclosure_class": "A",
        "prediction_files": [{"file": rel, "sha256": ""} for rel in e["files"].values()],
        "coverage": {"interventions": 16, "outcomes": 13},
        "blinding_attestation": True,
    })
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (dst / "registration.md").write_text(reg_form(e["text"]))

    r = subprocess.run(["make", "manifest"], cwd=dst, capture_output=True, text=True)
    c = subprocess.run(["make", "check"], cwd=dst, capture_output=True, text=True)
    z = subprocess.run(["make", "zenodo_citation"], cwd=dst, capture_output=True, text=True)
    tail = [l for l in c.stdout.splitlines() if "OVERALL" in l or "fail" in l.lower()]
    return f"{e['dir']}: {tail[-1] if tail else c.stdout[-200:]} (manifest rc={r.returncode}, zenodo rc={z.returncode})"


if __name__ == "__main__":
    SUBS.mkdir(exist_ok=True)
    for e in ENTRIES:
        print(assemble(e))
