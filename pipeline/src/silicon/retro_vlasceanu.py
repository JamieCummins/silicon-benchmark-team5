"""Retrodiction ground 1: Vlasceanu et al. (2024) climate intervention tournament,
U.S. arm. 11 message interventions x 4 outcomes with known ATEs (data/raw/
vlasceanu2024_climate/us_ates.csv, computed from participant-level Zenodo data).

The same elicitation machinery used for the benchmark runs here; observed ATEs
then ground the aggregation hyperparameters (tune.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .outcomes import Outcome
from .studyspec import StudySpec
from .tune import Ground

ROOT = DATA_DIR / "raw" / "vlasceanu2024_climate"
STIM = ROOT / "stimuli_us_extracted"

# condName in data -> (display name used in prompts, stimulus file)
CONDITIONS = {
    "WorkTogetherNorm": ("Working Together Norm", "09_2._Identity-Social-Norms-Intervention.txt"),
    "NegativeEmotions": ("Negative Emotions", "17_3._Negative-Emotion-Intervention.txt"),
    "SciConsens": ("Scientific Consensus", "19_4._Scientific_Consensus_Intervention.txt"),
    "CollectAction": ("Collective Action", "20_5._Collective_Action_Intervention_New.txt"),
    "SystemJust": ("System Justification", "21_6._System_Justification_Intervention.txt"),
    "PsychDistance": ("Decreasing Psychological Distance", "23_7._Decreasing_Psychological_Distance_Intervention.txt"),
    "PluralIgnorance": ("Correcting Pluralistic Ignorance", "22_8._Correcting_Pluralistic_Ignorance_Intervention.txt"),
    "LetterFutureGen": ("Letter to Future Generations", "29_9._A_Letter_to_Future_GenerationsV2.txt"),
    "DynamicNorm": ("Dynamic Social Norms", "24_10._Dynamic_Social_Norms.txt"),
    "FutureSelfCont": ("Future Self-Continuity", "25_11._Future_Self-Continuity_Intervention.txt"),
    "BindingMoral": ("Binding Moral Foundations", "26_12._A_Binding_Moral_Foundations_Intervention_v1Globe.txt"),
}
CONTROL_FILE = "30_1._Control_Distracter.txt"

VLASCEANU_OUTCOMES = [
    Outcome(
        "belief", "primary", "points on the 0-100 slider scale",
        "Belief in climate change: mean of 4 accuracy ratings (0-100 sliders) of the "
        "statements 'Human activities are causing climate change', 'Climate change poses a "
        "serious threat to humanity', 'Taking action to fight climate change is necessary to "
        "avoid a global catastrophe', 'Climate change is a global emergency'.",
        (-20, 20),
    ),
    Outcome(
        "policy", "primary", "points on the 0-100 slider scale",
        "Climate policy support: mean of 9 agreement sliders (0-100), 'I support...' raising "
        "carbon taxes on fossil fuels; expanding public transportation; more EV charging "
        "stations; more wind/solar energy; airline carbon taxes; protecting forests/land; "
        "green jobs investment; clean-waterways laws; taxes on carbon-intense foods.",
        (-20, 20),
    ),
    Outcome(
        "share", "behavioral", "probability (proportion willing to share, 0-1 scale)",
        "Willingness to share climate information on social media (yes/no): participants saw "
        "a short factual climate post (about cutting food-related emissions by eating less "
        "meat/dairy) and were asked 'Are you willing to share this information on your social "
        "media?'. Predict the ATE as a change in the proportion answering yes (the control-"
        "group proportion is roughly 0.5).",
        (-0.4, 0.4),
    ),
    Outcome(
        "wept", "behavioral", "pages completed on the 0-8 task scale",
        "Work for Environmental Protection Task (real effortful behavior): a voluntary, "
        "tedious number-identification task of up to 8 pages; each completed page triggers a "
        "real tree-planting donation. Participants can stop at any time. Outcome = number of "
        "pages completed (0-8; the control-group mean is roughly 5).",
        (-2, 2),
    ),
]

ANCHOR_BLOCK = """Reference points for calibration:
- In comparable megastudies, average effects of single text messages on 0-100 attitude
  sliders are small: |ATE| typically 0-3 points, most often under 1 point.
- Effects of a single message exposure on real effortful behavior are typically near zero;
  low-cost in-survey actions (like agreeing to share a post) can move somewhat more.
- Proximal outcomes (beliefs directly targeted by the message) move more than distal
  ones (policy support, behavior).
- U.S. climate attitudes are strongly polarized; many liberals are near the scale ceiling,
  so headroom is concentrated among conservatives and moderates.
- A few messages backfire on some outcomes; the average message does not."""

DESCRIPTION = (
    "A preregistered online experiment (part of a 63-country 'intervention tournament') "
    "tested short interventions to stimulate climate action. This is the U.S. arm: 8,253 "
    "U.S. adults recruited online, ~670 per intervention arm and ~670 in control. "
    "Between-subjects design: each participant was randomly assigned to ONE intervention "
    "(or a neutral control text), then immediately completed the outcome measures. "
    "Belief and policy outcomes are 0-100 sliders."
)

CONTROL_INTRO = (
    "participants read a neutral literary excerpt (from Dickens' Great Expectations) "
    "of similar length, reproduced below."
)

_HEADER = re.compile(r"^BLOCK:.*\n^SOURCE:.*\n", re.M)
_TIMER = re.compile(
    r"^### \S*[Tt]imer\S*\nTiming\n\[Response options: First Click.*?\]\n?", re.M
)


def _clean(path: Path) -> str:
    text = path.read_text()
    text = _HEADER.sub("", text)
    text = _TIMER.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def vlasceanu_spec() -> StudySpec:
    conditions = {}
    for cond, (display, fname) in CONDITIONS.items():
        text = _clean(STIM / fname)
        note = (
            "participants saw the following intervention material (question prompts shown "
            "inline are part of the intervention experience)."
        )
        conditions[display] = (note, text)
    return StudySpec(
        study_id="vlasceanu_us",
        description=DESCRIPTION,
        control_intro=CONTROL_INTRO,
        control_block=_clean(STIM / CONTROL_FILE),
        conditions=conditions,
        outcomes=VLASCEANU_OUTCOMES,
        anchor_block=ANCHOR_BLOCK,
    )


# Retro-specific aggregation inputs (module-level so tune.py can vary them)
PRIORS = {"belief": 1.5, "policy": 1.0, "share": 0.02, "wept": 0.0}
EPSILON = {"belief": 0.02, "policy": 0.02, "share": 0.002, "wept": 0.005}
RANGES = {"belief": 100.0, "policy": 100.0, "share": 1.0, "wept": 8.0}


def observed_ates() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "us_ates.csv")
    df = df[df["sample"] == "US"].copy()
    display = {k: v[0] for k, v in CONDITIONS.items()}
    df["condition"] = df["intervention"].map(display)
    if df["condition"].isna().any():
        missing = df.loc[df["condition"].isna(), "intervention"].unique()
        raise ValueError(f"unmapped interventions in us_ates.csv: {missing}")
    return df.rename(columns={"ate_vs_control": "ate", "se_welch": "se"})[
        ["condition", "outcome", "ate", "se"]
    ]


def vlasceanu_ground(elicitations_csv: Path) -> Ground:
    return Ground(
        study_id="vlasceanu_us",
        elicitations_csv=elicitations_csv,
        observed=observed_ates(),
        ranges=RANGES,
        priors=PRIORS,
        epsilon=EPSILON,
    )


if __name__ == "__main__":
    spec = vlasceanu_spec()
    obs = observed_ates()
    print(f"{len(spec.conditions)} conditions, {len(spec.outcomes)} outcomes, "
          f"{len(obs)} observed ATE cells")
    for name, (_, text) in spec.conditions.items():
        print(f"  {name:38s} {len(text):6d} chars")
    print("\nobserved ATE snapshot (pp of range):")
    obs["pp"] = obs.apply(lambda r: r["ate"] * 100 / RANGES[r["outcome"]], axis=1)
    print(obs.pivot_table(index="condition", columns="outcome", values="pp").round(2).to_string())
