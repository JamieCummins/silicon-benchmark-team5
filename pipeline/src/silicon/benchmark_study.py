"""The Silicon Sample Benchmark target study as a StudySpec."""

from __future__ import annotations

from .materials import (
    WEATHER_CASE_NAMES,
    WEATHER_CASE_WEIGHTS,
    Materials,
    load_materials,
)
from .outcomes import OUTCOMES
from .studyspec import StudySpec

DESCRIPTION = (
    "A preregistered online field experiment tested short text interventions to increase "
    "trust in climate scientists. Sample: ~18,000 U.S. adults from an opt-in online panel, "
    "census-matched on gender x age and gender x race/ethnicity, ~1,000 per intervention arm "
    "and ~2,000 in control. Between-subjects design: each participant was randomly assigned "
    "to read ONE intervention (or one neutral control text), then immediately answered all "
    "outcome measures. Outcomes are empty-by-default integer sliders unless noted."
)

CONTROL_INTRO = (
    "participants read one of three neutral filler texts (randomly chosen), "
    "reproduced verbatim below."
)

ANCHOR_BLOCK = """Reference points for calibration:
- In comparable megastudies, average effects of single text messages on 0-100 attitude
  sliders are small: |ATE| typically 0-3 points, most often under 1 point.
- Effects of a single message exposure on real behavior (donations, sign-ups) are
  typically near zero; predicting them close to zero is usually correct.
- Proximal outcomes (about trust in climate scientists itself) move more than distal
  ones (policy support, personal behavior intentions).
- Most Democrats are near the scale ceiling on trust outcomes; headroom is concentrated
  among Republicans and Independents, which caps how far sample-average effects can move.
- A few messages backfire on some outcomes; the average message does not."""


def _control_block(m: Materials) -> str:
    parts = []
    for i, (title, text) in enumerate(m.control_fillers.items(), 1):
        parts.append(f'--- control filler {i}/3: "{title}" ---\n{text}')
    return "\n\n".join(parts)


def _intervention_entry(name: str, m: Materials) -> tuple[str, str]:
    if name == "Extreme weather predictions":
        note = (
            "this arm is STATE-ADAPTIVE. Each participant saw an intro naming their own state "
            "and risk type, then exactly ONE of four case texts matched to their state. "
            "Approximate share of participants per case: "
            + "; ".join(
                f"Case {n} ({WEATHER_CASE_NAMES[n]}): ~{int(WEATHER_CASE_WEIGHTS[n] * 100)}%"
                for n in sorted(m.weather_cases)
            )
            + ". Predict the ATE for the arm as a whole (the case mixture)."
        )
        intro = f"Intro shown (with the participant's state and risk label filled in):\n{m.weather_intro}"
        cases = "\n\n".join(
            f"--- Case {n} (~{int(WEATHER_CASE_WEIGHTS[n] * 100)}% of participants) ---\n{m.weather_cases[n]}"
            for n in sorted(m.weather_cases)
        )
        return note, f"{intro}\n\n{cases}"
    text = m.interventions[name]
    if "page break" in text:
        note = (
            "participants experienced this as a multi-page sequence in the order shown "
            "(page breaks marked); embedded questions are part of the intervention itself."
        )
    else:
        note = "participants read the following text."
    return note, text


def benchmark_spec(m: Materials | None = None) -> StudySpec:
    m = m or load_materials()
    return StudySpec(
        study_id="benchmark",
        description=DESCRIPTION,
        control_intro=CONTROL_INTRO,
        control_block=_control_block(m),
        conditions={name: _intervention_entry(name, m) for name in m.interventions},
        outcomes=list(OUTCOMES),
        anchor_block=ANCHOR_BLOCK,
    )
