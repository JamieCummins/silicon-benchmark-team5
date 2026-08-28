"""T3-B: logprob readout of predicted condition means (the Silicon Crowds method).

One call per (model x framing x text-unit x outcome): ask for the predicted
AVERAGE response as a single digit 0-9, then read the probability-weighted
expectation over digit tokens from the first answer token's top-logprobs —
no sampling, no personas, no priors.

Scale mappings (single-digit by design; multi-digit answers break readout):
- 0-100 sliders: digit = tens place of the average -> value = 10*E[digit] + 5
- donation (0-10 dollars): digit = average in whole dollars (9 = $9+) -> E[digit]
- newsletter (0/1): digit = subscribers per 10 participants -> p = E[digit]/10

Text units: 16 interventions (extreme weather expanded to its 4 state cases,
each with its intro resolved) + the 3 control fillers. ATEs are assembled as
condition expectation minus the mean of the three filler expectations, with
weather cases recombined by population weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import RUNS_DIR
from .materials import WEATHER_CASE_WEIGHTS, Materials, load_materials
from .outcomes import OUTCOMES, Outcome
from .providers import CallLogger, call

MIN_DIGIT_MASS = 0.5  # a usable answer token must carry at least this much digit mass


# --- scale adaptations ---------------------------------------------------------

def readout_scale(o: Outcome) -> tuple[str, Callable[[float], float]]:
    """(digit instruction, E[digit] -> native value)."""
    if o.key == "donation_ams":
        return (
            "Report the predicted AVERAGE donation in whole dollars as a single digit 0-9 "
            "(answer 9 if the average would be $9 or more).",
            lambda e: e,
        )
    if o.key == "newsletter_signup":
        return (
            "Report how many out of every 10 participants you predict will subscribe, "
            "as a single digit 0-9.",
            lambda e: e / 10.0,
        )
    return (
        "Report the TENS DIGIT of the predicted average on the 0-100 scale as a single "
        "digit 0-9 (e.g. if the average would be about 62, answer 6; about 8, answer 0).",
        lambda e: 10.0 * e + 5.0,
    )


# --- prompt framings -----------------------------------------------------------

FRAMINGS = {
    "sim": (
        "You are simulating the results of a large preregistered online experiment with "
        "~18,000 U.S. adults, census-matched on gender x age and gender x race/ethnicity. "
        "Predict average responses accurately, as they would actually come out."
    ),
    "pollster": (
        "You are a veteran U.S. survey researcher with deep knowledge of public-opinion "
        "distributions on science and climate. Estimate average survey responses precisely."
    ),
    "analyst": (
        "You are a data analyst filling in the results table of a randomized experiment "
        "run on a census-matched U.S. adult sample. Give your single best numeric estimate."
    ),
    "plain": (
        "Answer with your single best numeric estimate. No explanation."
    ),
}

USER_TEMPLATE = """In the study, a participant was randomly assigned to read the following material, then immediately answered survey questions.

--- material read by the participant ---
{text}
--- end material ---

Question answered immediately afterwards:
[{key}] {description}

Predict the AVERAGE result across all U.S. adult participants who read this material. {digit_instruction}
Answer with a single digit (0-9) and nothing else."""


def build_readout_messages(framing: str, text: str, o: Outcome) -> list[dict[str, str]]:
    instr, _ = readout_scale(o)
    return [
        {"role": "system", "content": FRAMINGS[framing]},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                text=text, key=o.key, description=o.description, digit_instruction=instr
            ),
        },
    ]


# --- text units ----------------------------------------------------------------

@dataclass(frozen=True)
class TextUnit:
    uid: str  # e.g. "Consensus", "control:The Rules of Baseball", "Extreme weather predictions:case2"
    text: str


def text_units(m: Materials | None = None) -> list[TextUnit]:
    m = m or load_materials()
    units: list[TextUnit] = []
    for title, text in m.control_fillers.items():
        units.append(TextUnit(f"control:{title}", text))
    for name, text in m.interventions.items():
        if name == "Extreme weather predictions":
            for n in sorted(m.weather_cases):
                if n == 4:
                    intro = m.weather_intro_generic
                else:
                    phrase = m.weather_case_phrases.get(n, "states facing extreme weather")
                    intro = m.weather_intro.replace("[STATE]", "their home state").replace(
                        "[CASE]", phrase
                    )
                units.append(TextUnit(f"{name}:case{n}", f"{intro}\n\n{m.weather_cases[n]}"))
        else:
            units.append(TextUnit(name, text))
    return units


# --- expectation extraction ----------------------------------------------------

def digit_expectation(logprobs: list[dict] | None) -> tuple[float, float] | None:
    """Scan for the first token whose top-list carries enough single-digit mass;
    return (E[digit] renormalized over digits, raw digit mass). None if absent."""
    for tok in logprobs or []:
        mass = 0.0
        num = 0.0
        for alt in tok.get("top", []):
            t = alt["token"].strip()
            if t.isdigit() and len(t) == 1:
                p = math.exp(alt["logprob"])
                mass += p
                num += int(t) * p
        if mass >= MIN_DIGIT_MASS:
            return num / mass, mass
    return None


# --- runner --------------------------------------------------------------------

def default_readout_csv() -> Path:
    return RUNS_DIR / "t3b_readouts.csv"


def run_readout(
    roster: list[tuple[str, str, str]],
    framings: list[str],
    units: list[TextUnit],
    outcomes: list[Outcome],
    out_csv: Path,
    logger: CallLogger | None = None,
) -> pd.DataFrame:
    logger = logger or CallLogger()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, str, str]] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["model"], prev["framing"], prev["unit"], prev["outcome"]))

    n_run = n_fail = 0
    for provider, model, short in roster:
        # reasoning models (gpt-oss) need output room before the answer token
        mt = 2048 if "oss" in short else 8
        for framing in framings:
            for unit in units:
                for o in outcomes:
                    if (short, framing, unit.uid, o.key) in done:
                        continue
                    tag = f"t3b:{short}:{framing}:{unit.uid}:{o.key}"
                    result = None
                    err = ""
                    for attempt in (1, 2):
                        try:
                            r = call(
                                provider, model,
                                build_readout_messages(framing, unit.text, o),
                                logger, tag=tag if attempt == 1 else f"{tag}:r{attempt}",
                                max_tokens=mt, logprobs=True, top_logprobs=20,
                                require_logprobs_provider=True,
                            )
                            result = digit_expectation(r["logprobs"])
                            err = "" if result else f"no digit mass (text={r['text'][:20]!r})"
                        except Exception as e:
                            result, err = None, f"{type(e).__name__}: {e}"
                        if result:
                            break
                    n_run += 1
                    if result is None:
                        n_fail += 1
                        print(f"  FAIL {tag}: {err[:90]}")
                        continue
                    e_digit, mass = result
                    _, to_native = readout_scale(o)
                    pd.DataFrame(
                        [{
                            "model": short, "provider": provider, "framing": framing,
                            "unit": unit.uid, "outcome": o.key,
                            "e_digit": round(e_digit, 4), "digit_mass": round(mass, 4),
                            "native": round(to_native(e_digit), 4), "log": logger.run_id,
                        }]
                    ).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    print(f"\nreadout done: {n_run} calls ({n_fail} failed), log runs/{logger.run_id}.jsonl")
    return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()


# --- assembly: readouts -> ATEs -------------------------------------------------

def t3b_ates(readout_csv: Path) -> pd.DataFrame:
    """Average expectations over models x framings, recombine weather cases,
    subtract the pooled-control expectation. Returns intervention x outcome ATEs."""
    df = pd.read_csv(readout_csv)
    cell = df.groupby(["unit", "outcome"], as_index=False)["native"].mean()

    def unit_kind(uid: str) -> str:
        return "control" if uid.startswith("control:") else "treat"

    control = (
        cell[cell["unit"].str.startswith("control:")]
        .groupby("outcome")["native"].mean()
    )

    rows = []
    treat = cell[~cell["unit"].str.startswith("control:")].copy()
    treat["condition"] = treat["unit"].str.split(":case").str[0]
    for (cond, outcome), grp in treat.groupby(["condition", "outcome"]):
        if grp["unit"].str.contains(":case").any():
            w = grp["unit"].str.split(":case").str[1].astype(int).map(WEATHER_CASE_WEIGHTS)
            value = (grp["native"] * w).sum() / w.sum()
        else:
            value = grp["native"].mean()
        rows.append({"condition": cond, "outcome": outcome,
                     "ate": value - control[outcome]})
    return pd.DataFrame(rows)
