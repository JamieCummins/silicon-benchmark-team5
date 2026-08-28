"""Parse the benchmark's survey materials into structured intervention/control texts.

Source of truth: template/survey/questionnaire.txt (verbatim texts) and
template/survey/condition_codenames.csv (the 17 scored condition labels).
`load_materials()` validates that exactly the expected 16 interventions + 3
control fillers are recovered, so silent parser drift fails loudly.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field

from .config import SURVEY_DIR

QUESTIONNAIRE = SURVEY_DIR / "questionnaire.txt"
CODENAMES = SURVEY_DIR / "condition_codenames.csv"

# Approximate share of respondents seeing each extreme-weather case, from state
# populations (flood/wildfire/winter lists in questionnaire.txt) + a small
# prefer-not-to-say remainder. Refine against CPS in the T1 phase.
WEATHER_CASE_WEIGHTS = {1: 0.56, 2: 0.24, 3: 0.19, 4: 0.01}
WEATHER_CASE_NAMES = {
    1: "high or recurrent flood risk (e.g. TX, FL, PA, OH, GA...)",
    2: "high or increasing wildfire risk (e.g. CA, AZ, WA, CO...)",
    3: "severe cold, snow, ice, or blizzards (e.g. NY, MI, NJ, MA...)",
    4: "no state reported (generic extreme-weather version)",
}


@dataclass
class Materials:
    interventions: dict[str, str]  # scored label -> verbatim section text
    control_fillers: dict[str, str]  # filler title -> verbatim text
    weather_cases: dict[int, str] = field(default_factory=dict)  # case n -> text
    weather_intro: str = ""  # the page-2 intro (ELSE branch, with placeholders)
    weather_intro_generic: str = ""  # the IF branch (no state reported)
    weather_case_phrases: dict[int, str] = field(default_factory=dict)  # n -> quoted risk label


def scored_labels() -> list[str]:
    """The 16 intervention labels (excluding control), from condition_codenames.csv."""
    with CODENAMES.open() as f:
        rows = list(csv.DictReader(f))
    return sorted({r["title"] for r in rows if r["title"] != "control"})


def _strip_hidden(text: str) -> str:
    """Cut authoring-only tails: References / Risk categories blocks."""
    m = re.search(r"^(References|Risk categories) \[not displayed", text, flags=re.M)
    return text[: m.start()].rstrip() if m else text.rstrip()


def load_materials() -> Materials:
    raw = QUESTIONNAIRE.read_text()

    # Split the whole file on '### ' headings; keep sections we recognize.
    parts = re.split(r"^### (.+)$", raw, flags=re.M)
    # parts = [preamble, title1, body1, title2, body2, ...]
    sections: dict[str, str] = {}
    for title, body in zip(parts[1::2], parts[2::2]):
        # A section ends where the next structural separator begins.
        body = re.split(r"^-{20,}\s*$|^={20,}\s*$", body, flags=re.M)[0]
        sections[title.strip()] = _strip_hidden(body.strip())

    labels = scored_labels()
    interventions = {lab: sections[lab] for lab in labels if lab in sections}
    fillers = {
        t.split(":", 1)[1].strip(): b
        for t, b in sections.items()
        if t.startswith("control — filler text")
    }

    missing = set(labels) - set(interventions)
    if missing:
        raise ValueError(f"parser failed to find intervention sections: {missing}")
    if len(fillers) != 3:
        raise ValueError(f"expected 3 control fillers, found {len(fillers)}: {list(fillers)}")

    # Extreme weather: pull the four case texts and the page-2 intro.
    ew = interventions["Extreme weather predictions"]
    cases: dict[int, str] = {}
    case_iter = list(re.finditer(r"^Case (\d)\s*$", ew, flags=re.M))
    for i, m in enumerate(case_iter):
        n = int(m.group(1))
        end = case_iter[i + 1].start() if i + 1 < len(case_iter) else len(ew)
        body = ew[m.end() : end].strip()
        if len(body) > 400:  # skip the short assignment-logic 'Case n – "..."' lines
            cases[n] = body
    if set(cases) != {1, 2, 3, 4}:
        raise ValueError(f"expected weather cases 1-4, found {sorted(cases)}")

    intro_m = re.search(r"^ELSE:\s*\n(.+?)(?=\n\s*\n)", ew, flags=re.M | re.S)
    weather_intro = intro_m.group(1).strip() if intro_m else ""
    if_m = re.search(r"^IF state=.*?:\s*\n(.+?)(?=\n\s*\n)", ew, flags=re.M | re.S)
    weather_intro_generic = if_m.group(1).strip() if if_m else ""
    phrases = {
        int(n): p.strip()
        for n, p in re.findall(r"^Case (\d)\s*[–-]\s*[“”\"](.+?)[“”\"]", ew, flags=re.M)
    }

    return Materials(
        interventions=interventions,
        control_fillers=fillers,
        weather_cases=cases,
        weather_intro=weather_intro,
        weather_intro_generic=weather_intro_generic,
        weather_case_phrases=phrases,
    )


if __name__ == "__main__":
    m = load_materials()
    print(f"{len(m.interventions)} interventions, {len(m.control_fillers)} fillers, "
          f"{len(m.weather_cases)} weather cases")
    for lab in sorted(m.interventions):
        print(f"  {lab:35s} {len(m.interventions[lab]):6d} chars")
    for t in m.control_fillers:
        print(f"  [control] {t:26s} {len(m.control_fillers[t]):6d} chars")
