"""The 13 scored outcomes, transcribed verbatim from template/survey/questionnaire.txt
and validated against template/codebook.csv target labels.

`unit` is the native unit the ATE must be predicted in (and submitted in for Tier 3).
`sanity` bounds are for parse validation of elicited ATEs, not for clipping.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from .config import CODEBOOK


@dataclass(frozen=True)
class Outcome:
    key: str  # submission column / Tier-3 `outcome` value
    tier: str
    unit: str
    description: str  # shown to models: what the measure is, verbatim item wording
    sanity: tuple[float, float]  # plausible ATE bounds for parse validation


OUTCOMES: list[Outcome] = [
    Outcome(
        "trust_multidimensional", "primary", "points on the 0-100 slider scale",
        "PRIMARY OUTCOME. Trust in climate scientists, 12-item composite (mean of four "
        "3-item subscales; every item a 0-100 slider about 'most climate scientists'): "
        "Competence (incompetent-competent, unintelligent-intelligent, unqualified-qualified), "
        "Integrity (dishonest-honest, unethical-ethical, insincere-sincere), "
        "Benevolence (unconcerned-concerned about people's wellbeing, uneager-eager to improve "
        "others' lives, inconsiderate-considerate of others' interests), "
        "Openness (openness to feedback, willingness to be transparent, attention to other "
        "people's views).",
        (-15, 15),
    ),
    Outcome(
        "trust_post", "secondary", "points on the 0-100 slider scale",
        "Single item: 'How much do you trust climate scientists?' 0 = not at all, "
        "100 = very strongly.",
        (-15, 15),
    ),
    Outcome(
        "distrust_post", "secondary", "points on the 0-100 slider scale",
        "Single item: 'How much do you DISTRUST climate scientists?' 0 = not at all, "
        "100 = very strongly. NOTE THE POLARITY: an intervention that increases trust "
        "should usually DECREASE this outcome (negative ATE).",
        (-15, 15),
    ),
    Outcome(
        "funding_perceptions", "secondary", "points on the 0-100 scale",
        "Support for federal climate-research funding, recoded so HIGHER = supports MORE "
        "funding. Raw item: 'Is the federal government spending too much, too little or "
        "about the right amount on climate change research?' (0 = far too little, 50 = about "
        "right, 100 = far too much), then reversed as 100 minus raw.",
        (-15, 15),
    ),
    Outcome(
        "policy_role_mean", "secondary", "points on the 0-100 slider scale",
        "Scientists' role in policy making, mean of 4 agree-disagree sliders (0 = strongly "
        "disagree, 100 = strongly agree): climate scientists should work closely with policy "
        "makers; should actively advocate for specific policies; should communicate findings "
        "to policy makers; should be more involved in the policy-making process.",
        (-15, 15),
    ),
    Outcome(
        "inst_trust_mean", "secondary", "points on the 0-100 slider scale",
        "Trust in institutions, mean of 5 sliders (0 = not at all, 100 = very strongly): "
        "EPA, NASA, NOAA, universities and colleges, the federal government.",
        (-15, 15),
    ),
    Outcome(
        "belief_post", "tertiary", "points on the 0-100 slider scale",
        "Accuracy rating of the statement 'Human activities are causing climate change' "
        "(0 = not at all accurate, 100 = extremely accurate).",
        (-15, 15),
    ),
    Outcome(
        "concern_mean", "tertiary", "points on the 0-100 slider scale",
        "Climate concern, mean of 3 sliders: how concerned about climate change; how serious "
        "a problem; how important relative to other issues facing the U.S.",
        (-15, 15),
    ),
    Outcome(
        "policy_general", "tertiary", "points on the 0-100 slider scale",
        "Support for 'The U.S. government should do more to reduce global warming' "
        "(0 = strongly oppose, 100 = strongly support).",
        (-15, 15),
    ),
    Outcome(
        "policy_specific_mean", "tertiary", "points on the 0-100 slider scale",
        "Support for specific climate policies, mean of 7 sliders (0 = strongly oppose, 100 = "
        "strongly support): fossil-fuel taxes; public-transport infrastructure; sustainable "
        "energy expansion; protecting forests/land; carbon-intensive food taxes; green jobs "
        "investment; clean-waterways laws.",
        (-15, 15),
    ),
    Outcome(
        "behavior_mean", "tertiary", "points on the 0-100 slider scale",
        "Self-reported likelihood of 6 mitigation behaviors in the next 12 months (0 = not "
        "likely at all, 100 = extremely likely): eat less meat; use car alternatives; install "
        "solar; fly less; talk to friends/family about climate; donate to an environmental NGO.",
        (-15, 15),
    ),
    Outcome(
        "donation_ams", "behavioral", "US dollars (the 0-10 dollar donation scale)",
        "REAL behavior: of a $10 bonus, how much the participant donates to the American "
        "Meteorological Society ($0-$10 in $1 increments). Predict the ATE in dollars "
        "(e.g. +0.08 means 8 cents more donated on average).",
        (-3, 3),
    ),
    Outcome(
        "newsletter_signup", "behavioral", "probability (proportion signing up, 0-1 scale)",
        "REAL behavior: whether the participant subscribed to climate scientist Katharine "
        "Hayhoe's free 'Talking Climate' newsletter when shown an optional offer page with a "
        "signup link mid-survey (1 = subscribed, 0 = not). Predict the ATE as a change in "
        "signup probability (e.g. +0.005 means half a percentage point more subscribers).",
        (-0.3, 0.3),
    ),
]

OUTCOME_KEYS = [o.key for o in OUTCOMES]


def validate_against_codebook() -> None:
    """Every outcome key must appear as a target_label in the official codebook."""
    with CODEBOOK.open() as f:
        targets = {row["target_label"] for row in csv.DictReader(f)}
    missing = [k for k in OUTCOME_KEYS if k not in targets]
    if missing:
        raise ValueError(f"outcome keys not found in codebook: {missing}")


if __name__ == "__main__":
    validate_against_codebook()
    print(f"{len(OUTCOMES)} outcomes validated against codebook")
