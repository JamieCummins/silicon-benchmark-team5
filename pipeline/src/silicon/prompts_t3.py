"""Elicitation prompt construction: 12 variants = 3 frames x 2 anchor placements
x 2 outcome orders, over any StudySpec.

Prompt bias cancels through variety (Cummins & Wulff): the frames and orderings
are deliberately different ways of asking the same question, aggregated later.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .studyspec import StudySpec

FRAMES = {
    "forecaster": (
        "You are an elite forecaster with a strong track record on prediction platforms, "
        "specializing in social-science experiments. You reason from base rates and reference "
        "classes first, then adjust for specifics. You know that message interventions in "
        "megastudies typically produce small attitudinal effects and near-zero behavioral ones, "
        "and that forecasters' most common error is predicting effects that are too large."
    ),
    "methodologist": (
        "You are a quantitative social scientist who has run and reviewed dozens of "
        "preregistered field experiments on science communication. You think in terms of "
        "statistical power, measurement reactivity, ceiling effects, and the gap between "
        "attitudes and behavior. You are skeptical of large claimed effects."
    ),
    "meta-analyst": (
        "You are a meta-analyst who knows the climate-communication and persuasion literatures "
        "in detail: consensus messaging, inoculation, trust-building, and megastudies comparing "
        "many messages head-to-head. You predict new results by placing them in the "
        "distribution of published and replicated effect sizes, correcting for publication bias."
    ),
}

STUDY_TEMPLATE = """{description}

CONTROL condition: {control_intro}

{control_block}

INTERVENTION — "{name}": {presentation_note}

{intervention_block}

OUTCOME MEASURES (all answered immediately post-treatment):

{outcome_block}

{anchor_after}Task: for EACH outcome, predict the average treatment effect (ATE) of THIS intervention versus the pooled control group — intervention mean minus control mean — in the outcome's native units as specified above. Consider the mechanism of this specific message, its likely reception across the U.S. political spectrum, measurement reactivity, and how it compares to typical messages in such megastudies. Small and null effects are common; sign matters — give your best point estimate, not a hedge toward zero if you genuinely expect an effect.

Respond with ONLY a JSON object, no markdown fences, in exactly this form:
{json_spec}"""


@dataclass(frozen=True)
class Variant:
    frame: str
    anchored_first: bool  # anchors before materials (True) or after (False)
    outcome_order_seed: int | None  # None = canonical order

    @property
    def vid(self) -> str:
        pos = "pre" if self.anchored_first else "post"
        order = "can" if self.outcome_order_seed is None else f"s{self.outcome_order_seed}"
        return f"{self.frame}-{pos}-{order}"


def all_variants() -> list[Variant]:
    return [
        Variant(frame, anchored_first, seed)
        for frame in FRAMES
        for anchored_first in (True, False)
        for seed in (None, 7)
    ]


def build_messages(spec: StudySpec, name: str, variant: Variant) -> list[dict[str, str]]:
    outcomes = list(spec.outcomes)
    if variant.outcome_order_seed is not None:
        rng = random.Random(variant.outcome_order_seed)
        rng.shuffle(outcomes)

    outcome_block = "\n\n".join(
        f"[{o.key}] (predict ATE in {o.unit})\n{o.description}" for o in outcomes
    )
    json_spec = json.dumps(
        {"predictions": {o.key: "<float>" for o in outcomes}, "rationale": "<2-4 sentences>"}
    ).replace('"<float>"', "<float>")

    note, block = spec.conditions[name]
    system = FRAMES[variant.frame]
    anchor_after = ""
    if spec.anchor_block:
        if variant.anchored_first:
            system = system + "\n\n" + spec.anchor_block
        else:
            anchor_after = spec.anchor_block + "\n\n"

    user = STUDY_TEMPLATE.format(
        description=spec.description,
        control_intro=spec.control_intro,
        control_block=spec.control_block,
        name=name,
        presentation_note=note,
        intervention_block=block,
        outcome_block=outcome_block,
        anchor_after=anchor_after,
        json_spec=json_spec,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
