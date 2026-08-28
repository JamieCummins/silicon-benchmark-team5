"""Generic description of an elicitation target: a study whose per-condition
treatment effects we want a model crowd to forecast.

The benchmark megastudy is one StudySpec (see benchmark_study.py); each
retrodiction study (Vlasceanu et al., Ashokkumar archive entries, ...) is
another. Everything downstream (prompt building, elicitation, aggregation)
consumes StudySpec so the exact same machinery is tuned on public studies and
applied to the sealed target.
"""

from __future__ import annotations

from dataclasses import dataclass

from .outcomes import Outcome


@dataclass(frozen=True)
class StudySpec:
    study_id: str
    # One paragraph: design, sample, assignment, measurement timing.
    description: str
    # Sentence introducing the control condition, e.g. "participants read one of
    # three neutral filler texts (randomly chosen), reproduced verbatim below."
    control_intro: str
    # Verbatim control materials (may be multiple fillers concatenated with headers).
    control_block: str
    # condition label -> (presentation_note, verbatim block)
    conditions: dict[str, tuple[str, str]]
    outcomes: list[Outcome]
    # Study-appropriate reference-class calibration anchors ("" to disable).
    anchor_block: str

    @property
    def outcome_keys(self) -> list[str]:
        return [o.key for o in self.outcomes]

    @property
    def sanity(self) -> dict[str, tuple[float, float]]:
        return {o.key: o.sanity for o in self.outcomes}
