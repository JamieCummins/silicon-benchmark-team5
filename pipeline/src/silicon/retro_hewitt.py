"""Retrodiction ground 2: Hewitt/Ashokkumar archive slice (multi-study).

Consumes pipeline/data/derived/hewitt_slice/{contrasts.csv, stimuli.csv}
produced by the extraction step. Unlike the Vlasceanu ground (one study, 11
arms), this is many small studies: each study becomes its own StudySpec; the
pooled Ground uses study-scoped condition and outcome labels so scales,
centering, and priors never collide across studies.

Prospective priors: sign from the archive's hypothesized_direction metadata
(mechanical, set before observed values are looked at), magnitude 0.8pp of
scale range for attitude outcomes (see reference/retro2_preanalysis.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .outcomes import Outcome
from .studyspec import StudySpec
from .tune import Ground

SLICE = DATA_DIR / "derived" / "hewitt_slice"

PRIOR_PP = 0.8      # prior |mean effect| in pp of range, attitude outcomes
EPSILON_PP = 0.02   # smallest submitted |ATE| in pp of range

ANCHOR_BLOCK = """Reference points for calibration:
- In survey experiments like this, average effects of reading a short text on
  attitude items are small: typically 0-5 points on a 0-100 equivalent scale,
  most often under 2 points.
- Persuasive messages usually move attitudes in their intended direction, but
  some treatments backfire on some outcomes; consider whether this one could.
- Effects are usually larger on items closely targeted by the text than on
  broader or more identity-laden attitudes.
- U.S. samples are politically heterogeneous; strong partisan priors on an
  issue cap how far the average can move."""


def scoped(study_id: str, name: str) -> str:
    return f"{name} [study {study_id}]"


def load_slice() -> tuple[pd.DataFrame, pd.DataFrame]:
    contrasts = pd.read_csv(SLICE / "contrasts.csv", dtype={"study_id": str})
    stimuli = pd.read_csv(SLICE / "stimuli.csv", dtype={"study_id": str})
    return contrasts, stimuli


def _outcome_for(row: pd.Series) -> Outcome:
    rng = float(row["scale_max"]) - float(row["scale_min"])
    labels = row.get("scale_labels")
    labels_txt = f" Scale anchors: {labels}." if isinstance(labels, str) and labels.strip() else ""
    return Outcome(
        key=str(row["outcome_key"]),
        tier="archive",
        unit=f"points on the {row['scale_min']:g}-{row['scale_max']:g} response scale",
        description=f"{row['outcome_text']}{labels_txt}",
        sanity=(-0.3 * rng, 0.3 * rng),
    )


def study_specs() -> dict[str, StudySpec]:
    """One StudySpec per study; conditions are study-scoped for global uniqueness."""
    contrasts, stimuli = load_slice()
    stim_map = {(r["study_id"], r["condition"]): r for _, r in stimuli.iterrows()}
    specs: dict[str, StudySpec] = {}
    for sid, grp in contrasts.groupby("study_id"):
        ref_name = grp["reference"].iloc[0]
        ref_row = stim_map.get((sid, ref_name))
        ref_text = ""
        if ref_row is not None and isinstance(ref_row["stimulus_text"], str) and ref_row["stimulus_text"].strip():
            control_intro = "participants read the following material, reproduced verbatim below."
            ref_text = ref_row["stimulus_text"]
        else:
            control_intro = ("participants in the reference condition saw no additional material "
                             "and answered the outcome questions directly.")
        context = ""
        any_row = ref_row if ref_row is not None else next(iter(
            stim_map.get((sid, c)) for c in grp["condition"].unique()
            if stim_map.get((sid, c)) is not None), None)
        if any_row is not None and isinstance(any_row.get("study_context"), str) and any_row["study_context"].strip():
            context = f"\n\nShared material shown to ALL participants first:\n{any_row['study_context']}"

        outcomes: dict[str, Outcome] = {}
        for _, r in grp.iterrows():
            outcomes.setdefault(str(r["outcome_key"]), _outcome_for(r))

        n_ref = int(grp["n_ref"].iloc[0])
        conditions: dict[str, tuple[str, str]] = {}
        for cond in grp["condition"].unique():
            srow = stim_map.get((sid, cond))
            text = srow["stimulus_text"] if srow is not None else ""
            if not isinstance(text, str) or not text.strip():
                continue  # cannot elicit without a stimulus; logged by caller
            n_t = int(grp[grp["condition"] == cond]["n_treat"].iloc[0])
            conditions[scoped(sid, cond)] = (
                "participants read the following material.", text
            )
        label = grp["study_label"].iloc[0]
        specs[sid] = StudySpec(
            study_id=f"hewitt_{sid}",
            description=(
                f"A preregistered U.S. survey experiment ('{label}') on a nationally "
                f"representative or general-population U.S. adult sample. Participants were "
                f"randomly assigned between conditions (~{max(n_ref, 50)} per condition), then "
                f"immediately answered the outcome questions.{context}"
            ),
            control_intro=control_intro,
            control_block=ref_text,
            conditions=conditions,
            outcomes=list(outcomes.values()),
            anchor_block=ANCHOR_BLOCK,
        )
    return specs


def _scoped_frames() -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict[str, float]]:
    contrasts, _ = load_slice()
    obs = contrasts.copy()
    obs["condition"] = [scoped(s, c) for s, c in zip(obs["study_id"], obs["condition"])]
    obs["outcome_scoped"] = [f"{o}@{s}" for o, s in zip(obs["outcome_key"], obs["study_id"])]
    ranges = {}
    priors = {}
    epsilon = {}
    for _, r in obs.iterrows():
        key = r["outcome_scoped"]
        rng = float(r["scale_max"]) - float(r["scale_min"])
        ranges[key] = rng
        d = r.get("hypothesized_direction")
        sign = 1.0 if pd.isna(d) or d == 0 else float(d) / abs(float(d))
        priors[key] = sign * PRIOR_PP / 100.0 * rng
        epsilon[key] = EPSILON_PP / 100.0 * rng
    obs = obs.rename(columns={"outcome_scoped": "outcome"})[
        ["condition", "outcome", "ate", "se", "study_id"]
    ]
    return obs, ranges, priors, epsilon


def scope_elicitations(el: pd.DataFrame) -> pd.DataFrame:
    """Elicitation rows carry study-scoped conditions but unscoped outcome keys
    (models answer each study's own JSON). Scope the outcomes for scoring."""
    sid = el["intervention"].str.extract(r"\[study (.+)\]$")[0]
    out = el.copy()
    out["outcome"] = out["outcome"] + "@" + sid
    return out


def hewitt_ground(elicitations_csv: Path) -> Ground:
    obs, ranges, priors, epsilon = _scoped_frames()
    return Ground(
        study_id="hewitt_slice",
        elicitations_csv=elicitations_csv,
        observed=obs,
        ranges=ranges,
        priors=priors,
        epsilon=epsilon,
    )


if __name__ == "__main__":
    specs = study_specs()
    obs, ranges, priors, epsilon = _scoped_frames()
    n_cond = sum(len(s.conditions) for s in specs.values())
    n_out = sum(len(s.outcomes) for s in specs.values())
    print(f"{len(specs)} studies, {n_cond} elicitable conditions, {n_out} study-outcomes, "
          f"{len(obs)} observed contrasts")
    neg = (obs['ate'] / obs['outcome'].map(ranges) * 100).lt(0).mean()
    print(f"observed negative-effect share: {neg*100:.0f}%")
    import numpy as np
    print(f"prior sign distribution: +{sum(1 for v in priors.values() if v > 0)}, "
          f"-{sum(1 for v in priors.values() if v < 0)}")