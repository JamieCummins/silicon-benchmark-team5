"""Generative elicitation runner over a StudySpec: (model x variant x condition) -> tidy ATEs.

Checkpointed: reruns skip (model, variant, intervention) cells already present in
the output CSV, so interrupted runs resume for free. Raw calls land in runs/*.jsonl.
Used both for the benchmark target (T3-A) and for retrodiction studies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import RUNS_DIR
from .prompts_t3 import Variant, all_variants, build_messages
from .providers import CallLogger, call
from .studyspec import StudySpec


def parse_predictions(
    text: str, keys: list[str], sanity: dict[str, tuple[float, float]]
) -> tuple[dict[str, float] | None, str, str]:
    """Returns (predictions, rationale, error). Predictions None on failure."""
    raw = text.strip()
    # thinking models (qwen3.6 on Groq, etc.) may inline <think>...</think> before the JSON
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    # Some hosts inject a stray glitch token between a key's colon and its number
    # (observed with deepseek-v3.2: '"trust_post": une 1.4'). Drop junk that sits
    # directly before a number; junk *replacing* the number still fails -> resampled.
    raw = re.sub(r'(:\s*)[^\d\s"\-+.{}\[\],:]+\s+(?=-?\d)', r"\1", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None, "", "no JSON object found"
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        return None, "", f"json decode: {e}"
    preds = obj.get("predictions", obj)  # tolerate flat top-level dicts
    missing = [k for k in keys if k not in preds]
    if missing:
        return None, "", f"missing keys: {missing[:3]}..."
    out: dict[str, float] = {}
    for k in keys:
        try:
            v = float(preds[k])
        except (TypeError, ValueError):
            return None, "", f"non-numeric value for {k}: {preds[k]!r}"
        lo, hi = sanity[k]
        if not (lo <= v <= hi):
            return None, "", f"insane ATE for {k}: {v} (bounds {lo}..{hi})"
        out[k] = v
    return out, str(obj.get("rationale", "")), ""


def _elicit_cell(
    spec: StudySpec,
    provider: str,
    model: str,
    short: str,
    variant: Variant,
    name: str,
    logger: CallLogger,
    max_tokens: int,
) -> tuple[list[dict] | None, str, str]:
    """One (model, variant, condition) cell with up to 2 attempts: glitch tokens
    mid-JSON (e.g. deepseek emitting stray unicode) are stochastic, so one
    resample usually recovers."""
    tag = f"{spec.study_id}:{short}:{variant.vid}:{name}"
    preds, rationale, err = None, "", ""
    for attempt in (1, 2):
        try:
            r = call(
                provider, model, build_messages(spec, name, variant),
                logger, tag=tag if attempt == 1 else f"{tag}:r{attempt}",
                max_tokens=max_tokens,
            )
            preds, rationale, err = parse_predictions(r["text"], spec.outcome_keys, spec.sanity)
        except Exception as e:  # keep the run going; cell can be retried later
            preds, rationale, err = None, "", f"{type(e).__name__}: {e}"
        if preds is not None:
            break
    if preds is None:
        return None, tag, err
    rows = [
        {
            "model": short, "provider": provider, "variant": variant.vid,
            "intervention": name, "outcome": k, "ate": v,
            "rationale": rationale, "log": logger.run_id,
        }
        for k, v in preds.items()
    ]
    return rows, tag, ""


def run_elicitation(
    spec: StudySpec,
    roster: list[tuple[str, str, str]],
    variants: list[Variant],
    conditions: list[str],
    out_csv: Path,
    logger: CallLogger | None = None,
    max_tokens: int = 8192,
    workers: int = 1,
    variants_by_model: dict[str, list[Variant]] | None = None,
) -> pd.DataFrame:
    """Elicit all pending cells. `variants_by_model` overrides `variants` for
    specific models (e.g. a reduced grid for an expensive reasoning model).
    Only the main thread writes the CSV, so no write lock is needed."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger = logger or CallLogger()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, str]] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["model"], prev["variant"], prev["intervention"]))

    tasks = []
    for provider, model, short in roster:
        for variant in (variants_by_model or {}).get(short, variants):
            for name in conditions:
                if (short, variant.vid, name) not in done:
                    tasks.append((provider, model, short, variant, name))

    n_run = n_fail = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(_elicit_cell, spec, p, m, s, v, n, logger, max_tokens)
            for p, m, s, v, n in tasks
        ]
        for fut in as_completed(futures):
            rows, tag, err = fut.result()
            n_run += 1
            if rows is None:
                n_fail += 1
                print(f"  FAIL {tag}: {err[:100]}", flush=True)
                continue
            pd.DataFrame(rows).to_csv(
                out_csv, mode="a", header=not out_csv.exists(), index=False
            )
            if n_run % 25 == 0 or n_run == len(tasks):
                print(f"  {n_run}/{len(tasks)} cells done ({n_fail} failed)", flush=True)
    print(f"\ndone: {n_run} calls this session ({n_fail} failed), "
          f"log runs/{logger.run_id}.jsonl")
    return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()


def default_out_csv() -> Path:
    return RUNS_DIR / "t3a_elicitations.csv"


def pilot_spec() -> tuple[list, list[Variant], list[str]]:
    """2 cheap models x 3 variants x 2 interventions (one simple, one state-adaptive)."""
    from .config import GENERATIVE_ROSTER

    roster = [r for r in GENERATIVE_ROSTER if r[2] in ("luna", "deepseek")]
    variants = all_variants()[:3]
    return roster, variants, ["Consensus", "Extreme weather predictions"]


def full_spec() -> tuple[list, list[Variant], list[str]]:
    from .config import GENERATIVE_ROSTER
    from .materials import scored_labels

    return GENERATIVE_ROSTER, all_variants(), scored_labels()
