"""Retrodiction run 2 CLI (Hewitt archive slice). See reference/retro2_preanalysis.md.

  uv run python scripts/run_retro2.py --dry-run   # slice summary + cost estimate, no calls
  uv run python scripts/run_retro2.py --pilot     # 1 model x 2 variants x 2 studies (~0.5 cents)
  uv run python scripts/run_retro2.py --run       # the ~$5 run (prints estimate, 5s abort window)
"""

import argparse

from silicon.config import GENERATIVE_ROSTER, RUNS_DIR
from silicon.elicit_t3 import run_elicitation
from silicon.prompts_t3 import all_variants, build_messages
from silicon.providers import CallLogger
from silicon.retro_hewitt import study_specs

OUT_CSV = RUNS_DIR / "retro_hewitt_elicitations.csv"

PRICES = {"luna": (0.20, 1.20), "maverick": (0.20, 0.80), "deepseek": (0.27, 0.40),
          "gptoss": (0.15, 0.60), "terra": (2.00, 12.00)}
OUT_TOK = {"luna": 700, "maverick": 500, "deepseek": 500, "gptoss": 900, "terra": 1200}

ROSTER = [r for r in GENERATIVE_ROSTER if r[2] in ("luna", "maverick", "deepseek", "gptoss", "terra")]
V = all_variants()
VARIANTS6 = [V[i] for i in (0, 3, 4, 7, 8, 11)]
VARIANTS3 = [V[i] for i in (0, 7, 11)]
BY_MODEL = {"terra": VARIANTS3}


def estimate() -> tuple[int, float, dict[str, int]]:
    specs = study_specs()
    in_tokens_per_cond = {}
    for sid, spec in specs.items():
        for name in spec.conditions:
            msgs = build_messages(spec, name, V[0])
            in_tokens_per_cond[name] = sum(len(m["content"]) for m in msgs) // 4
    n_calls = 0
    cost = 0.0
    for _, _, short in ROSTER:
        k = len(BY_MODEL.get(short, VARIANTS6))
        pi, po = PRICES[short]
        for name, tin in in_tokens_per_cond.items():
            n_calls += k
            cost += k * (tin / 1e6 * pi + OUT_TOK[short] / 1e6 * po)
    return n_calls, cost, in_tokens_per_cond


parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--dry-run", action="store_true")
mode.add_argument("--pilot", action="store_true")
mode.add_argument("--run", action="store_true")
args = parser.parse_args()

specs = study_specs()
n_calls, cost, per_cond = estimate()
print(f"slice: {len(specs)} studies, {len(per_cond)} elicitable conditions")
print(f"planned: {n_calls} calls, estimated ${cost:.2f} "
      f"(roster {[r[2] for r in ROSTER]}, 6 variants / terra 3)")

if args.dry_run:
    sid, spec = next(iter(specs.items()))
    name = next(iter(spec.conditions))
    msgs = build_messages(spec, name, V[0])
    print(f"\n=== sample prompt ({name}) ===")
    print(msgs[1]["content"][:1600])
elif args.pilot:
    logger = CallLogger()
    roster = [r for r in ROSTER if r[2] == "luna"]
    for sid in list(specs)[:2]:
        spec = specs[sid]
        run_elicitation(spec, roster, V[:2], list(spec.conditions), OUT_CSV,
                        logger=logger, workers=4)
elif args.run:
    if cost > 6.0:
        raise SystemExit(f"estimated ${cost:.2f} exceeds the ~$5 authorization — trim the slice first")
    print("firing in 5s (Ctrl-C to abort)...")
    import time

    time.sleep(5)
    logger = CallLogger()
    for i, (sid, spec) in enumerate(specs.items(), 1):
        print(f"[study {i}/{len(specs)}: {sid}]", flush=True)
        run_elicitation(spec, ROSTER, VARIANTS6, list(spec.conditions), OUT_CSV,
                        logger=logger, workers=8, variants_by_model=BY_MODEL)
