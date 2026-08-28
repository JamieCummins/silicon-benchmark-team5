"""Retrodiction elicitation CLI (ground 1: Vlasceanu US).

  uv run python scripts/run_retro.py --dry-run   # spec summary + sample prompt, no calls
  uv run python scripts/run_retro.py --pilot     # 1 model x 2 variants x 2 conditions (~0.5 cents)
  uv run python scripts/run_retro.py --full      # 7 models x 12 variants x 11 conditions (~$15-25)

After elicitation, tune with silicon.tune.grid_search([vlasceanu_ground(csv)]).
"""

import argparse

from silicon.config import GENERATIVE_ROSTER, RUNS_DIR
from silicon.elicit_t3 import run_elicitation
from silicon.prompts_t3 import all_variants, build_messages
from silicon.retro_vlasceanu import vlasceanu_spec

OUT_CSV = RUNS_DIR / "retro_vlasceanu_elicitations.csv"

parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--dry-run", action="store_true")
mode.add_argument("--pilot", action="store_true")
mode.add_argument("--small", action="store_true",
                  help="~$3-4: 6 cheap models on the full grid + terra on 6 balanced variants")
mode.add_argument("--full", action="store_true")
args = parser.parse_args()

spec = vlasceanu_spec()

if args.dry_run:
    for name in list(spec.conditions)[:1]:
        msgs = build_messages(spec, name, all_variants()[0])
        n = sum(len(x["content"]) for x in msgs)
        print(f"=== {name} | ~{n} chars ≈ {n // 4} tokens ===")
        print(msgs[1]["content"][:1500])
elif args.pilot:
    roster = [r for r in GENERATIVE_ROSTER if r[2] == "luna"]
    variants = all_variants()[:2]
    conditions = ["Scientific Consensus", "Decreasing Psychological Distance"]
    print(f"PILOT: {[r[2] for r in roster]} x {[v.vid for v in variants]} x {conditions}")
    run_elicitation(spec, roster, variants, conditions, OUT_CSV)
elif args.small:
    variants = all_variants()
    # terra gets a balanced half-grid: each frame twice, pre/post x3, both orders x3
    terra_variants = [variants[i] for i in (0, 3, 4, 7, 8, 11)]
    conditions = list(spec.conditions)
    n_cheap = sum(1 for r in GENERATIVE_ROSTER if r[2] != "terra") * len(variants) * len(conditions)
    n_terra = len(terra_variants) * len(conditions)
    print(f"SMALL RETRO RUN: {n_cheap} cheap-model calls + {n_terra} terra calls "
          f"= {n_cheap + n_terra} total, expected ~$3-4.")
    run_elicitation(
        spec, GENERATIVE_ROSTER, variants, conditions, OUT_CSV,
        workers=8, variants_by_model={"terra": terra_variants},
    )
elif args.full:
    n = len(GENERATIVE_ROSTER) * len(all_variants()) * len(spec.conditions)
    print(f"FULL RETRO RUN: {n} calls (~$5-8). Ctrl-C now to abort...")
    import time

    time.sleep(5)
    run_elicitation(spec, GENERATIVE_ROSTER, all_variants(), list(spec.conditions), OUT_CSV,
                    workers=8)
