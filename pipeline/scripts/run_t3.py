"""T3-A elicitation CLI.

  uv run python scripts/run_t3.py --dry-run          # render 2 prompts, no API calls
  uv run python scripts/run_t3.py --pilot            # 2 models x 3 variants x 2 interventions (~1-2 cents)
  uv run python scripts/run_t3.py --full             # full crowd  (SPENDS REAL MONEY, ~$30-45)
"""

import argparse

from silicon.benchmark_study import benchmark_spec
from silicon.elicit_t3 import default_out_csv, full_spec, pilot_spec, run_elicitation
from silicon.prompts_t3 import all_variants, build_messages

parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--dry-run", action="store_true")
mode.add_argument("--pilot", action="store_true")
mode.add_argument("--full", action="store_true")
args = parser.parse_args()

spec = benchmark_spec()

if args.dry_run:
    for name in ("Consensus", "Extreme weather predictions"):
        msgs = build_messages(spec, name, all_variants()[0])
        n_chars = sum(len(x["content"]) for x in msgs)
        print(f"=== {name} | variant {all_variants()[0].vid} | ~{n_chars} chars ===")
        print(msgs[0]["content"][:400])
        print("...\n[USER]")
        print(msgs[1]["content"][:1200])
        print(f"\n[... truncated, total {n_chars} chars ≈ {n_chars // 4} tokens]\n")
elif args.pilot:
    roster, variants, interventions = pilot_spec()
    print(f"PILOT: {[r[2] for r in roster]} x {[v.vid for v in variants]} x {interventions}")
    run_elicitation(spec, roster, variants, interventions, default_out_csv())
elif args.full:
    roster, variants, interventions = full_spec()
    V = all_variants()
    terra_variants = [V[i] for i in (0, 3, 4, 7, 8, 11)]  # balanced half-grid
    n_cheap = sum(1 for r in roster if r[2] != "terra") * len(V) * len(interventions)
    n_terra = len(terra_variants) * len(interventions)
    print(f"FULL BENCHMARK RUN: {n_cheap} cheap-model + {n_terra} terra calls "
          f"= {n_cheap + n_terra} total, expected ~$3. Ctrl-C now to abort...")
    import time

    time.sleep(5)
    run_elicitation(spec, roster, variants, interventions, default_out_csv(),
                    workers=8, variants_by_model={"terra": terra_variants})
