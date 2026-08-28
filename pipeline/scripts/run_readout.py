"""T3-B readout CLI.

  uv run python scripts/run_readout.py --dry-run   # render 2 prompts + unit inventory, no calls
  uv run python scripts/run_readout.py --pilot     # 1 model x 2 framings x 2 units x 4 outcomes (~1 cent)
  uv run python scripts/run_readout.py --full      # verified readout models x 4 framings x 22 units x 13 outcomes (~$2-4)
"""

import argparse

from silicon.config import READOUT_ROSTER
from silicon.outcomes import OUTCOMES
from silicon.readout import (
    build_readout_messages,
    default_readout_csv,
    run_readout,
    text_units,
)

parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--dry-run", action="store_true")
mode.add_argument("--pilot", action="store_true")
mode.add_argument("--full", action="store_true")
args = parser.parse_args()

units = text_units()
VERIFIED = [r for r in READOUT_ROSTER if r[2] in ("deepseek", "gptoss")]

if args.dry_run:
    print(f"{len(units)} text units:")
    for u in units:
        print(f"  {u.uid:45s} {len(u.text):6d} chars")
    o = OUTCOMES[0]
    msgs = build_readout_messages("sim", units[3].text, o)
    print(f"\n=== sample prompt: {units[3].uid} x {o.key} ===")
    print("[SYSTEM]", msgs[0]["content"])
    print("[USER]", msgs[1]["content"][:900], "...")
elif args.pilot:
    pilot_units = [u for u in units if u.uid in ("Consensus", "control:The History of Neckties")]
    pilot_outcomes = [o for o in OUTCOMES if o.key in
                      ("trust_multidimensional", "distrust_post", "donation_ams", "newsletter_signup")]
    roster = VERIFIED[:1]
    print(f"PILOT: {[r[2] for r in roster]} x ['sim','pollster'] x "
          f"{[u.uid for u in pilot_units]} x {[o.key for o in pilot_outcomes]}")
    run_readout(roster, ["sim", "pollster"], pilot_units, pilot_outcomes, default_readout_csv())
elif args.full:
    n = len(VERIFIED) * 4 * len(units) * len(OUTCOMES)
    print(f"FULL READOUT: {n} calls (~$2-4). Ctrl-C now to abort...")
    import time

    time.sleep(5)
    run_readout(VERIFIED, list(("sim", "pollster", "analyst", "plain")), units, OUTCOMES,
                default_readout_csv())
