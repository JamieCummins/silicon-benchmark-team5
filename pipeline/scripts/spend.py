"""Actual spend accounting from raw call logs.

Prices per MTok (input, output), updated Aug 17 2026. qwen3.6-27b on Groq is
approximate (not on the fetched price list).

  uv run python scripts/spend.py            # all logs
  uv run python scripts/spend.py 20260817   # logs whose run_id starts with prefix
"""

import json
import pathlib
import sys
from collections import defaultdict

from silicon.config import RUNS_DIR

PRICES = {  # (provider, model substring) -> ($/MTok in, $/MTok out)
    ("openai", "gpt-5.6-terra"): (2.00, 12.00),
    ("openai", "gpt-5.6-luna"): (0.20, 1.20),
    ("openai", "gpt-5-mini"): (0.25, 2.00),
    ("groq", "gpt-oss-120b"): (0.15, 0.60),
    ("groq", "qwen3.6-27b"): (0.30, 0.60),
    ("openrouter", "deepseek"): (0.27, 0.40),
    ("openrouter", "kimi"): (0.57, 2.85),
    ("openrouter", "maverick"): (0.20, 0.80),
    ("openrouter", "gemma"): (0.08, 0.45),
    ("openrouter", "gpt-oss-120b"): (0.03, 0.17),
    ("openrouter", "qwen3.5"): (0.29, 2.40),
}


def price_for(provider: str, model: str) -> tuple[float, float]:
    for (p, sub), pr in PRICES.items():
        if p == provider and sub in model:
            return pr
    return (0.5, 2.0)  # unknown-model fallback, flagged in output


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    tokens = defaultdict(lambda: [0, 0, 0])  # (provider, model) -> [in, out, calls]
    for f in sorted(RUNS_DIR.glob(f"{prefix}*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = (rec.get("response") or {}).get("raw_usage") or {}
            t_in = u.get("prompt_tokens") or 0
            t_out = u.get("completion_tokens") or 0
            k = (rec.get("provider", "?"), rec.get("model", "?"))
            tokens[k][0] += t_in
            tokens[k][1] += t_out
            tokens[k][2] += 1

    total = 0.0
    by_provider = defaultdict(float)
    print(f"{'provider':11s} {'model':28s} {'calls':>6s} {'in_tok':>10s} {'out_tok':>9s} {'cost':>8s}")
    for (prov, model), (t_in, t_out, n) in sorted(tokens.items()):
        pi, po = price_for(prov, model)
        cost = t_in / 1e6 * pi + t_out / 1e6 * po
        total += cost
        by_provider[prov] += cost
        print(f"{prov:11s} {model:28s} {n:6d} {t_in:10d} {t_out:9d} ${cost:7.3f}")
    print("-" * 76)
    for prov, c in sorted(by_provider.items()):
        print(f"{prov:11s} {'':28s} {'':6s} {'':10s} {'':9s} ${c:7.3f}")
    print(f"{'TOTAL':11s} {'':28s} {'':6s} {'':10s} {'':9s} ${total:7.3f}")


if __name__ == "__main__":
    main()
