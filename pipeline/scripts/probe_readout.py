"""Probe which OpenRouter models/hosts return usable first-token top-logprobs.

A usable readout host returns, for the FIRST content token, a top-20 list whose
digit tokens carry nearly all probability mass. Cost: ~5 tiny calls, <0.1 cent.
Run: uv run python scripts/probe_readout.py
"""

import math
import os

from dotenv import load_dotenv

from silicon.providers import CallLogger, call

load_dotenv()

QUESTION = (
    "On a scale from 0 (no trust) to 9 (complete trust), how much does the average "
    "US adult trust climate scientists? Answer with a single digit and nothing else."
)

CANDIDATES = [
    ("google/gemma-3-27b-it", {}),
    ("deepseek/deepseek-v3.2", {}),
    ("qwen/qwen3.5-122b-a10b", {}),
    ("openai/gpt-oss-120b", {}),
]

log = CallLogger()
print(f"log -> runs/{log.path.name}\n")

for model, extra in CANDIDATES:
    try:
        r = call(
            "openrouter", model,
            [{"role": "user", "content": QUESTION}],
            log, tag=f"probe-{model}", max_tokens=2048,  # room for reasoning models
            logprobs=True, top_logprobs=20, require_logprobs_provider=True, **extra,
        )
    except Exception as e:
        print(f"{model:32s} ERROR {type(e).__name__}: {str(e)[:70]}")
        continue

    lp = r["logprobs"] or []
    host = r.get("provider_served")
    text = r["text"].strip()[:12]
    if not lp:
        print(f"{model:32s} host={host!s:14s} text={text!r}  NO LOGPROBS")
        continue

    # find the first token whose top-list carries digit mass
    verdict = "no digit token found"
    for i, tok in enumerate(lp):
        mass = 0.0
        est = 0.0
        for alt in tok["top"]:
            t = alt["token"].strip()
            if t.isdigit() and len(t) == 1:
                p = math.exp(alt["logprob"])
                mass += p
                est += int(t) * p
        if mass > 0.5:
            verdict = f"token[{i}]={tok['token']!r} digit_mass={mass:.3f} E[digit]={est / mass:.2f} n_top={len(tok['top'])}"
            break
    print(f"{model:32s} host={host!s:14s} text={text!r}  {verdict}  (n_tokens={len(lp)})")

print("\n--- OpenAI gpt-5* catalog (free) ---")
from openai import OpenAI

oa = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
for m in sorted(m.id for m in oa.models.list() if m.id.startswith("gpt-5")):
    print("  ", m)
