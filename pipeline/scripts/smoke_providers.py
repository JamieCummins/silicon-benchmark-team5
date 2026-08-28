"""Provider smoke test: one tiny call per provider + logprobs probes.

Total spend: well under one cent per key. Run: uv run python scripts/smoke_providers.py
"""

import math

from silicon.providers import CallLogger, call

log = CallLogger()
print(f"log -> runs/{log.path.name}\n")

# 1. Groq basic (cheapest model on the roster)
r = call(
    "groq", "openai/gpt-oss-120b",
    [{"role": "user", "content": "Reply with exactly: HARNESS OK"}],
    log, tag="smoke-groq", max_tokens=512,
)
print(f"[groq/gpt-oss-120b]        text={r['text'].strip()[:40]!r}")

# 2. Groq logprobs probe (expected unsupported -> error or None)
try:
    r = call(
        "groq", "llama-3.3-70b-versatile",
        [{"role": "user", "content": "Reply with a single digit: 7"}],
        log, tag="smoke-groq-logprobs", max_tokens=16, logprobs=True,
    )
    status = "RETURNED" if r["logprobs"] else "accepted param but returned None"
    print(f"[groq logprobs]            {status}")
except Exception as e:
    print(f"[groq logprobs]            unsupported ({type(e).__name__}: {str(e)[:80]})")

# 3. OpenRouter basic
r = call(
    "openrouter", "openai/gpt-oss-120b",
    [{"role": "user", "content": "Reply with exactly: HARNESS OK"}],
    log, tag="smoke-or", max_tokens=512,
)
print(f"[openrouter/gpt-oss-120b]  text={r['text'].strip()[:40]!r}")

# 4. OpenRouter READOUT mechanics: single-digit answer + top-20 logprobs on a
#    non-reasoning model. This is the T3-B/T1 load-bearing test.
r = call(
    "openrouter", "google/gemma-3-27b-it",
    [{"role": "user", "content": (
        "On a scale from 0 (no trust) to 9 (complete trust), how much does the average "
        "US adult trust climate scientists? Answer with a single digit and nothing else."
    )}],
    log, tag="smoke-or-readout", max_tokens=8, logprobs=True, top_logprobs=20,
    require_logprobs_provider=True,
)
print(f"[openrouter readout]       text={r['text'].strip()[:10]!r}")
if r["logprobs"]:
    first = r["logprobs"][0]
    digit_mass = {}
    for alt in first["top"]:
        tok = alt["token"].strip()
        if tok.isdigit() and len(tok) == 1:
            digit_mass[tok] = digit_mass.get(tok, 0.0) + math.exp(alt["logprob"])
    total = sum(digit_mass.values())
    exp_val = sum(int(d) * p for d, p in digit_mass.items()) / total if total else float("nan")
    dist = {d: round(p, 3) for d, p in sorted(digit_mass.items())}
    print(f"[openrouter readout]       digit mass={round(total, 3)}  E[digit]={round(exp_val, 2)}")
    print(f"[openrouter readout]       distribution={dist}")
else:
    print("[openrouter readout]       NO LOGPROBS RETURNED — readout blocked, investigate")
