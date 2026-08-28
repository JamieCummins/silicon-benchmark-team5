"""Central configuration: model roster, paths, run settings.

Every model is addressed as (provider, model_id). All three providers are
OpenAI-compatible (OpenAI, Groq, OpenRouter).
Keys come from .env: OPENAI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY.
"""

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = PIPELINE_DIR.parent
TEMPLATE_DIR = REPO_DIR / "template"
RUNS_DIR = PIPELINE_DIR / "runs"  # raw call logs (required for Tier 1-2 deposits)
DATA_DIR = PIPELINE_DIR / "data"  # reference survey data

SURVEY_DIR = TEMPLATE_DIR / "survey"
CODEBOOK = TEMPLATE_DIR / "codebook.csv"

# --- T3-A generative crowd roster ---------------------------------------------
# (provider, model_id, short_name). 5 model families for crowd diversity.
# Groq catalog verified Aug 17 (llama-3.3 and kimi are gone from Groq).
# Retro-validated crowd (both retrodiction grounds ran on this five).
# Dropped after run 1: qwen36 (think-tag failures + Groq 429s; 5 usable cells),
# kimi (56% of run cost in reasoning tokens for mid contribution, LOMO ~0).
GENERATIVE_ROSTER = [
    ("openai", "gpt-5.6-terra", "terra"),  # reasoning arm: ~5.5-level at $2/$12
    ("openai", "gpt-5.6-luna", "luna"),    # volume arm: $0.20/$1.20
    ("groq", "openai/gpt-oss-120b", "gptoss"),
    ("openrouter", "deepseek/deepseek-v3.2", "deepseek"),
    ("openrouter", "meta-llama/llama-4-maverick", "maverick"),
]

# --- T3-B / T1 readout roster (logprobs required) ------------------------------
# Verified Aug 17 via scripts/probe_readout.py: full top-20 first-token
# distribution comes back host-dependently. Groq = no logprobs (404s the param).
READOUT_ROSTER = [
    ("openrouter", "deepseek/deepseek-v3.2", "deepseek"),   # VERIFIED (DigitalOcean, mass .999)
    ("openrouter", "openai/gpt-oss-120b", "gptoss"),        # VERIFIED (DigitalOcean, mass 1.0; peaked post-reasoning)
    ("openrouter", "google/gemma-3-27b-it", "gemma3"),      # needs provider pin (Parasail drops top-logprobs)
    ("openrouter", "meta-llama/llama-4-maverick", "maverick"),  # to verify
    ("openrouter", "qwen/qwen3.5-122b-a10b", "qwen35"),     # to verify (default host returned no choices)
]

# Benchmark constants (from template scripts/check + prereg)
CONDITIONS = 17  # 16 interventions + control
OUTCOMES = 13
N_PROMPT_VARIANTS_PILOT = 3
N_PROMPT_VARIANTS_FULL = 12  # revisit after Spearman-Brown on pilot inter-prompt correlations
