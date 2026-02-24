# Hutsul Dialect LLM Baseline Evaluation

Compares LLMs on translating Hutsul dialect to standard literary Ukrainian using promptfoo.
Establishes a baseline before comparing with fine-tuned models.

## Quick Start

```bash
# 1. Generate test cases from eval.csv (first 50 rows)
make eval-dialect-prepare

# 2. Run evaluation (requires OPENROUTER_API_KEY in environment)
make eval-dialect

# 3. View results in browser
make eval-dialect-view
```

## Models Evaluated

| Model | Provider ID | Notes |
|-------|-------------|-------|
| GPT-4o Mini | `openrouter:openai/gpt-4o-mini` | General-purpose, fast |
| Gemini 2.0 Flash | `openrouter:google/gemini-2.0-flash-001` | General-purpose, also used as judge |
| Mistral Large | `openrouter:mistralai/mistral-large-2411` | Strong multilingual |
| Lapa LLM | local vLLM (commented out) | Ukrainian-specific, Gemma-3-12B based |
| MamayLM | local vLLM (commented out) | Ukrainian-specific, Gemma-3-4B based |

Lapa and MamayLM are not on OpenRouter (as of 2026-02). To enable them:

```bash
# Terminal 1
vllm serve lapa-llm/lapa-v0.1.2-instruct --port 8001 --dtype bfloat16

# Terminal 2
vllm serve INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0 --port 8002 --dtype bfloat16
```

Then uncomment the `http` provider sections in `promptfooconfig.yaml`.

## Prompt Strategies

| File | Description |
|------|-------------|
| `prompts/zero_shot.txt` | Minimal instruction, no examples |
| `prompts/few_shot.txt` | System instruction + 3 curated dialect examples |
| `prompts/rules_aware.txt` | System instruction with condensed Hutsul transformation rules |

## Assertions

| Assertion | Type | Threshold | Purpose |
|-----------|------|-----------|---------|
| Cyrillic vowel check | `contains-any` | - | Output is not empty or garbage |
| Literary Ukrainian fluency | `llm-rubric` (Gemini Flash) | 0.6 | No remaining dialect features |
| chrF score | `javascript` (custom) | 0.45 | Character-level similarity to reference |

## Running the Full Dataset

```bash
make eval-dialect-prepare-all   # Uses all ~255 rows instead of 50
make eval-dialect
```

## File Layout

```
src/scripts/promptfoo-dialect-eval/
- promptfooconfig.yaml          Main eval config (prompts x providers x assertions)
- prepare_eval_data.py          Converts eval.csv -> test_cases/hutsul.json
- README.md                     This file
- prompts/
    - zero_shot.txt             Minimal system+user prompt
    - few_shot.txt              3 curated examples prepended
    - rules_aware.txt           Condensed Hutsul rules in system prompt
- assertions/
    - chrf_score.js             Custom chrF character F-score assertion (pure JS)
- test_cases/
    - hutsul.json               Generated from eval.csv (gitignored)
- results/
    - eval-YYYYMMDD_HHMMSS.json Timestamped results (gitignored)
    - baseline_summary.md       Committed summary table of results
```
