# Feature: Promptfoo LLM Baseline Evaluation - Hutsul Dialect Translation

**Goal**: Evaluate 5 LLMs on Hutsul-to-standard-Ukrainian translation using promptfoo.
Establishes a baseline before comparing with fine-tuned models.

**Directory**: `src/scripts/promptfoo-dialect-eval/`
**Reference config**: `examples/promptfoo/promptfoo.yaml`
**Reference data script**: `examples/promptfoo/prepare_test_data.py`
**Reference assertion**: `examples/promptfoo/assertions/image_relevance.js`

---

## Phase 1: Directory Scaffold

- [x] Create `src/scripts/promptfoo-dialect-eval/` with subdirectories: `prompts/`, `assertions/`, `test_cases/`, `results/`
- [x] Add `.gitkeep` files to `test_cases/` and `results/` so directories are committed
- [x] Create `src/scripts/promptfoo-dialect-eval/.gitignore` ignoring `test_cases/hutsul.json` and `results/*.json`

## Phase 2: Data Preparation Script

- [x] Create `src/scripts/promptfoo-dialect-eval/prepare_eval_data.py`
  - Reads `data/parallel/eval.csv` (columns: `source`, `target`)
  - Writes `test_cases/hutsul.json` as a list of `{ "vars": { "source": "...", "reference": "..." } }`
  - `--limit` CLI option (default: 50; pass 0 for all rows)
  - Uses `typer` for CLI, `csv.DictReader` for reading, `json.dump` with `ensure_ascii=False`

## Phase 3: Prompt Files

- [x] Create `src/scripts/promptfoo-dialect-eval/prompts/zero_shot.txt`
  - JSON chat array `[{"role":"system",...},{"role":"user","content":"{{source}}"}]`
  - System: "Translate from Hutsul dialect to standard literary Ukrainian. Provide only the translation."
- [x] Create `src/scripts/promptfoo-dialect-eval/prompts/few_shot.txt`
  - Same system + user message with 3 curated dialect examples prepended before `{{source}}`
  - Examples drawn from `data/parallel/eval.csv`: short, illustrative pairs showing key dialect features
- [x] Create `src/scripts/promptfoo-dialect-eval/prompts/rules_aware.txt`
  - System prompt includes condensed Hutsul transformation rules from `prompts/hutsul_rules_system.txt`
  - Covers: vowel changes (єк->як, и->і, у->ю, ві->ви), negation (ни->не), conjunctions (тай->та й/і), reflexive (си/сі->ся), verb endings (-єт->-є)

## Phase 4: Custom chrF Assertion

- [x] Create `src/scripts/promptfoo-dialect-eval/assertions/chrf_score.js`
  - Pure JavaScript, no npm dependencies (runs in Node.js as-is)
  - Implements chrF: character n-grams n=1..6, beta=2 (recall-weighted), averaged
  - Returns `{ pass: score >= 0.45, score: float, reason: string }`
  - Reads reference from `context.vars.reference`

## Phase 5: Main promptfoo Config

- [x] Create `src/scripts/promptfoo-dialect-eval/promptfooconfig.yaml`
  - 3 prompt variants: zero_shot, few_shot, rules_aware
  - 3 active cloud providers via OpenRouter: `openai/gpt-4o-mini`, `google/gemini-2.0-flash-001`, `mistralai/mistral-large-2411`
  - 2 commented-out local providers: Lapa LLM (port 8001), MamayLM (port 8002) - with setup instructions in comments
  - `defaultTest` assertions: `contains-any` (Cyrillic vowels), `llm-rubric` (Gemini Flash judge, threshold 0.6), `chrf_score.js` (threshold 0.45)
  - `maxConcurrency: 2`, `temperature: 0.1`, `max_tokens: 512`

## Phase 6: Makefile Targets

- [x] Add to `Makefile`:
  - `eval-dialect-prepare` - generates 50 test cases (default, fast iteration)
  - `eval-dialect-prepare-all` - generates all ~255 test cases
  - `eval-dialect` - runs `promptfoo eval` with timestamped output
  - `eval-dialect-view` - opens promptfoo web UI

## Phase 7: Documentation

- [x] Create `src/scripts/promptfoo-dialect-eval/README.md` with quick start, model table, prompt strategy table, assertion table, and file layout

---

## Phase 8: Smoke Test and Dry Run

- [ ] Run `make eval-dialect-prepare`
  - Confirm `test_cases/hutsul.json` exists with exactly 50 objects
  - Confirm each object has `{ "vars": { "source": "...", "reference": "..." } }`
  - Spot-check 3 random entries against `data/parallel/eval.csv`
- [ ] Run config validation:
  ```bash
  promptfoo eval --config src/scripts/promptfoo-dialect-eval/promptfooconfig.yaml --dry-run
  ```
  - Confirm no YAML parse errors
  - Confirm all `file://` references resolve (prompts, assertions)
  - Confirm 50 test cases loaded
- [ ] Single-model smoke test (cheap):
  ```bash
  OPENROUTER_API_KEY=$OPENROUTER_API_KEY promptfoo eval \
    --config src/scripts/promptfoo-dialect-eval/promptfooconfig.yaml \
    --providers openrouter:google/gemini-2.0-flash-001 \
    --max-concurrency 1 \
    --output /tmp/smoke.json
  ```
  - Confirm `contains-any` assertion passes
  - Confirm `chrf_score.js` returns a numeric score
  - Confirm `llm-rubric` runs

---

## Phase 9: Full Baseline Eval (3 Cloud Models)

- [ ] Run `make eval-dialect` (3 providers x 3 prompts x 50 test cases = 450 LLM calls)
  - Estimated cost: ~$0.10-0.20 total
- [ ] Run `make eval-dialect-view` and review results grid
- [ ] Record aggregate pass rates per provider and prompt in `src/scripts/promptfoo-dialect-eval/results/baseline_summary.md`:
  ```markdown
  | Model | Prompt | chrF (avg) | Fluency pass% | Notes |
  |-------|--------|-----------|--------------|-------|
  | GPT-4o Mini | zero_shot | ... | ...% | |
  ...
  ```

---

## Phase 10: Ukrainian-Specific Models (Follow-up)

- [ ] Research Lapa LLM local deployment
  - Model: `lapa-llm/lapa-v0.1.2-instruct` (~12B parameters, Gemma-3-12B base)
  - Check GPU availability: `nvidia-smi`
  - Serve: `vllm serve lapa-llm/lapa-v0.1.2-instruct --port 8001 --dtype bfloat16`
  - Uncomment Lapa provider in `promptfooconfig.yaml`
- [ ] Research MamayLM local deployment
  - Model: `INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0` (already in Makefile as `DEC_ONLY_MODEL`)
  - Serve: `vllm serve INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0 --port 8002 --dtype bfloat16`
  - Uncomment MamayLM provider in `promptfooconfig.yaml`
- [ ] Re-run full eval with all 5 models
- [ ] Update `baseline_summary.md` with Ukrainian model results

---

## Phase 11: Extend to All Dialects (Future)

- [ ] Add `data/parallel/boiko/` eval subset (pick 50 representative rows)
- [ ] Add `data/parallel/surzhyk/` eval subset
- [ ] Add `data/parallel/transcarpathian/` eval subset
- [ ] Create corresponding `prepare_eval_data_*.py` scripts or extend with a `--dialect` flag
- [ ] Add `promptfooconfig_boiko.yaml`, `promptfooconfig_surzhyk.yaml`, etc.
  (or a single config with dialect variable)
- [ ] Create Makefile targets: `eval-boiko`, `eval-surzhyk`, `eval-transcarpathian`

---

## File Summary

| File | Status | Purpose |
|------|--------|---------|
| `src/scripts/promptfoo-dialect-eval/promptfooconfig.yaml` | Done | Main eval config |
| `src/scripts/promptfoo-dialect-eval/prepare_eval_data.py` | Done | CSV -> JSON test cases |
| `src/scripts/promptfoo-dialect-eval/prompts/zero_shot.txt` | Done | Minimal prompt |
| `src/scripts/promptfoo-dialect-eval/prompts/few_shot.txt` | Done | 3-example prompt |
| `src/scripts/promptfoo-dialect-eval/prompts/rules_aware.txt` | Done | Rules-guided prompt |
| `src/scripts/promptfoo-dialect-eval/assertions/chrf_score.js` | Done | chrF score assertion |
| `src/scripts/promptfoo-dialect-eval/.gitignore` | Done | Ignore generated files |
| `src/scripts/promptfoo-dialect-eval/README.md` | Done | Usage documentation |
| `Makefile` | Done | 4 new eval-dialect targets |
| `src/scripts/promptfoo-dialect-eval/results/baseline_summary.md` | Pending | Results summary table |
