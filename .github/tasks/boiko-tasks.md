# Task: Parse Boykivian Dictionary (boiko-dict.md) into Structured CSV

**Goal**: Extract structured `source`/`target` word pairs from the raw OCR'd Boykivian dialect dictionary (`examples/boiko-dict.md`, 75k lines) into a CSV matching the format of `data/dicts/hutsul_ukrainian_dictionary.csv`.

**Approach**: Hybrid (Regex segmentation + LLM extraction via free OpenRouter model)

---

## Source & Target

**Source** (`examples/boiko-dict.md`, lines 1981-60671):
```
БАШНЯ, башна /Пі-ц/ "вежа", "башта"; силосна башна /Пі-ц/ "силосна башта"
БАЮРА, байура 1. "калюжа" /Б-те, Ж-ня/; 2. "болото" /С. Коло Км./
```
- Headwords in UPPERCASE, definitions in "quotes", location codes in /slashes/
- Multi-meaning entries (numbered 1, 2, 3...), cross-references (`Див.`, `Ще:`)
- Heavy OCR artifacts: garbled chars, unbalanced quotes, digits replacing letters
- Lines 60680+: alphabetical index (can be used for validation)

**Target** (`data/dicts/boykivian_ukrainian_dictionary.csv`):
```csv
,Boykivian,Ukrainian,uk_lemma
0,башня,вежа,вежа
1,башня,башта,башта
2,баюра,калюжа,калюжа
3,баюра,болото,болото
```

---

## Implementation Steps

### Step 1: Entry segmentation (regex, deterministic)
- [ ] Read lines 1981-60671 from `boiko-dict.md`
- [ ] Strip standalone page numbers (lines matching `^\d{1,3}$` between blank lines)
- [ ] Split into entries by detecting UPPERCASE headword lines: `^[А-ЯІЇЄҐ][А-ЯІЇЄҐ' -]+`
- [ ] Accumulate continuation lines into each entry
- [ ] Expected output: ~4,500-5,000 raw entries

### Step 2: Pre-filter cross-references
- [ ] Skip entries that are only cross-references (contain `Див.` but no quoted definitions)
- [ ] ~1,200 entries expected to be filtered out
- [ ] Log skipped entries for review

### Step 3: LLM batch extraction
- [ ] Batch ~20 entries per API call via existing `OpenRouterClient`
- [ ] Use a free model on OpenRouter (e.g. `google/gemma-2-9b-it:free`, `meta-llama/llama-3.1-8b-instruct:free`, or `qwen/qwen-2.5-7b-instruct:free`)
- [ ] System prompt: extract `{boykivian, ukrainian, uk_lemma}` from OCR'd dictionary text
- [ ] Use JSON output (free models may not support strict structured output — parse with fallback)
- [ ] One row per meaning (word with 3 meanings → 3 rows)
- [ ] Estimated ~200-350 API calls, $0 cost (free tier)
- [ ] Add appropriate delays between batches (free model rate limits)
- [ ] Expected accuracy ~80-85%

### Step 4: Post-processing & CSV output
- [ ] Deduplicate exact matches
- [ ] Validate non-empty `source` and `target` for every row
- [ ] Write CSV to `data/dicts/boykivian_ukrainian_dictionary.csv`
- [ ] Cross-validate headword coverage against the alphabetical index (lines 60680+)

### Step 5: Verification
- [ ] Run the script: `python src/scripts/parse_boiko_dict.py`
- [ ] Check CSV has ~10,000-14,000 rows with non-empty source/target
- [ ] Spot-check 20-30 random entries against the source text
- [ ] Compare CSV headwords against the alphabetical index for coverage

---

## Files

| Action | Path |
|--------|------|
| **Create** | `src/scripts/parse_boiko_dict.py` (~200 lines) |
| **Read** | `examples/boiko-dict.md` (source dictionary) |
| **Read** | `data/dicts/hutsul_ukrainian_dictionary.csv` (format reference) |
| **Reuse** | `src/surdo_perevodchik/data_generation/openrouter_client.py` (LLM client) |
| **Output** | `data/dicts/boykivian_ukrainian_dictionary.csv` |
