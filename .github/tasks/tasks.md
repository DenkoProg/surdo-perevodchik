# Project Roadmap: Ukrainian Dialect → Standard Ukrainian

**Repository**: `surdo-perevodchik`
**Goal**: Normalize text from Ukrainian dialects and Surzhyk into standard literary Ukrainian.

---

## Target Dialects (Priority Order)

| # | Dialect | Status | Data Available | Next Action |
|---|---------|--------|----------------|-------------|
| 1 | **Hutsul** | 🟢 In Progress | ~29K pairs | Evaluate if sufficient |
| 2 | **Russian-Ukrainian Surzhyk** | 🟡 Planned | Programmatic generation possible | Create errorification rules |
| 3 | **Trans-Carpathian** | 🟡 Planned | Freiburg Rusyn Corpus + Pankevych DB | Contact universities |
| 4 | **Boyko** | 🟡 Planned | Ethnographic records, GRAC | Web scraping + rules |

---

## Current State

### What We Have
- [x] Project structure with training pipeline (`src/surdo_perevodchik/`)
- [x] umT5-base model fine-tuned on Hutsul (~29K pairs)
- [x] Evaluation framework (BLEU, chrF++, TER)
- [x] Gradio demo app (`app.py`)
- [x] Synthetic data generation script (`generate_corpus.py`)
- [x] Hutsul dictionary (~7K words)
- [x] Hutsul linguistic rules prompt

### What We Need to Decide
- [ ] **Is 29K pairs enough for Hutsul?** → Evaluate current model quality
- [ ] **Should we expand Hutsul first or add new dialects?**
- [ ] **Which LLM to fine-tune: Lapa, MamayLM, or Mistral?**

---

## Phase 0: Evaluation & Decision (CURRENT)

> **Objective**: Understand current model quality to inform next steps.

### 0.1 Evaluate Current Hutsul Model
- [ ] Run evaluation on `umt5-base-hutsul-baseline` checkpoint
- [ ] Analyze BLEU/chrF++ scores — is it production-ready?
- [ ] Manual review: sample 50 translations, note error patterns
- [ ] **Decision**: If BLEU < 40 → need more data; If BLEU ≥ 40 → can proceed to LLM fine-tuning

### 0.2 Data Sufficiency Analysis
- [ ] Compare with similar low-resource translation projects (target: 50-100K pairs)
- [ ] Estimate cost/time to generate additional Hutsul data (current: 29K)
- [ ] Estimate cost/time to create first version of other dialects

### 0.3 LLM Selection Research
- [ ] Test Lapa LLM on raw dialect translation (zero-shot)
- [ ] Test MamayLM-Gemma-3-4B on same samples
- [ ] Test Mistral-7B-Instruct on same samples
- [ ] Compare: quality, inference speed, VRAM requirements
- [ ] **Decision**: Select primary LLM for fine-tuning

---

## Phase 1: Data Expansion

> **Objective**: Build sufficient parallel corpora for all 4 target dialects.

### 1.1 Hutsul (Expansion if needed)
- [ ] Analyze error patterns from Phase 0 evaluation
- [ ] Generate additional synthetic pairs targeting weak areas
- [ ] Target: 50K total pairs (currently: 29K → generate ~20K more)

### 1.2 Russian-Ukrainian Surzhyk
- [ ] Adapt `RussismErrorifier` from `unlp/errorification/`
- [ ] Create `prompts/surzhyk_rules.txt`:
  - Phonetic patterns (і→и, є→е, unstressed о→а)
  - Lexical calques (повинен→должен, тільки→только)
  - Grammatical interference (Russian verb endings, case confusion)
- [ ] Generate 30K pairs programmatically + 10K via LLM
- [ ] Target: 40K total pairs

### 1.3 Trans-Carpathian
- [ ] Contact University of Freiburg (Achim Rabus) for Rusyn Corpus
- [ ] Request access to Pankevych Lexical Database (CEEOL)
- [ ] Create `prompts/transcarpathian_rules.txt`:
  - Hungarian/Slovak loanwords
  - Vowel shifts specific to region
  - Four subdialects: Borzhava, Uzh, Maramorosh, Verkhovyna
- [ ] Target: 20K pairs (10K from sources + 10K synthetic)

### 1.4 Boyko
- [ ] Scrape ukrainer.net for Boyko dialect texts
- [ ] Extract Boyko data from GRAC corpus (if accessible)
- [ ] Create `prompts/boyko_rules.txt`:
  - Similar to Hutsul (Carpathian group) but less extreme vowel shifts
  - Archaic Proto-Slavic features
  - Mountain terminology
- [ ] Target: 15K pairs

### 1.5 Quality Assurance
- [ ] Run LLM-as-judge filtering (min score 3.5/5)
- [ ] Sample 100 pairs per dialect for human validation
- [ ] Remove duplicates and outliers (length ratio > 2.0)

---

## Phase 2: Encoder-Decoder Baseline

> **Objective**: Establish baseline metrics with umT5/mBART.

### 2.1 Multi-Dialect Training
- [ ] Combine all dialects into unified dataset
- [ ] Add dialect tags to input: `<hutsul>`, `<surzhyk>`, `<transcarpathian>`, `<boyko>`
- [ ] Train umT5-base on combined corpus
- [ ] Evaluate per-dialect and overall metrics

### 2.2 Baseline Metrics Target
| Dialect | Target BLEU | Target chrF++ |
|---------|-------------|---------------|
| Hutsul | ≥ 45 | ≥ 65 |
| Surzhyk | ≥ 50 | ≥ 70 |
| Trans-Carpathian | ≥ 40 | ≥ 60 |
| Boyko | ≥ 40 | ≥ 60 |

---

## Phase 3: LLM Fine-Tuning

> **Objective**: Achieve production-quality with instruction-tuned LLM.

### 3.1 Data Formatting
- [ ] Convert parallel corpus to instruction format:
  ```
  User: Переклади цей текст літературною українською мовою: <dialect_text>
  Assistant: <standard_ukrainian>
  ```
- [ ] Create train/val/test splits (80/10/10)

### 3.2 LoRA Fine-Tuning
- [ ] Configure LoRA: r=16, alpha=32, target_modules=[q,k,v,o,gate,up,down]
- [ ] Train with QLoRA (4-bit quantization) for memory efficiency
- [ ] Hyperparameters: lr=5e-5, epochs=3, batch=1, grad_accum=128
- [ ] Monitor with W&B

### 3.3 Evaluation
- [ ] Compare with encoder-decoder baseline
- [ ] LLM-as-judge scoring (fluency, adequacy, dialectal authenticity)
- [ ] A/B testing with native speakers (if possible)

---

## Phase 4: Demo & Deployment

> **Objective**: Make the system usable.

### 4.1 Gradio Demo Update
- [ ] Add dialect selector dropdown (4 dialects)
- [ ] Show model confidence/scores
- [ ] Add example sentences for each dialect

### 4.2 Model Publishing
- [ ] Push best model to Hugging Face Hub
- [ ] Create model card with usage examples
- [ ] Document limitations and known issues

---

## Immediate Next Steps

1. **Run Hutsul evaluation** (`make evaluate-encoder-decoder`)
2. **Review evaluation results** — is 29K pairs enough?
3. **Zero-shot test** Lapa/MamayLM/Mistral on dialect samples
4. **Decision meeting**: Expand Hutsul OR start new dialects first?

---

## Resources & Contacts

| Resource | URL/Contact | Purpose |
|----------|-------------|---------|
| Freiburg Rusyn Corpus | russinisch.de | Trans-Carpathian data |
| Pankevych Database | CEEOL.com | Trans-Carpathian lexicon |
| GRAC Corpus | grfrequency.org.ua | Regional Ukrainian texts |
| Ukrainer.net | ukrainer.net | Boyko ethnographic texts |
| Lapa LLM | HuggingFace (TBD) | Primary LLM candidate |
| MamayLM | INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0 | Alternative LLM |
