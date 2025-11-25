# Surdo Perevodchik - Machine Translation Project

This project trains sequence-to-sequence models (mT5-based) to translate between Standard Ukrainian and Hutsul dialect using Hugging Face Transformers.

## Architecture Overview

**Core Components:**
- `src/surdo_perevodchik/training/train.py` - Fine-tunes mT5 models on parallel corpora using Seq2SeqTrainer
- `src/surdo_perevodchik/evaluation/evaluate.py` - Generates predictions and computes BLEU/chrF++/TER metrics
- `src/surdo_perevodchik/evaluation/metrics.py` - Wrapper around sacrebleu for standardized metrics
- `src/notebooks/load_base_corpora.ipynb` - Data preparation from HuggingFace datasets

**Data Flow:**
1. Raw datasets (`grammarly/spivavtor`, `hutsul/hutsul-manually-annotated`) → `data/raw/` and `data/parallel/`
2. Parallel corpus (`hutsul_parallel.csv`) with `source,target` columns feeds training
3. Models saved to `models/mt5-hutsul-small/` with checkpoints in subdirectories
4. Evaluation outputs predictions, references, and metrics JSON to `results/evaluation/`

**Key Design Decisions:**
- Uses `rootutils` for project root discovery (`.project-root` marker) - enables imports from notebooks
- CSV format for all datasets (not JSONL/Parquet) with strict `source`/`target` column naming
- Module execution via `python -m src.surdo_perevodchik.training.train` (note: inconsistent - eval uses `surdo_perevodchik.evaluation`)
- Generation parameters tuned for dialect translation: beam_search=5, `no_repeat_ngram_size=3`, blocks `<extra_id_*>` tokens

## Developer Workflows

**Setup:**
```bash
make install  # Creates venv, runs uv sync --frozen, installs pre-commit hooks
```

**Training:**
```bash
make train  # Runs with default hyperparams: epochs=15, batch_size=16, lr=5e-5
# Or customize: uv run python -m src.surdo_perevodchik.training.train --epochs 20 --batch_size 32
```

**Evaluation:**
```bash
make evaluate  # Generates predictions and computes BLEU/chrF++/TER metrics
# Outputs to results/evaluation/{predictions.txt, references.txt, results.json}
```

**Code Quality:**
```bash
make format  # Runs ruff format + ruff check --fix
make lint    # Runs ruff check (fails CI on violations)
```

**Running Python Code:**
- Always prefix with `uv run` (e.g., `uv run python script.py`) to use managed environment
- Training/eval modules must use `-m` module syntax, not direct file paths
- Notebooks require `rootutils.setup_root(Path.cwd(), indicator=".project-root", pythonpath=True)` before project imports

## Project-Specific Conventions

**Import Patterns:**
- Notebooks: Use `rootutils` to set pythonpath before importing project modules
- Scripts: Import as `from surdo_perevodchik.evaluation.metrics import compute_metrics` (no `src.` prefix in eval)
- Inconsistency alert: Training uses `src.surdo_perevodchik` while eval uses `surdo_perevodchik` - check Makefile for correct invocation

**Data Format Standards:**
- All CSV files must have headers (`source,target` for parallel data)
- Dataset splits use 90/10 train/val by default (`val_size=0.1` in train.py)
- Text encoding: UTF-8 everywhere, newline-delimited predictions/references

**Model Configuration:**
- Default base model: `google/mt5-small`
- Tokenizer: `use_fast=False` (compatibility with mT5 SentencePiece)
- Generation hyperparams hardcoded in `evaluate.py` (not configurable via CLI)
- Bad words IDs `[[250099], [250098], [250097]]` block unwanted special tokens

**Code Style (Ruff Config):**
- Line length: 119 characters
- Quote style: Double quotes everywhere (strings, docstrings)
- Import sorting: Combined imports (`from x import a, b`), force-sort within sections
- Ignored rules: E501 (line length), F401 (unused imports), D1 (missing docstrings)

## Critical Integration Points

**Hugging Face Datasets:**
- Loads external datasets: `grammarly/spivavtor`, `hutsul/hutsul-manually-annotated`
- Uses `load_dataset("csv", data_files=...)` for local CSV files

**Torch & Transformers:**
- Device selection: Auto-detects CUDA, falls back to CPU
- FP16 training disabled by default (`fp16=False` in training args)
- Model checkpoints: Saves every 500 steps, keeps only last 2 checkpoints

**Dependency Management:**
- Package manager: `uv` (not pip/poetry/conda)
- Lockfile: `uv.lock` for reproducibility (use `uv sync --frozen` in CI)
- Dev extras: Includes `ipykernel`, `pre-commit`, `ruff` for development

**Pre-commit Hooks:**
- Auto-installed via `make install`
- Enforces Ruff formatting/linting before commits
