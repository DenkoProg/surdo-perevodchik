
# =============================================================================
# Variables
# =============================================================================

DATA_PATH := data/parallel

ENC_DEC_MODEL        := google/umt5-base
ENC_DEC_MULTI_OUTPUT := models/umt5-base-multidialect
ENC_DEC_MAX_LEN      := 256

DEC_ONLY_MODEL        := INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0
DEC_ONLY_MULTI_OUTPUT := models/mamaylm-multidialect
DEC_ONLY_MAX_LEN      := 512

GEN_MODEL := mistralai/Mistral-Small-24B-Instruct-2501
GEN_SEED  := 42

# =============================================================================
# Setup
# =============================================================================

.PHONY: install
install: ## Install dependencies and setup pre-commit hooks
	@echo "🚀 Installing dependencies from lockfile"
	@uv sync --frozen
	@uv run pre-commit install

.PHONY: lint
lint: ## Run ruff linter
	uv run ruff check

.PHONY: format
format: ## Format code and fix linting issues
	uv run ruff format
	uv run ruff check --fix

# =============================================================================
# Data Preparation
# =============================================================================

.PHONY: prepare-data
prepare-data: ## Prepare multi-dialect train/val/test splits from all dialect corpora
	@echo "Preparing multi-dialect data..."
	@uv run python src/scripts/prepare_multidialect_data.py \
		--output_dir $(DATA_PATH) \
		--val_size 0.1 \
		--eval_per_dialect 50 \
		--seed 42

.PHONY: generate-all-local
generate-all-local: generate-hutsul-local generate-boiko-local generate-transcarpathian-local generate-surzhyk-local ## Generate all dialect corpora sequentially (Local GPU)

.PHONY: generate-hutsul-local
generate-hutsul-local: ## Generate synthetic Hutsul corpus (Local GPU, 4-bit quantization)
	@echo "🚀 Generating Hutsul corpus with local GPU model..."
	@uv run python src/scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/hutsul/synthetic_hutsul_corpus.csv \
		--rules prompts/hutsul_rules_system.txt \
		--dictionary data/dicts/hutsul_ukrainian_dictionary.csv \
		--provider local \
		--model $(GEN_MODEL) \
		--load-in-4bit \
		--random-seed $(GEN_SEED) \
		--batch-size 5 \
		--limit 30000

.PHONY: generate-boiko-local
generate-boiko-local: ## Generate synthetic Boikivian corpus (Local GPU, 4-bit quantization)
	@echo "🚀 Generating Boikivian corpus with local GPU model..."
	@uv run python src/scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/boiko/synthetic_boiko_corpus.csv \
		--rules prompts/boiko_rules_system.txt \
		--dictionary data/dicts/boykivian_ukrainian_dictionary.csv \
		--provider local \
		--model $(GEN_MODEL) \
		--load-in-4bit \
		--random-seed $(GEN_SEED) \
		--batch-size 5 \
		--limit 30000

.PHONY: generate-transcarpathian-local
generate-transcarpathian-local: ## Generate synthetic Transcarpathian corpus (Local GPU, 4-bit quantization)
	@echo "🚀 Generating Transcarpathian corpus with local GPU model..."
	@uv run python src/scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/transcarpathian/synthetic_transcarpathian_corpus.csv \
		--rules prompts/transcarpathian_rules_system.txt \
		--dictionary data/dicts/transcarpathian_ukrainian_dictionary.csv \
		--provider local \
		--model $(GEN_MODEL) \
		--load-in-4bit \
		--random-seed $(GEN_SEED) \
		--batch-size 5 \
		--limit 30000

.PHONY: generate-surzhyk-local
generate-surzhyk-local: ## Generate synthetic Surzhyk corpus (Local GPU, 4-bit quantization)
	@echo "🚀 Generating Surzhyk corpus with local GPU model..."
	@uv run python src/scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/surzhyk/synthetic_surzhyk_corpus_llm.csv \
		--rules prompts/surzhyk_rules_system.txt \
		--dictionary data/dicts/surzhyk_ukrainian_dictionary.csv \
		--provider local \
		--model $(GEN_MODEL) \
		--load-in-4bit \
		--random-seed $(GEN_SEED) \
		--batch-size 5 \
		--limit 30000

# =============================================================================
# Training (requires prepare-data first)
# =============================================================================

.PHONY: train-decoder-only-multi
train-decoder-only-multi: ## Fine-tune MamayLM on all dialects with QLoRA
	@echo "Training decoder-only on all dialects: $(DEC_ONLY_MODEL)..."
	@uv run python -m src.surdo_perevodchik.training.train_decoder_only \
		--train_file $(DATA_PATH)/train.csv \
		--val_file $(DATA_PATH)/val.csv \
		--model_name $(DEC_ONLY_MODEL) \
		--output_dir $(DEC_ONLY_MULTI_OUTPUT) \
		--epochs 3 \
		--batch_size 1 \
		--grad_accum 4 \
		--lr 2e-4 \
		--max_length $(DEC_ONLY_MAX_LEN) \
		--grad_checkpoint \
		--use_lora \
		--lora_r 4 \
		--lora_alpha 4 \
		--use_4bit \

.PHONY: train-encoder-decoder-multi
train-encoder-decoder-multi: ## Fine-tune umt5-base on all dialects
	@echo "Training encoder-decoder on all dialects: $(ENC_DEC_MODEL)..."
	@uv run python -m src.surdo_perevodchik.training.train_encoder_decoder \
		--train_file "$(DATA_PATH)/train.csv" \
		--val_file "$(DATA_PATH)/val.csv" \
		--model_name $(ENC_DEC_MODEL) \
		--output_dir $(ENC_DEC_MULTI_OUTPUT)-longer \
		--epochs 40 \
		--batch_size 4 \
		--grad_accum 4 \
		--weight_decay 0.1 \
		--label_smoothing 0.1 \
		--lr 5e-5 \
		--bf16 \
		--optim adamw_bnb_8bit \
		--max_length $(ENC_DEC_MAX_LEN)

# =============================================================================
# Evaluation
# =============================================================================

.PHONY: evaluate-decoder-only-base
evaluate-decoder-only-base: ## Evaluate base MamayLM before fine-tuning (baseline)
	@echo "🔍 Evaluating base decoder-only model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_decoder_only \
		--model_path $(DEC_ONLY_MODEL) \
		--test_file $(DATA_PATH)/test.csv \
		--output_dir results/evaluation/$(notdir $(DEC_ONLY_MODEL))-base \
		--use_4bit

.PHONY: evaluate-decoder-only
evaluate-decoder-only: ## Evaluate fine-tuned MamayLM
	@echo "🔍 Evaluating decoder-only model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_decoder_only \
		--model_path $(DEC_ONLY_MULTI_OUTPUT) \
		--test_file $(DATA_PATH)/test.csv \
		--output_dir results/evaluation/$(notdir $(DEC_ONLY_MULTI_OUTPUT)) \
		--use_4bit

.PHONY: evaluate-encoder-decoder-multi
evaluate-encoder-decoder-multi: ## Evaluate fine-tuned umt5-base (all dialects)
	@echo "🔍 Evaluating multidialect encoder-decoder model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_encoder_decoder \
		--model_path $(ENC_DEC_MULTI_OUTPUT)-longer/final_model \
		--test_file $(DATA_PATH)/test.csv \
		--output_dir results/evaluation/umt5-base-multidialect-longer

.PHONY: eval-dialect-gpt
eval-dialect-gpt: ## Evaluate GPT-4o Mini on Hutsul translation (requires .env with OpenRouter key)
	@cd src/scripts/dialect-eval && python cli.py remote --model gpt-4o-mini

.PHONY: eval-dialect-gemini
eval-dialect-gemini: ## Evaluate Gemini 2.0 Flash on Hutsul translation
	@cd src/scripts/dialect-eval && python cli.py remote --model google/gemini-2.0-flash-001

.PHONY: eval-dialect-mistral
eval-dialect-mistral: ## Evaluate Mistral Large on Hutsul translation
	@cd src/scripts/dialect-eval && python cli.py remote --model mistralai/mistral-large-2411

# =============================================================================
# Utilities
# =============================================================================

.PHONY: tensorboard
tensorboard: ## Launch TensorBoard for all model runs
	@echo "📊 Launching TensorBoard..."
	@uv run tensorboard --logdir models

.PHONY: demo
demo: ## Launch Gradio demo for dialect translation
	@echo "🎭 Launching Gradio demo..."
	@uv run python app.py

.PHONY: pdf
pdf: ## Compile LaTeX document to PDF
	@echo "📄 Compiling LaTeX document..."
	@cd docs && xelatex -interaction=nonstopmode main.tex && xelatex -interaction=nonstopmode main.tex
	@$(MAKE) clean-pdf

.PHONY: clean-pdf
clean-pdf: ## Clean LaTeX auxiliary files
	@echo "🧹 Cleaning LaTeX auxiliary files..."
	@cd docs && rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
