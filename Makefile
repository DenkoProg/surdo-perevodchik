
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

# Default Config
MODEL_NAME := google/byt5-large
OUTPUT_DIR := models/byt5-hutsul-large
DATA_PATH := data/parallel
MAX_LEN := 1024
EVAL_MODEL := $(OUTPUT_DIR)
EVAL_OUTPUT_DIR := results/evaluation/$(notdir $(OUTPUT_DIR))

.PHONY: train
train:
	@echo "🚀 Training $(MODEL_NAME) on 24GB GPU..."
	@uv run python -m src.surdo_perevodchik.training.train \
		--train_file $(DATA_PATH)/merged.csv \
		--model_name $(MODEL_NAME) \
		--output_dir $(OUTPUT_DIR) \
		--epochs 10 \
		--batch_size 4 \
		--grad_accum 8 \
		--lr 1e-4 \
		--max_length $(MAX_LEN) \
		--optim adafactor \
		--fp16 \
		--grad_checkpoint \
		--eval_steps 200 \
		--save_steps 200

.PHONY: evaluate
evaluate:
		@echo "🔍 Evaluating model..."
		@uv run python -m surdo_perevodchik.evaluation.evaluate \
			--model_path $(EVAL_MODEL) \
			--test_file $(DATA_PATH)/eval.csv \
			--output_dir $(EVAL_OUTPUT_DIR)


.PHONY: generate-hutsul
generate-hutsul: ## Generate synthetic Hutsul corpus
	@echo "🧪 Generating Hutsul corpus..."
	@uv run python scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/hutsul/synthetic_hutsul_corpus.csv \
		--rules prompts/hutsul_rules_system.txt \
		--dictionary data/dicts/hutsul_ukrainian_dictionary.csv \
		--limit 5000 \
		--model mistralai/ministral-14b-2512 \
		--batch-size 3

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help