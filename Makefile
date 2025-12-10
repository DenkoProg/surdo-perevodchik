
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

.PHONY: train
train:
	@echo "🚀 Training mt5 model..."
	@uv run python -m src.surdo_perevodchik.training.train \
		--train_file data/parallel/hutsul_parallel.csv \
		--model_name google/mt5-small \
		--output_dir models/mt5-hutsul-small \
		--epochs 15 \
		--batch_size 16 \
		--lr 5e-5

.PHONY: evaluate
evaluate:
	@echo "🔍 Evaluating model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate \
		--model_path models/mt5-hutsul-small \
		--test_file data/parallel/hutsul_parallel.csv \
		--output_dir results/evaluation


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