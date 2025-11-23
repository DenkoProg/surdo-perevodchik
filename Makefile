
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
		--epochs 3 \
		--batch_size 8 \
		--lr 5e-5

.PHONY: evaluate
evaluate:
	@echo "🔍 Evaluating model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_model \
		--model_path models/mt5-hutsul-small \
		--test_file data/parallel/hutsul_parallel.csv \
		--output_dir results/evaluation

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help