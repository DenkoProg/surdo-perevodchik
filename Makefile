
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
# Encoder-Decoder Models (mT5, ByT5, etc.)
# =============================================================================

ENC_DEC_MODEL := google/byt5-large
ENC_DEC_OUTPUT := models/byt5-hutsul-large
DATA_PATH := data/parallel
ENC_DEC_MAX_LEN := 1024

.PHONY: train-encoder-decoder
train-encoder-decoder: ## Fine-tune encoder-decoder model (mT5, ByT5)
	@echo "🚀 Training encoder-decoder: $(ENC_DEC_MODEL)..."
	@uv run python -m src.surdo_perevodchik.training.train_encoder_decoder \
		--train_file $(DATA_PATH)/merged.csv \
		--model_name $(ENC_DEC_MODEL) \
		--output_dir $(ENC_DEC_OUTPUT) \
		--epochs 10 \
		--batch_size 4 \
		--grad_accum 8 \
		--lr 1e-4 \
		--max_length $(ENC_DEC_MAX_LEN) \
		--optim adafactor \
		--fp16 \
		--grad_checkpoint \
		--eval_steps 200 \
		--save_steps 200

.PHONY: evaluate-encoder-decoder
evaluate-encoder-decoder: ## Evaluate encoder-decoder model
	@echo "🔍 Evaluating encoder-decoder model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_encoder_decoder \
		--model_path $(ENC_DEC_OUTPUT) \
		--test_file $(DATA_PATH)/eval.csv \
		--output_dir results/evaluation/$(notdir $(ENC_DEC_OUTPUT))

# =============================================================================
# Decoder-Only Models (MamayLM, Gemma, Llama, etc.)
# =============================================================================

DEC_ONLY_MODEL := INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0
DEC_ONLY_OUTPUT := models/mamaylm-hutsul
DEC_ONLY_MAX_LEN := 512

.PHONY: train-decoder-only
train-decoder-only: ## Fine-tune decoder-only model with LoRA (MamayLM, Gemma)
	@echo "🚀 Training decoder-only: $(DEC_ONLY_MODEL)..."
	@uv run python -m src.surdo_perevodchik.training.train_decoder_only \
		--train_file $(DATA_PATH)/merged.csv \
		--model_name $(DEC_ONLY_MODEL) \
		--output_dir $(DEC_ONLY_OUTPUT) \
		--epochs 3 \
		--batch_size 1 \
		--grad_accum 16 \
		--lr 2e-5 \
		--max_length $(DEC_ONLY_MAX_LEN) \
		--bf16 \
		--grad_checkpoint \
		--use_lora \
		--lora_r 16 \
		--lora_alpha 32 \
		--use_4bit \
		--eval_steps 100 \
		--save_steps 100

.PHONY: train-decoder-only-full
train-decoder-only-full: ## Full fine-tune decoder-only model (requires more VRAM)
	@echo "🚀 Full fine-tuning decoder-only: $(DEC_ONLY_MODEL)..."
	@uv run python -m src.surdo_perevodchik.training.train_decoder_only \
		--train_file $(DATA_PATH)/merged.csv \
		--model_name $(DEC_ONLY_MODEL) \
		--output_dir $(DEC_ONLY_OUTPUT)-full \
		--epochs 3 \
		--batch_size 1 \
		--grad_accum 16 \
		--lr 5e-6 \
		--max_length $(DEC_ONLY_MAX_LEN) \
		--bf16 \
		--grad_checkpoint \
		--eval_steps 100 \
		--save_steps 100

.PHONY: evaluate-decoder-only
evaluate-decoder-only: ## Evaluate decoder-only model
	@echo "🔍 Evaluating decoder-only model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_decoder_only \
		--model_path $(DEC_ONLY_OUTPUT) \
		--test_file $(DATA_PATH)/eval.csv \
		--output_dir results/evaluation/$(notdir $(DEC_ONLY_OUTPUT)) \
		--use_4bit

.PHONY: evaluate-decoder-only-base
evaluate-decoder-only-base: ## Evaluate base decoder-only model (before fine-tuning)
	@echo "🔍 Evaluating base decoder-only model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_decoder_only \
		--model_path $(DEC_ONLY_MODEL) \
		--test_file $(DATA_PATH)/eval.csv \
		--output_dir results/evaluation/$(notdir $(DEC_ONLY_MODEL))-base \
		--use_4bit

# =============================================================================

.PHONY: generate-hutsul
generate-hutsul: ## Generate synthetic Hutsul corpus (OpenRouter API)
	@echo "🧪 Generating Hutsul corpus..."
	@uv run python scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/hutsul/synthetic_hutsul_corpus.csv \
		--rules prompts/hutsul_rules_system.txt \
		--dictionary data/dicts/hutsul_ukrainian_dictionary.csv \
		--limit 15000 \
		--model mistralai/mistral-7b-instruct:free \
		--batch-size 3

.PHONY: generate-hutsul-local
generate-hutsul-local: ## Generate synthetic Hutsul corpus (Local GPU, 8-bit quantization)
	@echo "🚀 Generating Hutsul corpus with local GPU model..."
	@uv run python scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/hutsul/synthetic_hutsul_corpus.csv \
		--rules prompts/hutsul_rules_system.txt \
		--dictionary data/dicts/hutsul_ukrainian_dictionary.csv \
		--provider local \
		--model mistralai/Mistral-7B-Instruct-v0.2 \
		--load-in-8bit \
		--batch-size 5 \
		--limit 15000

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help