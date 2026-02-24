
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
# Encoder-Decoder Models (mT5, mbart, etc.)
# =============================================================================

ENC_DEC_MODEL := google/umt5-base
ENC_DEC_OUTPUT := models/umt5-base-hutsul-aligned
CHECKPOINT := checkpoint-16560
DATA_PATH := data/parallel
ENC_DEC_MAX_LEN := 256

.PHONY: train-encoder-decoder
train-encoder-decoder: ## Fine-tune encoder-decoder model (mT5, umT5, mbart)
	@echo "🚀 Training encoder-decoder: $(ENC_DEC_MODEL)..."
	@uv run python -m src.surdo_perevodchik.training.train_encoder_decoder \
		--train_file "$(DATA_PATH)/merged.csv" \
		--model_name $(ENC_DEC_MODEL) \
		--output_dir $(ENC_DEC_OUTPUT) \
		--resume_from_checkpoint $(ENC_DEC_OUTPUT)/$(CHECKPOINT) \
		--epochs 20 \
		--batch_size 4 \
		--grad_accum 4 \
		--weight_decay 0.1 \
		--label_smoothing 0.1 \
		--lr 5e-5 \
		--bf16 \
		--optim adamw_bnb_8bit \
		--max_length $(ENC_DEC_MAX_LEN) \

.PHONY: evaluate-encoder-decoder
evaluate-encoder-decoder: ## Evaluate encoder-decoder model
	@echo "🔍 Evaluating encoder-decoder model..."
	@uv run python -m surdo_perevodchik.evaluation.evaluate_encoder_decoder \
		--model_path $(ENC_DEC_OUTPUT)/$(CHECKPOINT) \
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

GEN_MODEL := mistralai/Mistral-Small-24B-Instruct-2501
GEN_SEED  := 42

.PHONY: generate-hutsul
generate-hutsul: ## Generate synthetic Hutsul corpus (OpenRouter API)
	@echo "🧪 Generating Hutsul corpus..."
	@uv run python src/scripts/generate_corpus.py generate \
		--input data/raw/standard_ukrainian.csv \
		--output data/parallel/hutsul/synthetic_hutsul_corpus.csv \
		--rules prompts/hutsul_rules_system.txt \
		--dictionary data/dicts/hutsul_ukrainian_dictionary.csv \
		--limit 20000 \
		--model mistralai/ministral-14b-2512 \
		--random-seed $(GEN_SEED) \
		--batch-size 3

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
generate-surzhyk-local: ## Generate synthetic Surzhyk corpus via LLM (Local GPU, 4-bit quantization)
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

.PHONY: generate-all-local
generate-all-local: generate-hutsul-local generate-boiko-local generate-transcarpathian-local generate-surzhyk-local ## Generate all dialect corpora sequentially (Local GPU)

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

# =============================================================================
# Promptfoo Dialect Evaluation
# =============================================================================

.PHONY: eval-dialect-prepare
eval-dialect-prepare: ## Prepare Hutsul dialect eval test cases (first 50 rows of eval.csv)
	@echo "Preparing dialect eval test cases..."
	@uv run python src/scripts/promptfoo-dialect-eval/prepare_eval_data.py

.PHONY: eval-dialect-prepare-all
eval-dialect-prepare-all: ## Prepare ALL rows of eval.csv as test cases
	@echo "Preparing ALL dialect eval test cases..."
	@uv run python src/scripts/promptfoo-dialect-eval/prepare_eval_data.py --limit 0

.PHONY: eval-dialect
eval-dialect: ## Run Hutsul dialect LLM baseline evaluation (requires OPENROUTER_API_KEY)
	@echo "Running dialect eval..."
	@OPENROUTER_API_KEY=$(OPENROUTER_API_KEY) promptfoo eval \
		--config src/scripts/promptfoo-dialect-eval/promptfooconfig.yaml \
		--output src/scripts/promptfoo-dialect-eval/results/eval-$(shell date +%Y%m%d_%H%M%S).json

.PHONY: eval-dialect-view
eval-dialect-view: ## Open promptfoo web UI to view dialect evaluation results
	promptfoo view

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help