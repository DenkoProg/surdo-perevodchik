# Surdo Perevodchik - Ukrainian Dialect Translator

Machine translation system for translating Ukrainian dialects (Hutsul, Boikivian, Transcarpathian, Surzhyk) to Standard Literary Ukrainian. Supports two model architectures:

- **MamayLM** (Gemma-3 4B, decoder-only) — fine-tuned with QLoRA via Unsloth
- **umt5-base** (encoder-decoder) — full fine-tune baseline

## Demo

```bash
make demo
```

Starts a Gradio web interface at `http://localhost:7860`.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/DenkoProg/surdo-perevodchik
cd surdo-perevodchik
make install
```

## Workflow

### 1. Generate / prepare data

```bash
make generate-all-local   # generate synthetic corpora with local GPU (4-bit)
make prepare-data         # merge and split into train/val/test
```

### 2. Train

```bash
make train-decoder-only-multi     # MamayLM QLoRA — all dialects (recommended)
make train-encoder-decoder-multi  # umt5-base — all dialects
```

### 3. Evaluate

```bash
make evaluate-decoder-only-base   # baseline (before fine-tuning)
make evaluate-decoder-only        # fine-tuned MamayLM
make evaluate-encoder-decoder-multi
```

Run `make help` to see all available commands.
