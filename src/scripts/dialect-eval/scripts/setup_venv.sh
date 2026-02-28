#!/bin/bash
set -e

uv venv .enveval --python 3.12
source .enveval/bin/activate

uv pip install -U vllm --torch-backend=cu128
uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
uv pip install loguru typer python-dotenv pandas
