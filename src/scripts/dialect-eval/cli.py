import os
import subprocess
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from loguru import logger
import typer


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent

load_dotenv(BASE_DIR / ".env")

app = typer.Typer()


@app.command()
def remote(
    model: Annotated[str, typer.Option(help="Model version (OpenRouter model ID)")] = "gpt-4o-mini",
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 8,
):
    """Evaluate a remote model via OpenAI-compatible API (OpenRouter)."""
    os.chdir(PROJECT_ROOT)
    cmd = [
        "lmms-eval",
        "--model", "openai_compatible",
        "--include_path", str(BASE_DIR / "tasks"),
        "--model_args", f"model_version={model}",
        "--tasks", "hutsul_translation",
        "--batch_size", str(batch_size),
        "--log_samples",
        "--output_path", str(BASE_DIR / "logs"),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.command()
def local(
    model: Annotated[str, typer.Option(help="HuggingFace model ID")] = "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0",
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 8,
):
    """Evaluate a local model via vLLM."""
    os.chdir(PROJECT_ROOT)
    cmd = [
        "lmms-eval",
        "--model", "vllm",
        "--include_path", str(BASE_DIR / "tasks"),
        "--model_args", f"model={model},dtype=bfloat16,max_model_len=4096,gpu_memory_utilization=0.75,enforce_eager=True",
        "--tasks", "hutsul_translation",
        "--batch_size", str(batch_size),
        "--log_samples",
        "--output_path", str(BASE_DIR / "logs"),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
