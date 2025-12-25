#!/usr/bin/env python
"""
Generate synthetic dialect parallel corpus using OpenRouter LLM.

Supports any Ukrainian dialect (Hutsul, Surzhyk, etc.) via rules files.

Usage:
    # API key is loaded from .env file automatically

    # Generate Hutsul corpus (test with 20 sentences)
    uv run python scripts/generate_corpus.py \
        --input data/raw/standard_ukrainian.csv \
        --output data/parallel/hutsul/synthetic.csv \
        --rules prompts/hutsul_rules_system.txt \
        --limit 20
"""

import csv
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
import typer

from surdo_perevodchik.data_generation import DialectCorpusGenerator, create_llm_client


load_dotenv()


app = typer.Typer(
    name="generate-corpus",
    help="Generate synthetic dialect parallel corpus using LLM.",
    add_completion=False,
)


def load_sentences_from_csv(
    input_path: Path,
    text_column: str,
    limit: int | None = None,
    min_length: int = 15,
    max_length: int = 1000,
) -> list[str]:
    """Load sentences from CSV file."""
    sentences = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column, "").strip()
            if text and min_length <= len(text) <= max_length:
                sentences.append(text)
                if limit and len(sentences) >= limit:
                    break
    return sentences


def load_sentences_from_txt(
    input_path: Path,
    limit: int | None = None,
    min_length: int = 15,
    max_length: int = 1000,
) -> list[str]:
    """Load sentences from text file (one per line)."""
    sentences = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and min_length <= len(text) <= max_length:
                sentences.append(text)
                if limit and len(sentences) >= limit:
                    break
    return sentences


@app.command()
def generate(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input file with standard Ukrainian sentences (CSV or TXT).",
            exists=True,
            dir_okay=False,
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output CSV file path for parallel corpus.",
        ),
    ],
    rules_file: Annotated[
        Path,
        typer.Option(
            "--rules",
            "-r",
            help="Path to dialect rules file (e.g., hutsul_rules_system.txt).",
            exists=True,
            dir_okay=False,
        ),
    ],
    text_column: Annotated[
        str,
        typer.Option(
            "--text-column",
            help="Column name for text in CSV input.",
        ),
    ] = "text",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            help="Max number of sentences to process.",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Number of sentences per API call.",
        ),
    ] = 10,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            "-t",
            help="LLM temperature (0.0-1.0).",
        ),
    ] = 0.7,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model ID (OpenRouter or HuggingFace path).",
        ),
    ] = "mistralai/mistral-7b-instruct:free",
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="LLM provider: 'openrouter' (API) or 'local' (GPU).",
        ),
    ] = "openrouter",
    load_in_8bit: Annotated[
        bool,
        typer.Option(
            "--load-in-8bit",
            help="Use 8-bit quantization for local models (faster, less memory).",
        ),
    ] = True,
    load_in_4bit: Annotated[
        bool,
        typer.Option(
            "--load-in-4bit",
            help="Use 4-bit quantization for local models (even less memory).",
        ),
    ] = False,
    no_resume: Annotated[
        bool,
        typer.Option(
            "--no-resume",
            help="Start fresh instead of resuming from existing output.",
        ),
    ] = False,
    no_jsonl: Annotated[
        bool,
        typer.Option(
            "--no-jsonl",
            help="Don't save detailed JSONL provenance file.",
        ),
    ] = False,
    dictionary_file: Annotated[
        Path | None,
        typer.Option(
            "--dictionary",
            "-d",
            help="Path to CSV dictionary file with Hutsul/Ukrainian columns.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Generate synthetic dialect parallel corpus from standard Ukrainian sentences.

    The rules file should contain:
    - System instructions for the LLM
    - Transformation rules for the dialect
    - Few-shot examples of translations

    Output CSV format: source (dialect), target (standard)
    """
    typer.echo(f"📂 Loading sentences from {input_file}")

    if input_file.suffix == ".csv":
        sentences = load_sentences_from_csv(input_file, text_column, limit)
    else:
        sentences = load_sentences_from_txt(input_file, limit)

    if not sentences:
        typer.echo("❌ No valid sentences found in input file", err=True)
        raise typer.Exit(1)

    typer.echo(f"✅ Loaded {len(sentences)} sentences")

    # Create LLM client based on provider
    typer.echo(f"\n🔧 Initializing {provider} client...")
    client = create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
        load_in_8bit=load_in_8bit if provider == "local" else False,
        load_in_4bit=load_in_4bit if provider == "local" else False,
    )
    typer.echo(f"✅ Client ready: {client.name}")

    generator = DialectCorpusGenerator(
        rules_path=rules_file,
        output_path=output_file,
        batch_size=batch_size,
        llm_client=client,
        save_jsonl=not no_jsonl,
        dictionary_path=dictionary_file,
    )

    typer.echo("\n� Configuration:")
    typer.echo(f"   Dialect: {generator.dialect_name}")
    typer.echo(f"   Provider: {provider}")
    typer.echo(f"   Model: {model}")
    typer.echo(f"   Batch size: {batch_size}")
    typer.echo(f"   Dictionary: {dictionary_file or 'None'}")
    typer.echo(f"   Temperature: {temperature}")
    if provider == "local":
        quant = "4-bit" if load_in_4bit else "8-bit" if load_in_8bit else "fp16"
        typer.echo(f"   Quantization: {quant}")
    typer.echo(f"   Output: {output_file}")
    typer.echo()

    try:
        total = generator.generate_corpus(sentences, resume=not no_resume)
        typer.echo(f"\n🎉 Success! Generated {total} parallel pairs.")
    except KeyboardInterrupt:
        typer.echo("\n\n⚠️ Interrupted by user. Progress has been saved.")
        typer.echo("Run again to resume from where you left off.")
        raise typer.Exit(130)  # noqa: B904
    except Exception as e:
        typer.echo(f"\n❌ Error during generation: {e}", err=True)
        typer.echo("Progress has been saved. Run again to resume.", err=True)
        raise typer.Exit(1)  # noqa: B904


@app.command()
def info(
    rules_file: Annotated[
        Path,
        typer.Argument(
            help="Path to dialect rules file.",
            exists=True,
            dir_okay=False,
        ),
    ],
):
    """Show information about a rules file."""
    from surdo_perevodchik.data_generation import DialectPromptBuilder

    builder = DialectPromptBuilder(rules_file)

    typer.echo(f"📄 Rules file: {rules_file}")
    typer.echo(f"🏷️  Dialect: {builder.dialect_name}")


if __name__ == "__main__":
    app()
