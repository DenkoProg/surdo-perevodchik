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

    # Full generation
    uv run python scripts/generate_corpus.py \
        --input data/raw/standard_ukrainian.csv \
        --output data/parallel/hutsul/synthetic.csv \
        --rules prompts/hutsul_rules_system.txt
"""

import csv
from pathlib import Path
import sys
from typing import Annotated, Optional

from dotenv import load_dotenv
import typer


load_dotenv()

from surdo_perevodchik.data_generation import DialectCorpusGenerator, OpenRouterConfig


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
    max_length: int = 500,
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
    max_length: int = 500,
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
        Optional[int],
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
            help="OpenRouter model ID.",
        ),
    ] = "mistralai/mistral-7b-instruct:free",
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
        Optional[Path],
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
    # Load sentences
    typer.echo(f"📂 Loading sentences from {input_file}")

    if input_file.suffix == ".csv":
        sentences = load_sentences_from_csv(input_file, text_column, limit)
    else:
        sentences = load_sentences_from_txt(input_file, limit)

    if not sentences:
        typer.echo("❌ No valid sentences found in input file", err=True)
        raise typer.Exit(1)

    typer.echo(f"✅ Loaded {len(sentences)} sentences")

    # Configure OpenRouter
    config = OpenRouterConfig(
        model=model,
        temperature=temperature,
    )

    # Initialize generator
    generator = DialectCorpusGenerator(
        rules_path=rules_file,
        output_path=output_file,
        batch_size=batch_size,
        openrouter_config=config,
        save_jsonl=not no_jsonl,
        dictionary_path=dictionary_file,
    )

    # Show config
    typer.echo(f"\n🔧 Configuration:")
    typer.echo(f"   Dialect: {generator.dialect_name}")
    typer.echo(f"   Model: {model}")
    typer.echo(f"   Batch size: {batch_size}")
    typer.echo(f"   Dictionary: {dictionary_file or 'None'}")
    typer.echo(f"   Temperature: {temperature}")
    typer.echo(f"   Output: {output_file}")
    typer.echo()

    # Generate corpus
    try:
        total = generator.generate_corpus(sentences, resume=not no_resume)
        typer.echo(f"\n🎉 Success! Generated {total} parallel pairs.")
    except KeyboardInterrupt:
        typer.echo("\n\n⚠️ Interrupted by user. Progress has been saved.")
        typer.echo("Run again to resume from where you left off.")
        raise typer.Exit(130)
    except Exception as e:
        typer.echo(f"\n❌ Error during generation: {e}", err=True)
        typer.echo("Progress has been saved. Run again to resume.", err=True)
        raise typer.Exit(1)


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
    typer.echo(f"📏 System prompt length: {len(builder.system_prompt)} chars")
    typer.echo(f"🎯 Estimated tokens: ~{len(builder.system_prompt) // 4}")


if __name__ == "__main__":
    app()
