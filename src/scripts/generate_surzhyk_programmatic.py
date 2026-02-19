#!/usr/bin/env python
"""
Generate a programmatic Surzhyk parallel corpus using phrase substitution.

Adapted from diploma/unlp/errorification/SurzhErrorifier. No LLM required —
replaces standard Ukrainian phrases with Russian-calqued Surzhyk variants
using the surzhyk dictionary.

Usage:
    uv run python src/scripts/generate_surzhyk_programmatic.py \\
        --input data/raw/standard_ukrainian.csv \\
        --dictionary data/dicts/surzhyk_ukrainian_dictionary.csv \\
        --output data/parallel/surzhyk/programmatic_surzhyk_corpus.csv \\
        --limit 30000
"""

import csv
from pathlib import Path
from typing import Annotated, Optional

from tqdm import tqdm
import typer

from surdo_perevodchik.data_generation.surzhyk_errorifier import SurzhykErrorifier


app = typer.Typer(
    name="generate-surzhyk-programmatic",
    help="Generate Surzhyk parallel corpus using phrase substitution (no LLM).",
    add_completion=False,
)


def _load_sentences(
    input_path: Path,
    text_column: str,
    limit: int | None,
    min_length: int,
    max_length: int,
) -> list[str]:
    """Load sentences from a CSV file."""
    sentences: list[str] = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column, "").strip()
            if text and min_length <= len(text) <= max_length:
                sentences.append(text)
                if limit and len(sentences) >= limit:
                    break
    return sentences


def _get_processed_count(output_path: Path) -> int:
    """Count already-written rows for resume support."""
    if not output_path.exists():
        return 0
    with open(output_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _init_csv(output_path: Path) -> None:
    """Create output CSV with header if it doesn't exist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "target"])
            writer.writeheader()


def _append_csv(output_path: Path, pairs: list[tuple[str, str]]) -> None:
    """Append (surzhyk, standard) pairs to CSV."""
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        for surzhyk, standard in pairs:
            writer.writerow({"source": surzhyk, "target": standard})


@app.command()
def generate(
    input_file: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input CSV with standard Ukrainian sentences.",
            exists=True,
            dir_okay=False,
        ),
    ],
    dictionary_file: Annotated[
        Path,
        typer.Option(
            "--dictionary",
            "-d",
            help="Path to surzhyk dictionary CSV (Surzhyk,Ukrainian columns).",
            exists=True,
            dir_okay=False,
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output CSV path for the parallel corpus.",
        ),
    ],
    text_column: Annotated[
        str,
        typer.Option("--text-column", help="Column name for text in input CSV."),
    ] = "text",
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-n", help="Max source sentences to read."),
    ] = None,
    min_length: Annotated[
        int,
        typer.Option("--min-length", help="Minimum sentence length in characters."),
    ] = 15,
    max_length: Annotated[
        int,
        typer.Option("--max-length", help="Maximum sentence length in characters."),
    ] = 500,
    min_substitutions: Annotated[
        int,
        typer.Option(
            "--min-subs",
            help="Minimum substitutions required to emit a pair.",
        ),
    ] = 1,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip already-written rows."),
    ] = True,
) -> None:
    """Generate Surzhyk parallel corpus using rule-based phrase substitution."""
    typer.echo(f"Loading dictionary from {dictionary_file}")
    errorifier = SurzhykErrorifier(dictionary_file)
    typer.echo(f"  Loaded {len(errorifier.phrase_pairs) + len(errorifier.single_pairs)} substitution pairs")

    typer.echo(f"Loading sentences from {input_file}")
    sentences = _load_sentences(input_file, text_column, limit, min_length, max_length)
    typer.echo(f"  Found {len(sentences)} sentences")

    processed_count = 0
    if resume:
        processed_count = _get_processed_count(output_file)
        if processed_count > 0:
            typer.echo(
                f"  Resuming from {processed_count} existing pairs — skipping first {processed_count} sentences"
            )

    _init_csv(output_file)

    remaining = sentences[processed_count:]
    if not remaining:
        typer.echo("All sentences already processed!")
        raise typer.Exit()

    typer.echo(f"Processing {len(remaining)} sentences...")
    total_generated = processed_count
    total_skipped = 0

    for sentence in tqdm(remaining, desc="Surzhykifying", unit="sent"):
        surzhyk_sentence, n_subs = errorifier.surzhykify(sentence)
        if n_subs >= min_substitutions:
            _append_csv(output_file, [(surzhyk_sentence, sentence)])
            total_generated += 1
        else:
            total_skipped += 1

    typer.echo(f"\nDone!")
    typer.echo(f"  Pairs generated : {total_generated}")
    typer.echo(f"  Sentences skipped (no match): {total_skipped}")
    typer.echo(f"  Match rate: {total_generated / (total_generated + total_skipped):.1%}")
    typer.echo(f"  Output: {output_file}")


if __name__ == "__main__":
    app()
