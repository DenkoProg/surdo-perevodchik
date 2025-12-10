#!/usr/bin/env python
"""
Convert JSONL corpus file to CSV while cleaning invalid records.

This script:
1. Reads a JSONL file with source_standard and generated_dialect fields
2. Filters out invalid/corrupted records
3. Outputs a clean CSV file with source (dialect), target (standard) columns

Usage:
    uv run python scripts/helpers/jsonl_to_csv.py input.jsonl output.csv
    uv run python scripts/helpers/jsonl_to_csv.py input.jsonl output.csv --verbose
"""

import csv
import json
from pathlib import Path
import re
from typing import Annotated

import typer


app = typer.Typer(
    name="jsonl-to-csv",
    help="Convert JSONL corpus to CSV with cleaning.",
    add_completion=False,
)


def is_valid_translation(source: str, translation: str) -> tuple[bool, str]:
    """
    Check if a translation record is valid.

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Empty check
    if not source or not translation:
        return False, "empty source or translation"

    # Check for JSON artifacts in translation
    json_patterns = [
        r'^\s*\{\s*"translations"',  # JSON object start
        r"^\s*\[",  # Array start
        r'^\s*"[^"]*",?\s*$',  # Just a quoted string with comma
        r"translations\s*:",  # JSON key
    ]
    for pattern in json_patterns:
        if re.search(pattern, translation):
            return False, "JSON artifact in translation"

    # Check if translation is just the source (no change)
    if source.strip() == translation.strip():
        return False, "translation identical to source"

    # Check for very short translations (likely errors)
    if len(translation.strip()) < 2:
        return False, "translation too short"

    # Check for meta-text artifacts
    if translation.startswith('"') and translation.endswith('",'):
        return False, "translation looks like JSON string element"

    # Check for common error patterns
    error_patterns = [
        r"<SEP>.*<SEP>",  # Multiple SEP tokens (batch confusion)
        r"^\d+\.\s+",  # Starts with numbering
        r"^Переклад:",  # Meta text
        r"^Відповідь:",  # Meta text
    ]
    for pattern in error_patterns:
        if re.search(pattern, translation):
            return False, f"error pattern: {pattern}"

    return True, ""


def clean_translation(translation: str) -> str:
    """Clean up minor issues in translation."""
    # Remove leading/trailing quotes if they look like JSON artifacts
    if translation.startswith('"') and (translation.endswith('",') or translation.endswith('"')):
        translation = translation.strip('"').rstrip(",")

    # Remove numbering at start
    translation = re.sub(r"^\d+[\.\)\:]\s*", "", translation)

    return translation.strip()


@app.command()
def convert(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input JSONL file path.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Argument(
            help="Output CSV file path.",
        ),
    ],
    source_field: Annotated[
        str,
        typer.Option(
            "--source-field",
            "-s",
            help="Field name for source text in JSONL.",
        ),
    ] = "source_standard",
    target_field: Annotated[
        str,
        typer.Option(
            "--target-field",
            "-t",
            help="Field name for target/translation text in JSONL.",
        ),
    ] = "generated_dialect",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show details about filtered records.",
        ),
    ] = False,
):
    """
    Convert JSONL corpus to CSV.

    Reads source_standard and generated_dialect fields from JSONL,
    filters out invalid records, and writes a clean CSV with source,target columns.
    """
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    valid_records = 0
    invalid_reasons: dict[str, int] = {}

    rows: list[tuple[str, str]] = []

    typer.echo(f"📂 Reading {input_file}")

    with open(input_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    typer.echo(f"  ⚠️  Line {line_num}: JSON parse error - {e}")
                invalid_reasons["json_parse_error"] = invalid_reasons.get("json_parse_error", 0) + 1
                continue

            source = record.get(source_field, "").strip()
            translation = record.get(target_field, "").strip()

            # Validate
            is_valid, reason = is_valid_translation(source, translation)

            if not is_valid:
                if verbose:
                    typer.echo(f"  ⚠️  Line {line_num}: {reason}")
                    typer.echo(f"      Source: {source[:80]}...")
                    typer.echo(f"      Trans:  {translation[:80]}...")
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                continue

            # Clean and add (source=dialect, target=standard Ukrainian)
            translation = clean_translation(translation)
            rows.append((translation, source))  # dialect -> standard
            valid_records += 1

    # Write CSV output (source=dialect, target=standard)
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(rows)

    # Summary
    typer.echo("\n📊 Summary:")
    typer.echo(f"   Total records:   {total_records}")
    typer.echo(f"   Valid records:   {valid_records} ({100 * valid_records / total_records:.1f}%)")
    typer.echo(f"   Filtered out:    {total_records - valid_records}")

    if invalid_reasons:
        typer.echo("\n🚫 Filter reasons:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            typer.echo(f"   {reason}: {count}")

    typer.echo(f"\n✅ Output: {output_file} ({len(rows)} rows)")


@app.command()
def stats(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input JSONL file path.",
            exists=True,
            dir_okay=False,
        ),
    ],
    source_field: Annotated[
        str,
        typer.Option("--source-field", "-s"),
    ] = "source_standard",
    target_field: Annotated[
        str,
        typer.Option("--target-field", "-t"),
    ] = "generated_dialect",
):
    """Show statistics about a JSONL corpus file without converting."""
    total = 0
    valid = 0
    invalid_reasons: dict[str, int] = {}

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                record = json.loads(line)
                source = record.get(source_field, "").strip()
                translation = record.get(target_field, "").strip()

                is_valid, reason = is_valid_translation(source, translation)
                if is_valid:
                    valid += 1
                else:
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            except json.JSONDecodeError:
                invalid_reasons["json_parse_error"] = invalid_reasons.get("json_parse_error", 0) + 1

    typer.echo(f"📊 Statistics for {input_file.name}:")
    typer.echo(f"   Total records:   {total}")
    typer.echo(f"   Valid records:   {valid} ({100 * valid / total:.1f}%)" if total > 0 else "   No records")
    typer.echo(f"   Invalid records: {total - valid}")

    if invalid_reasons:
        typer.echo("\n🚫 Invalid record reasons:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            typer.echo(f"   {reason}: {count}")


if __name__ == "__main__":
    app()
