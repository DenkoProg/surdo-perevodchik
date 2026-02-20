#!/usr/bin/env python
"""
Remove useless rows from the Transcarpathian dictionary CSV.

A row is considered useless when the "Transcarpathian" headword and the
"Ukrainian" translation are the same word (or differ only by stress marks /
accent diacritics), meaning the word exists unchanged in standard Ukrainian and
provides no translation value.

Examples of rows to remove:
    думка   → думка   (identical)
    ду́мка  → думка   (only the stress mark differs)
    дурний  → дурний  (identical)

Usage:
    # In-place (overwrites the file)
    uv run python src/scripts/helpers/clean_transcarpathian_dict.py

    # Custom path
    uv run python src/scripts/helpers/clean_transcarpathian_dict.py \
        --input data/dicts/transcarpathian_ukrainian_dictionary.csv

    # Preview only (no write)
    uv run python src/scripts/helpers/clean_transcarpathian_dict.py --dry-run
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import typer


app = typer.Typer(add_completion=False)

DEFAULT_PATH = Path("data/dicts/transcarpathian_ukrainian_dictionary.csv")

# Combining characters that carry only prosodic (stress/tone) information and
# should be ignored when comparing headword ↔ translation.
# U+0301 COMBINING ACUTE ACCENT  — the standard Cyrillic stress mark
# U+0300 COMBINING GRAVE ACCENT  — occasionally used for secondary stress
_STRESS_CHARS = {"\u0301", "\u0300"}


def normalize(s: str) -> str:
    """Lowercase + strip stress marks for comparison purposes."""
    return "".join(c for c in s.lower().strip() if c not in _STRESS_CHARS)


def is_same_word(tc: str, uk: str) -> bool:
    """Return True if the two strings are the same after normalization."""
    return normalize(tc) == normalize(uk)


@app.command()
def main(
    input_path: Annotated[Path, typer.Option("--input", "-i")] = DEFAULT_PATH,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report counts but don't write")] = False,
) -> None:
    """Remove rows where Transcarpathian == Ukrainian (after stripping stress marks)."""
    if not input_path.exists():
        typer.echo(f"File not found: {input_path}", err=True)
        raise typer.Exit(1)

    with open(input_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Detect column positions (header may be ["", "Transcarpathian", "Ukrainian", "uk_lemma"])
    try:
        tc_col = next(i for i, h in enumerate(header) if h.strip().lower() in ("transcarpathian", "закарпатський"))
        uk_col = next(i for i, h in enumerate(header) if h.strip().lower() in ("ukrainian", "українська"))
    except StopIteration:
        typer.echo(f"Cannot find Transcarpathian/Ukrainian columns in header: {header}", err=True)
        raise typer.Exit(1)

    kept, removed = [], []
    for row in rows:
        if len(row) <= max(tc_col, uk_col):
            kept.append(row)
            continue
        tc = row[tc_col]
        uk = row[uk_col]
        if is_same_word(tc, uk):
            removed.append(row)
        else:
            kept.append(row)

    typer.echo(f"Total rows:   {len(rows)}")
    typer.echo(f"Removed:      {len(removed)}")
    typer.echo(f"Remaining:    {len(kept)}")

    if dry_run:
        typer.echo("\n[Dry run] First 20 removed rows:")
        for row in removed[:20]:
            typer.echo(f"  {row[tc_col]!r} → {row[uk_col]!r}")
        return

    # Re-number the index column if present
    idx_col = 0 if header[0].strip() == "" else None
    with open(input_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(kept):
            if idx_col is not None:
                row = list(row)
                row[idx_col] = str(i)
            writer.writerow(row)

    typer.echo(f"\nSaved → {input_path}")


if __name__ == "__main__":
    app()
