#!/usr/bin/env python
"""
Parse the raw OCR'd Boykivian dialect dictionary into a structured CSV.

Hybrid approach: regex segmentation + LLM extraction via OpenRouter.

Usage:
    # Full run with free model
    uv run python src/scripts/parse_boiko_dict.py

    # Custom model / batch size
    uv run python src/scripts/parse_boiko_dict.py \
        --model "meta-llama/llama-3.1-8b-instruct:free" \
        --batch-size 15

    # Dry run (segmentation only, no LLM calls)
    uv run python src/scripts/parse_boiko_dict.py --dry-run

    # Resume from a previous partial run
    uv run python src/scripts/parse_boiko_dict.py --resume
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import time
from typing import Annotated

from dotenv import load_dotenv
import typer

from surdo_perevodchik.data_generation import create_llm_client


load_dotenv()

app = typer.Typer(
    name="parse-boiko-dict",
    help="Parse the raw Boykivian dictionary into structured CSV.",
    add_completion=False,
)

# --- Constants ---
DICT_PATH = Path("examples/boiko-dict.md")
OUTPUT_PATH = Path("data/dicts/boykivian_ukrainian_dictionary.csv")
PARTIAL_PATH = Path("data/dicts/boykivian_ukrainian_dictionary.partial.jsonl")
DICT_START_LINE = 1981
DICT_END_LINE = 60671

# Regex: line starts with 2+ uppercase Cyrillic (possibly with apostrophe/space/dash)
HEADWORD_RE = re.compile(r"^[А-ЯІЇЄҐ][А-ЯІЇЄҐ'ʼ \-]{1,}")
# Standalone page number between blank context
PAGE_NUM_RE = re.compile(r"^\d{1,3}$")
# Has at least one quoted definition
HAS_DEFINITION_RE = re.compile(r'["\u201c\u201e\u00ab\'][^"\u201d\u201c\u00bb\'\n]{2,}["\u201d\u201c\u00bb\']')
# Cross-reference only
CROSS_REF_ONLY_RE = re.compile(r"(?:Д|д)ив[.,\s]", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are an expert in Ukrainian dialectology. You are parsing entries from a Boykivian dialect dictionary that was OCR-scanned, so the text contains many OCR artifacts (garbled characters, digits instead of letters, broken quotes).

For each dictionary entry, extract:
- "boykivian": the dialect headword in lowercase (e.g. "башня", "баюра")
- "ukrainian": the standard Ukrainian definition/translation (e.g. "вежа", "калюжа")
- "uk_lemma": the lemmatized (dictionary form) of the Ukrainian word

CRITICAL — the "ukrainian" field must contain REAL, existing Ukrainian words:
- The OCR text is heavily corrupted. Do NOT copy garbled text as the definition.
- If the text says "авбука" but you can tell the meaning is "абетка" (alphabet), write "абетка".
- If the text says "евзялисяї" — this is OCR garbage. Try to infer the real word from context, or skip the entry.
- Every word in the "ukrainian" field must be a real Ukrainian word that exists in standard dictionaries.
- Do NOT include grammatical annotations (наз.мн., род.одн., бот., перен., церк., etc.) in the definition.
- Do NOT include location codes (/Х-в/, /Км./, etc.) or bibliographic references.

Rules:
1. If an entry has multiple numbered meanings (1. "...", 2. "..."), output a separate object for each meaning.
2. Skip entries that are only cross-references (just "Див." pointing to another word) with no definition.
3. Skip entries where the definition is unreadable OCR garbage that you cannot confidently interpret.
4. The headword is the UPPERCASE word at the start. Convert it to lowercase for the "boykivian" field.
5. Extract the core definition — the word or short phrase in quotes. Do NOT include example sentences.
6. If the definition is a multi-word phrase, keep it as-is for "ukrainian" but lemmatize each content word for "uk_lemma".

Return ONLY a JSON array of objects. No markdown, no explanation. Example:
[{"boykivian": "башня", "ukrainian": "вежа", "uk_lemma": "вежа"}, {"boykivian": "башня", "ukrainian": "башта", "uk_lemma": "башта"}]
"""


def read_dictionary_lines(path: Path) -> list[str]:
    """Read the dictionary section of the file."""
    with open(path, encoding="utf-8") as f:
        all_lines = f.readlines()
    return all_lines[DICT_START_LINE - 1 : DICT_END_LINE]


def segment_entries(lines: list[str]) -> list[dict]:
    """Split raw lines into individual dictionary entries.

    Returns list of {"headword_line": int, "raw_text": str} dicts.
    """
    # Strip page numbers (standalone 1-3 digit lines surrounded by blanks)
    cleaned: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if PAGE_NUM_RE.match(stripped.strip()):
            # Check if surrounded by blank-ish context (skip page numbers)
            prev_blank = i == 0 or lines[i - 1].strip() == ""
            next_blank = i == len(lines) - 1 or lines[i + 1].strip() == ""
            if prev_blank or next_blank:
                continue
        cleaned.append((DICT_START_LINE + i, stripped))

    entries: list[dict] = []
    current_lines: list[str] = []
    current_start: int = 0

    for line_num, text in cleaned:
        if HEADWORD_RE.match(text) and text.strip():
            # Save previous entry
            if current_lines:
                raw = "\n".join(current_lines).strip()
                if raw:
                    entries.append({"headword_line": current_start, "raw_text": raw})
            current_lines = [text]
            current_start = line_num
        else:
            # Continuation line — skip if entry hasn't started yet
            if current_lines:
                current_lines.append(text)

    # Don't forget the last entry
    if current_lines:
        raw = "\n".join(current_lines).strip()
        if raw:
            entries.append({"headword_line": current_start, "raw_text": raw})

    return entries


def filter_cross_refs(entries: list[dict]) -> tuple[list[dict], int]:
    """Remove entries that are only cross-references with no definitions."""
    kept = []
    skipped = 0
    for entry in entries:
        text = entry["raw_text"]
        has_def = HAS_DEFINITION_RE.search(text)
        is_cross_ref = CROSS_REF_ONLY_RE.search(text)
        # Keep if it has a definition, or if it's not a cross-ref
        if has_def or not is_cross_ref:
            kept.append(entry)
        else:
            skipped += 1
    return kept, skipped


def parse_llm_response(response: str | None) -> list[dict]:
    """Parse LLM JSON response with fallback for malformed output."""
    if not response:
        return []

    # Try to find JSON array in the response
    text = response.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and d.get("boykivian") and d.get("ukrainian")]
        if isinstance(data, dict) and "entries" in data:
            return [d for d in data["entries"] if isinstance(d, dict) and d.get("boykivian") and d.get("ukrainian")]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array substring
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict) and d.get("boykivian") and d.get("ukrainian")]
        except json.JSONDecodeError:
            pass

    return []


def load_partial_results(path: Path) -> tuple[list[dict], int]:
    """Load previously saved partial results. Returns (rows, last_batch_index)."""
    rows: list[dict] = []
    last_batch = -1
    if not path.exists():
        return rows, last_batch
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("_batch_index") is not None:
                last_batch = max(last_batch, record["_batch_index"])
            rows.append(record)
    return rows, last_batch


def save_partial_batch(path: Path, rows: list[dict], batch_index: int) -> None:
    """Append a batch of extracted rows to the partial JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            row_with_meta = {**row, "_batch_index": batch_index}
            f.write(json.dumps(row_with_meta, ensure_ascii=False) + "\n")


def normalize_lemma(value: str | list | None, fallback: str = "") -> str:
    """Normalize uk_lemma: join lists, strip quotes/formatting."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value) if value else fallback


# Patterns that indicate OCR garbage leaked into a definition
_GARBAGE_RE = re.compile(
    r"/[А-ЯІЇЄҐа-яіїєґ]-[а-яіїєґ].*?/"  # location codes like /Х-в Км./
    r"|^\W{3,}$"  # pure punctuation
    r"|^[«»\"\'\s]{2,}$"  # just quotes
)
_LATIN_IN_CYRILLIC_RE = re.compile(r"[a-zA-Z]")


# Domain/style prefixes from the dictionary (бот. = botanical, перен. = figurative, etc.)
# Also handles comma typos: "перен," and stray opening parens after prefix
_DOMAIN_PREFIX_RE = re.compile(
    r"^(?:бот\.|перен[.,]|церк\.|діал\.|заст\.|розм\.|зневажл\.|жарт\.|лісів\.|мед\.|Пор\.)\s*\(?",
    re.IGNORECASE,
)
# Grammatical annotations: наз.мн., род.одн., ор.одн., місц.одн., etc.
_GRAMMAR_ANNOT_RE = re.compile(
    r"^(?:наз|род|дав|зн|ор|місц)\.\s*(?:мн|одн|жін|чол|сер)\.\s*",
    re.IGNORECASE,
)
# Leading digits or numbering artifacts: "1 як" -> "як", "2. болото" -> "болото"
_LEADING_DIGIT_RE = re.compile(r"^\d+\.?\s+")
# Non-Ukrainian gibberish: definitions where most chars are not Cyrillic/spaces/punctuation
_CYRILLIC_RE = re.compile(r"[а-яіїєґА-ЯІЇЄҐ]")


def clean_rows(rows: list[dict]) -> list[dict]:
    """Clean up LLM extraction artifacts."""
    cleaned = []
    for row in rows:
        boykivian = row.get("boykivian", "").strip()
        ukrainian = row.get("ukrainian", "").strip()
        uk_lemma = normalize_lemma(row.get("uk_lemma"), ukrainian)

        # Normalize headword: must be fully lowercase, strip stray uppercase
        boykivian = boykivian.lower()

        # Strip leading/trailing quotes and formatting chars
        ukrainian = ukrainian.strip("«»\u201c\u201d\u201e\"'!.")
        uk_lemma = uk_lemma.strip("«»\u201c\u201d\u201e\"'!.")

        # Strip domain prefixes (бот., перен., церк., etc.)
        ukrainian = _DOMAIN_PREFIX_RE.sub("", ukrainian)
        uk_lemma = _DOMAIN_PREFIX_RE.sub("", uk_lemma)

        # Strip grammatical annotations (наз.мн., род.одн., etc.)
        ukrainian = _GRAMMAR_ANNOT_RE.sub("", ukrainian)
        uk_lemma = _GRAMMAR_ANNOT_RE.sub("", uk_lemma)

        # Strip leading digits/numbering ("1 як" -> "як")
        ukrainian = _LEADING_DIGIT_RE.sub("", ukrainian)
        uk_lemma = _LEADING_DIGIT_RE.sub("", uk_lemma)

        # Strip stray parentheses left after prefix removal
        ukrainian = ukrainian.strip("()")
        uk_lemma = uk_lemma.strip("()")

        # Strip trailing unbalanced quotes and quote fragments like: ..."тьху
        ukrainian = re.sub(r'\s*["\u201c\u201e][^"\u201d\u201c]*$', "", ukrainian)
        uk_lemma = re.sub(r'\s*["\u201c\u201e][^"\u201d\u201c]*$', "", uk_lemma)
        ukrainian = ukrainian.rstrip("\"'«»\u201c\u201d\u201e")
        uk_lemma = uk_lemma.rstrip("\"'«»\u201c\u201d\u201e")

        # Re-strip whitespace after all transformations
        ukrainian = ukrainian.strip()
        uk_lemma = uk_lemma.strip()

        # Skip rows where definition contains location codes (OCR leak)
        if re.search(r"/[А-ЯІЇЄҐа-яіїєґ].*?/", ukrainian):
            continue

        # Skip rows with Latin characters mixed into Cyrillic headword
        if _LATIN_IN_CYRILLIC_RE.search(boykivian):
            continue

        # Skip definitions that are mostly non-Cyrillic (OCR garbage)
        cyrillic_chars = len(_CYRILLIC_RE.findall(ukrainian))
        if len(ukrainian) > 3 and cyrillic_chars / len(ukrainian) < 0.4:
            continue

        # Skip definitions where ALL words are <=3 chars (likely OCR noise like "рор, - гер")
        words = [w for w in re.findall(r"[а-яіїєґА-ЯІЇЄҐ'ʼ]+", ukrainian) if len(w) > 1]
        if words and all(len(w) <= 3 for w in words) and len(words) >= 2:
            continue

        # Skip empty or too-short definitions
        if len(ukrainian) < 2 or len(boykivian) < 1:
            continue

        cleaned.append(
            {
                "boykivian": boykivian,
                "ukrainian": ukrainian,
                "uk_lemma": uk_lemma,
            }
        )
    return cleaned


def write_csv(rows: list[dict], path: Path) -> None:
    """Write final CSV in the target format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Boykivian", "Ukrainian", "uk_lemma"])
        for i, row in enumerate(rows):
            writer.writerow(
                [
                    i,
                    row.get("boykivian", ""),
                    row.get("ukrainian", ""),
                    row.get("uk_lemma", row.get("ukrainian", "")),
                ]
            )


def deduplicate(rows: list[dict]) -> list[dict]:
    """Remove exact duplicate (boykivian, ukrainian) pairs."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row.get("boykivian", "").lower().strip(), row.get("ukrainian", "").lower().strip())
        if key not in seen and key[0] and key[1]:
            seen.add(key)
            unique.append(row)
    return unique


@app.command()
def main(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Path to boiko-dict.md")] = DICT_PATH,
    output_path: Annotated[Path, typer.Option("--output", "-o", help="Output CSV path")] = OUTPUT_PATH,
    model: Annotated[str, typer.Option("--model", "-m", help="OpenRouter model ID")] = "google/gemma-3-27b-it",
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Entries per LLM call")] = 15,
    delay: Annotated[float, typer.Option("--delay", "-d", help="Seconds between API calls")] = 2.0,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only segment entries, no LLM calls")] = False,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from partial results")] = False,
) -> None:
    """Parse the Boykivian dictionary into a structured CSV."""

    # --- Step 1: Segment entries ---
    typer.echo(f"Reading dictionary from {input_path}...")
    lines = read_dictionary_lines(input_path)
    typer.echo(f"  Read {len(lines)} lines (lines {DICT_START_LINE}-{DICT_END_LINE})")

    entries = segment_entries(lines)
    typer.echo(f"  Segmented into {len(entries)} entries")

    # --- Step 2: Pre-filter ---
    entries, skipped = filter_cross_refs(entries)
    typer.echo(f"  Filtered out {skipped} cross-reference-only entries")
    typer.echo(f"  Remaining: {len(entries)} entries to process")

    if dry_run:
        typer.echo("\n--- Dry run: first 10 entries ---")
        for entry in entries[:10]:
            text = entry["raw_text"][:120].replace("\n", " | ")
            typer.echo(f"  L{entry['headword_line']}: {text}")
        typer.echo(f"\nTotal entries to process: {len(entries)}")
        typer.echo(f"Estimated batches: {(len(entries) + batch_size - 1) // batch_size}")
        raise typer.Exit()

    # --- Step 3: LLM batch extraction ---
    typer.echo(f"\nInitializing LLM client (model: {model})...")
    client = create_llm_client(
        provider="openrouter",
        model=model,
        max_tokens=4096,
        temperature=0.1,
        use_structured_output=False,
    )

    all_rows: list[dict] = []
    start_batch = 0

    if resume:
        all_rows, last_batch = load_partial_results(PARTIAL_PATH)
        if last_batch >= 0:
            start_batch = last_batch + 1
            # Strip metadata
            all_rows = [{k: v for k, v in r.items() if k != "_batch_index"} for r in all_rows]
            typer.echo(f"  Resumed: {len(all_rows)} rows from {last_batch + 1} batches")
    elif PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()

    batches = [entries[i : i + batch_size] for i in range(0, len(entries), batch_size)]
    total_batches = len(batches)
    typer.echo(f"  Processing {len(entries)} entries in {total_batches} batches...")

    failed_batches = 0
    for batch_idx in range(start_batch, total_batches):
        batch = batches[batch_idx]
        # Build user prompt: entries separated by ---
        user_parts = []
        for entry in batch:
            user_parts.append(entry["raw_text"])
        user_prompt = "\n---\n".join(user_parts)

        response = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        parsed = parse_llm_response(response)

        if parsed:
            all_rows.extend(parsed)
            save_partial_batch(PARTIAL_PATH, parsed, batch_idx)
        else:
            failed_batches += 1

        # Progress
        done = batch_idx + 1
        typer.echo(
            f"  [{done}/{total_batches}] +{len(parsed)} rows (total: {len(all_rows)}) {'FAIL' if not parsed else 'ok'}"
        )

        # Rate limit delay
        if batch_idx < total_batches - 1:
            time.sleep(delay)

    # --- Step 4: Post-processing ---
    typer.echo(f"\nPost-processing {len(all_rows)} raw rows...")
    cleaned_rows = clean_rows(all_rows)
    typer.echo(f"  After cleaning: {len(cleaned_rows)} rows ({len(all_rows) - len(cleaned_rows)} garbage removed)")
    unique_rows = deduplicate(cleaned_rows)
    typer.echo(f"  After dedup: {len(unique_rows)} unique entries")

    write_csv(unique_rows, output_path)
    typer.echo(f"\nDone! CSV written to {output_path}")
    typer.echo(f"  Total rows: {len(unique_rows)}")
    if failed_batches:
        typer.echo(f"  Failed batches: {failed_batches}/{total_batches}")

    # Cleanup partial file on success
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        typer.echo("  Cleaned up partial results file.")


if __name__ == "__main__":
    app()
