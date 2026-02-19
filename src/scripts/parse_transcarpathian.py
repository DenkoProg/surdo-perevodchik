#!/usr/bin/env python
"""
Parse the Transcarpathian dialect PDF into structured CSV files.

This is a scanned PDF (no text layer), so we use a vision-capable LLM via
OpenRouter to OCR each page and extract structured content.

Extracts:
  1. Dictionary entries → data/dicts/transcarpathian_ukrainian_dictionary.csv
  2. Dialect texts     → data/raw/transcarpathian.csv

PDF structure (0-indexed pages):
  Pages   0-429  : Dictionary (Словник закарпатської говірки)
  Pages 430-480  : Texts appendix (dialect narratives)

Usage:
    # Extract dictionary only
    uv run python src/scripts/parse_transcarpathian.py --mode dict

    # Extract texts only
    uv run python src/scripts/parse_transcarpathian.py --mode texts

    # Both (default)
    uv run python src/scripts/parse_transcarpathian.py

    # Dry run (renders pages, skips API calls — useful for sanity check)
    uv run python src/scripts/parse_transcarpathian.py --mode dict --dry-run

    # Resume a previous partial run
    uv run python src/scripts/parse_transcarpathian.py --mode dict --resume

    # Use a different vision model
    uv run python src/scripts/parse_transcarpathian.py --model "google/gemini-2.0-flash-exp:free"
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
import re
import time
from typing import Annotated

from dotenv import load_dotenv
import fitz  # pymupdf
import requests
import typer


load_dotenv()

app = typer.Typer(
    name="parse-transcarpathian",
    help="Parse the Transcarpathian dialect PDF into dictionary and texts CSVs.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PDF_PATH = Path("examples/transcarpathian-text-and-dict.pdf")
DICT_OUTPUT = Path("data/dicts/transcarpathian_ukrainian_dictionary.csv")
TEXTS_OUTPUT = Path("data/raw/transcarpathian.csv")
DICT_PARTIAL = Path("data/dicts/transcarpathian_dict.partial.jsonl")
TEXTS_PARTIAL = Path("data/raw/transcarpathian_texts.partial.jsonl")

# 0-indexed page ranges
DICT_START_PAGE = 16  # book page 17 — first actual dictionary entries (А)
DICT_END_PAGE = 429  # book page 430 — last dictionary page (inclusive)
TEXTS_START_PAGE = 430  # book page 431 — "ДОДАТОК / ТЕКСТИ..."
TEXTS_END_PAGE = 480  # book page 481 — last page (inclusive)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

DICT_SYSTEM_PROMPT = """\
You are extracting entries from a scanned two-column Transcarpathian Ukrainian \
dialect dictionary page (Словник закарпатської говірки).

For EVERY dictionary entry visible on the page (both columns), extract:
- "transcarpathian": the headword (appears in BOLD ALL-CAPS in the scan), \
converted to lowercase
- "ukrainian": the standard Ukrainian definition or translation — just the \
core word/phrase, NOT example sentences or usage examples
- "uk_lemma": the lemma (dictionary/base form) of the Ukrainian word

Rules:
1. Process BOTH left and right columns of the two-column layout.
2. Each entry starts with a bold uppercase headword (e.g., ХЫЖА, ФАЙНИЙ, ÔСІНЬ).
3. Only extract the core definition. Do NOT include example sentences, \
   abbreviations like "м.", "ж.", "дієсл.", etc.
4. If an entry has multiple numbered meanings (1., 2., ...), output a \
   separate JSON object for each meaning.
5. Skip entries that are purely cross-references ("Пор.", "Div.") with no \
   own definition.
6. Preserve special characters exactly in the headword: ô, ÿ, ʼ, ы, і, ї, є, \
   stress marks.
7. The "ukrainian" field must contain real standard Ukrainian words only.
8. Skip page headers (running titles) and page numbers.

Return ONLY a valid JSON array. No markdown fences, no explanation.
Example:
[{"transcarpathian": "хыжа", "ukrainian": "хата", "uk_lemma": "хата"}, \
{"transcarpathian": "файний", "ukrainian": "гарний", "uk_lemma": "гарний"}]
"""

TEXTS_SYSTEM_PROMPT = """\
You are extracting Transcarpathian Ukrainian dialect text from a scanned book page.

The text is phonetically transcribed dialect speech. Key notation:
  //  phrase/sentence boundary
  /   minor pause within a phrase
  ô   closed/rounded [o]
  ÿ   labialised [i]
  ʼ   soft (palatalised) consonant marker
  ŷ   non-syllabic у
  ı̈   non-syllabic і

Instructions:
1. Extract ALL running dialect text from the page, preserving it EXACTLY as \
   printed (including ô, ÿ, ʼ, /, //).
2. If a new section title appears (centered bold text, e.g., \
   "Про сплавляння лісу плотами"), include it in the "title" field.
3. Include recording attribution lines at the bottom \
   (e.g., "Записано 2005 р. від ...") — put them in the "attribution" field.
4. Return ONLY the raw running text in "text", without titles or attributions.

Return a JSON object with this exact structure:
{
  "title": "section title if a new section begins on this page, else null",
  "text": "all running dialect text from this page",
  "attribution": "recording info line if present, else null"
}
"""

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def page_to_base64(doc: fitz.Document, page_idx: int, dpi: int = 200) -> str:
    """Render a PDF page to a base64-encoded PNG string."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[page_idx].get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()


def call_vision_llm(
    image_b64: str,
    system_prompt: str,
    user_text: str,
    api_key: str,
    model: str,
    max_retries: int = 5,
    retry_delay: float = 5.0,
    timeout: int = 120,
) -> str | None:
    """Call a vision-capable LLM via OpenRouter with a base64-encoded page image."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/DenkoProg/surdo-perevodchik",
        "X-Title": "Surdo Perevodchik",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                wait = retry_delay * (2**attempt)
                typer.echo(f"    Rate limited (429). Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = retry_delay * (attempt + 1)
                typer.echo(f"    Server error {resp.status_code}. Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            typer.echo(f"    Unexpected response: {data}")
            return None
        except requests.exceptions.Timeout:
            typer.echo(f"    Timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            typer.echo(f"    Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    typer.echo("    All retries exhausted.")
    return None


def strip_json_fences(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Dictionary extraction helpers
# ---------------------------------------------------------------------------


def parse_dict_response(response: str | None) -> list[dict]:
    """Parse LLM JSON response for dictionary entries."""
    if not response:
        return []
    text = strip_json_fences(response)

    def _valid(d: dict) -> bool:
        return isinstance(d, dict) and bool(d.get("transcarpathian")) and bool(d.get("ukrainian"))

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if _valid(d)]
        if isinstance(data, dict) and "entries" in data:
            return [d for d in data["entries"] if _valid(d)]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [d for d in data if _valid(d)]
        except json.JSONDecodeError:
            pass

    return []


_CYRILLIC_RE = re.compile(r"[а-яіїєґА-ЯІЇЄҐ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def clean_dict_rows(rows: list[dict]) -> list[dict]:
    """Light post-processing of LLM-extracted dictionary rows."""
    cleaned = []
    for row in rows:
        tc = row.get("transcarpathian", "").strip().lower()
        uk = row.get("ukrainian", "").strip()
        lemma = row.get("uk_lemma", uk).strip()

        if isinstance(lemma, list):
            lemma = ", ".join(str(v) for v in lemma)

        # Strip quotes and punctuation artifacts
        uk = uk.strip("«»\u201c\u201d\u201e\"'!.,")
        lemma = lemma.strip("«»\u201c\u201d\u201e\"'!.,")

        # Skip headwords with Latin characters (OCR noise)
        if _LATIN_RE.search(tc):
            continue

        # Skip definitions that are mostly non-Cyrillic
        if uk and len(uk) > 3:
            cyr = len(_CYRILLIC_RE.findall(uk))
            if cyr / len(uk) < 0.4:
                continue

        # Skip too-short entries
        if len(tc) < 1 or len(uk) < 2:
            continue

        cleaned.append({"transcarpathian": tc, "ukrainian": uk, "uk_lemma": lemma or uk})
    return cleaned


def deduplicate_dict(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in rows:
        key = (row.get("transcarpathian", "").lower().strip(), row.get("ukrainian", "").lower().strip())
        if key not in seen and key[0] and key[1]:
            seen.add(key)
            unique.append(row)
    return unique


def write_dict_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "Transcarpathian", "Ukrainian", "uk_lemma"])
        for i, row in enumerate(rows):
            writer.writerow([i, row["transcarpathian"], row["ukrainian"], row.get("uk_lemma", row["ukrainian"])])


# ---------------------------------------------------------------------------
# Texts extraction helpers
# ---------------------------------------------------------------------------


def parse_text_response(response: str | None) -> dict | None:
    """Parse LLM JSON response for a text page."""
    if not response:
        return None
    text = strip_json_fences(response)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "text" in data:
            return data
    except json.JSONDecodeError:
        pass
    # Fallback: treat the whole response as raw text
    return {"text": response, "title": None, "attribution": None}


def split_into_phrases(full_text: str) -> list[str]:
    """Split concatenated dialect text on // boundaries into phrases."""
    phrases = re.split(r"\s*//\s*", full_text)
    result = []
    for phrase in phrases:
        # Remove minor-pause markers (/), normalise whitespace
        phrase = re.sub(r"\s*/\s*", " ", phrase).strip()
        # Skip attribution lines, very short fragments, and blank lines
        if len(phrase) > 15 and not re.match(r"Записано\s+\d{4}", phrase) and not re.match(r"^\d+$", phrase):
            result.append(phrase)
    return result


def write_texts_csv(phrases: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for phrase in phrases:
            writer.writerow([phrase])


# ---------------------------------------------------------------------------
# Partial result persistence (resume support)
# ---------------------------------------------------------------------------


def load_partial(path: Path) -> tuple[list[dict], int]:
    """Load previously saved partial results. Returns (rows, last_page_index)."""
    rows, last_page = [], -1
    if not path.exists():
        return rows, last_page
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_page" in record:
                last_page = max(last_page, record["_page"])
            rows.append(record)
    return rows, last_page


def save_partial_rows(path: Path, rows: list[dict], page_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({**row, "_page": page_idx}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------


def extract_dict(
    doc: fitz.Document, api_key: str, model: str, delay: float, dry_run: bool, resume: bool, dpi: int
) -> None:
    typer.echo(f"\n=== Dictionary extraction (book pages {DICT_START_PAGE + 1}–{DICT_END_PAGE + 1}) ===")

    all_rows: list[dict] = []
    start_page = DICT_START_PAGE

    if resume:
        raw, last_page = load_partial(DICT_PARTIAL)
        if last_page >= DICT_START_PAGE:
            start_page = last_page + 1
            all_rows = [{k: v for k, v in r.items() if k != "_page"} for r in raw]
            typer.echo(f"  Resumed from page {last_page + 2} ({len(all_rows)} rows so far)")
    elif DICT_PARTIAL.exists():
        DICT_PARTIAL.unlink()

    page_range = range(start_page, DICT_END_PAGE + 1)
    typer.echo(f"  Pages to process: {len(page_range)}")

    if dry_run:
        typer.echo("  [Dry run] Rendering first 3 pages to check quality...")
        for idx in list(page_range)[:3]:
            b64 = page_to_base64(doc, idx, dpi)
            typer.echo(f"  Page {idx + 1}: {len(b64) // 1024} KB base64 PNG")
        typer.echo("  Would send each page to vision LLM for structured extraction.")
        return

    failed_pages = 0
    for i, page_idx in enumerate(page_range):
        b64 = page_to_base64(doc, page_idx, dpi)
        response = call_vision_llm(
            b64,
            DICT_SYSTEM_PROMPT,
            "Extract all dictionary entries from this page as a JSON array.",
            api_key,
            model,
        )
        parsed = parse_dict_response(response)
        if parsed:
            all_rows.extend(parsed)
            save_partial_rows(DICT_PARTIAL, parsed, page_idx)
        else:
            failed_pages += 1

        status = "FAIL" if not parsed else "ok"
        typer.echo(
            f"  [{i + 1}/{len(page_range)}] p{page_idx + 1}: +{len(parsed)} entries (total: {len(all_rows)}) {status}"
        )

        if i < len(page_range) - 1:
            time.sleep(delay)

    # Post-process
    typer.echo(f"\nPost-processing {len(all_rows)} raw rows...")
    cleaned = clean_dict_rows(all_rows)
    typer.echo(f"  After cleaning:  {len(cleaned)} rows ({len(all_rows) - len(cleaned)} removed)")
    unique = deduplicate_dict(cleaned)
    typer.echo(f"  After dedup:     {len(unique)} unique entries")

    write_dict_csv(unique, DICT_OUTPUT)
    typer.echo(f"  Saved → {DICT_OUTPUT}")

    if failed_pages:
        typer.echo(f"  Failed pages: {failed_pages}/{len(page_range)}")

    if DICT_PARTIAL.exists():
        DICT_PARTIAL.unlink()
        typer.echo("  Cleaned up partial file.")


def extract_texts(
    doc: fitz.Document, api_key: str, model: str, delay: float, dry_run: bool, resume: bool, dpi: int
) -> None:
    typer.echo(f"\n=== Texts extraction (book pages {TEXTS_START_PAGE + 1}–{TEXTS_END_PAGE + 1}) ===")

    all_pages: list[dict] = []
    start_page = TEXTS_START_PAGE

    if resume:
        raw, last_page = load_partial(TEXTS_PARTIAL)
        if last_page >= TEXTS_START_PAGE:
            start_page = last_page + 1
            all_pages = [{k: v for k, v in r.items() if k != "_page"} for r in raw]
            typer.echo(f"  Resumed from page {last_page + 2} ({len(all_pages)} pages so far)")
    elif TEXTS_PARTIAL.exists():
        TEXTS_PARTIAL.unlink()

    page_range = range(start_page, TEXTS_END_PAGE + 1)
    typer.echo(f"  Pages to process: {len(page_range)}")

    if dry_run:
        typer.echo("  [Dry run] Would extract dialect texts from text pages.")
        return

    for i, page_idx in enumerate(page_range):
        b64 = page_to_base64(doc, page_idx, dpi)
        response = call_vision_llm(
            b64,
            TEXTS_SYSTEM_PROMPT,
            "Extract the dialect text from this page as the specified JSON object.",
            api_key,
            model,
        )
        parsed = parse_text_response(response)
        if parsed and parsed.get("text"):
            all_pages.append(parsed)
            save_partial_rows(TEXTS_PARTIAL, [parsed], page_idx)

        title_tag = f" [{parsed['title']}]" if parsed and parsed.get("title") else ""
        chars = len(parsed.get("text", "")) if parsed else 0
        typer.echo(f"  [{i + 1}/{len(page_range)}] p{page_idx + 1}{title_tag}: {chars} chars")

        if i < len(page_range) - 1:
            time.sleep(delay)

    # Concatenate all pages and split into phrases
    full_text = " // ".join(p["text"] for p in all_pages if p.get("text"))
    phrases = split_into_phrases(full_text)
    typer.echo(f"\nExtracted {len(phrases)} dialect phrases")

    write_texts_csv(phrases, TEXTS_OUTPUT)
    typer.echo(f"  Saved → {TEXTS_OUTPUT}")

    if TEXTS_PARTIAL.exists():
        TEXTS_PARTIAL.unlink()
        typer.echo("  Cleaned up partial file.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    mode: Annotated[str, typer.Option("--mode", "-m", help="What to extract: dict | texts | both")] = "both",
    pdf_path: Annotated[Path, typer.Option("--pdf", help="Path to the Transcarpathian PDF")] = PDF_PATH,
    model: Annotated[
        str, typer.Option("--model", help="Vision-capable OpenRouter model ID")
    ] = "nvidia/nemotron-nano-12b-v2-vl:free",
    delay: Annotated[float, typer.Option("--delay", "-d", help="Seconds between API calls")] = 2.0,
    dpi: Annotated[int, typer.Option("--dpi", help="DPI for rendering PDF pages to images")] = 200,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Render pages only, skip LLM calls")] = False,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from a previous partial run")] = False,
) -> None:
    """Parse the Transcarpathian dialect PDF → dictionary CSV + texts CSV."""
    import os

    if mode not in ("dict", "texts", "both"):
        typer.echo(f"Error: --mode must be dict, texts, or both (got '{mode}')", err=True)
        raise typer.Exit(1)

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key and not dry_run:
        typer.echo("Error: OPENROUTER_API_KEY environment variable not set.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Opening PDF: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    typer.echo(f"  Total pages: {len(doc)}")
    typer.echo(f"  Model: {model}")
    typer.echo(f"  DPI:   {dpi}")

    if mode in ("dict", "both"):
        extract_dict(doc, api_key, model, delay, dry_run, resume, dpi)

    if mode in ("texts", "both"):
        extract_texts(doc, api_key, model, delay, dry_run, resume, dpi)

    typer.echo("\nAll done.")


if __name__ == "__main__":
    app()
