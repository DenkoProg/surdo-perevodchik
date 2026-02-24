"""
Prepare Hutsul dialect evaluation test cases for promptfoo.

Reads data/parallel/eval.csv (columns: source, target) and writes
promptfoo-compatible test cases to test_cases/hutsul.json.

Usage:
    uv run python src/scripts/promptfoo-dialect-eval/prepare_eval_data.py
    uv run python src/scripts/promptfoo-dialect-eval/prepare_eval_data.py --limit 0  # all rows
"""

import csv
import json
from pathlib import Path

import typer


def prepare_eval_data(limit: int = typer.Option(10, help="Max rows to include (0 = all rows)")) -> None:
    """Convert eval.csv to promptfoo test cases JSON."""
    root = Path(__file__).parent.parent.parent.parent
    input_path = root / "data" / "parallel" / "eval.csv"
    output_path = Path(__file__).parent / "test_cases" / "hutsul.json"

    test_cases = []
    with open(input_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("source") or "").strip()
            target = (row.get("target") or "").strip()
            if not source or not target:
                continue
            test_cases.append({"vars": {"source": source, "reference": target}})
            if limit > 0 and len(test_cases) >= limit:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(test_cases)} test cases to {output_path}")


if __name__ == "__main__":
    typer.run(prepare_eval_data)
