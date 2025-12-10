"""Corpus generator for synthetic dialect parallel data."""

import csv
from datetime import datetime, timezone, UTC
import json
from pathlib import Path
import re

from tqdm import tqdm

from surdo_perevodchik.data_generation.openrouter_client import OpenRouterClient, OpenRouterConfig
from surdo_perevodchik.data_generation.prompt_builder import DialectPromptBuilder


class DialectCorpusGenerator:
    """Generate synthetic dialect parallel corpus using LLM."""

    def __init__(
        self,
        rules_path: str | Path,
        output_path: str | Path,
        batch_size: int = 10,
        openrouter_config: OpenRouterConfig | None = None,
        save_jsonl: bool = True,
        dictionary_path: str | Path | None = None,
    ):
        """
        Initialize the corpus generator.

        Args:
            rules_path: Path to the dialect rules file.
            output_path: Path to save the output CSV.
            batch_size: Number of sentences per API call.
            openrouter_config: OpenRouter client configuration.
            save_jsonl: Also save detailed JSONL with provenance.
            dictionary_path: Optional path to CSV dictionary file.
        """
        self.prompt_builder = DialectPromptBuilder(rules_path, dictionary_path=dictionary_path)
        self.client = OpenRouterClient(openrouter_config)
        self.config = openrouter_config or OpenRouterConfig()
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.save_jsonl = save_jsonl
        self.dialect_name = self.prompt_builder.dialect_name
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_path.with_suffix(".jsonl")

    def parse_response(self, response: str | None, expected_count: int) -> list[str]:
        """
        Parse LLM response into individual translations.

        Handles both structured JSON output and text format.

        Args:
            response: Raw LLM response (JSON or text).
            expected_count: Expected number of translations.

        Returns:
            List of translations (may contain empty strings for failures).
        """
        if not response:
            return [""] * expected_count

        # Try JSON parsing first (structured output)
        translations = self._parse_json_response(response, expected_count)
        if translations is not None:
            return translations

        # Fall back to text parsing for non-structured responses
        return self._parse_text_response(response, expected_count)

    def _parse_json_response(self, response: str, expected_count: int) -> list[str] | None:
        """
        Parse structured JSON response.

        Args:
            response: Raw response that might be JSON.
            expected_count: Expected number of translations.

        Returns:
            List of translations if valid JSON, None otherwise.
        """
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict) and "translations" in data:
                translations = data["translations"]
                if isinstance(translations, list):
                    result = [str(t) if t else "" for t in translations]
                    while len(result) < expected_count:
                        result.append("")
                    return result[:expected_count]
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _parse_text_response(self, response: str, expected_count: int) -> list[str]:
        """
        Parse legacy text-based LLM response.

        Args:
            response: Raw text response.
            expected_count: Expected number of translations.

        Returns:
            List of translations (may contain empty strings for failures).
        """
        lines = response.strip().split("\n")
        translations = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("<s>") or "[OUT]" in line:
                continue

            if line.startswith("Відповідь") or line.startswith("Переклад"):
                continue

            if "→" in line:
                parts = line.split("→")
                if len(parts) >= 2:
                    line = parts[-1].strip()
                else:
                    continue

            line = re.sub(r"^\d+[\.\)\:]\s*", "", line)

            if not line:
                continue

            translations.append(line)

        while len(translations) < expected_count:
            translations.append("")

        return translations[:expected_count]

    def _get_processed_count(self) -> int:
        """Count already processed pairs from output file."""
        if not self.output_path.exists():
            return 0

        with open(self.output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)

    def _init_csv(self):
        """Initialize CSV file with header if it doesn't exist."""
        if not self.output_path.exists():
            with open(self.output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["source", "target"])
                writer.writeheader()

    def _append_csv(self, pairs: list[tuple[str, str]]):
        """Append pairs to CSV file."""
        with open(self.output_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "target"])
            for dialect, standard in pairs:
                if dialect:  # Only write if we got a translation
                    writer.writerow({"source": dialect, "target": standard})

    def _append_jsonl(self, records: list[dict]):
        """Append detailed records to JSONL file."""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_corpus(
        self,
        standard_sentences: list[str],
        resume: bool = True,
    ) -> int:
        """
        Generate parallel corpus from standard Ukrainian sentences.

        Args:
            standard_sentences: List of standard Ukrainian sentences.
            resume: Whether to resume from existing progress.

        Returns:
            Total number of pairs generated.
        """
        processed_count = 0
        if resume:
            processed_count = self._get_processed_count()
            if processed_count > 0:
                print(f"Resuming from {processed_count} existing pairs")

        self._init_csv()

        remaining_sentences = standard_sentences[processed_count:]
        if not remaining_sentences:
            print("All sentences already processed!")
            return processed_count

        print(f"Processing {len(remaining_sentences)} sentences in batches of {self.batch_size}")

        total_generated = processed_count

        progress = tqdm(
            range(0, len(remaining_sentences), self.batch_size),
            desc="Generating",
            unit="batch",
        )

        for i in progress:
            batch = remaining_sentences[i : i + self.batch_size]

            system_prompt = self.prompt_builder.build_system_prompt(batch)
            user_prompt = self.prompt_builder.build_user_prompt(batch)

            response = self.client.generate(system_prompt, user_prompt, num_sentences=len(batch))
            translations = self.parse_response(response, len(batch))

            pairs = []
            jsonl_records = []
            timestamp = datetime.now(UTC).isoformat()

            for standard, dialect in zip(batch, translations, strict=False):
                if dialect:
                    pairs.append((dialect, standard))

                    if self.save_jsonl:
                        jsonl_records.append(
                            {
                                "source_standard": standard,
                                "generated_dialect": dialect,
                                "dialect": self.dialect_name,
                                "method": "llm_generated",
                                "provenance": {
                                    "model": self.config.model,
                                    "rules_file": str(self.prompt_builder.rules_path),
                                    "timestamp": timestamp,
                                },
                            }
                        )

            if pairs:
                self._append_csv(pairs)
                total_generated += len(pairs)

            if jsonl_records and self.save_jsonl:
                self._append_jsonl(jsonl_records)

            progress.set_postfix({"total": total_generated, "batch_ok": len(pairs)})

        print(f"\nGeneration complete! Total pairs: {total_generated}")
        print(f"Output CSV: {self.output_path}")
        if self.save_jsonl:
            print(f"Output JSONL: {self.jsonl_path}")

        return total_generated
