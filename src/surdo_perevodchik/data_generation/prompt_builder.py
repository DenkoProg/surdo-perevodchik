"""Generalized prompt builder for dialect generation."""

from pathlib import Path


class DialectPromptBuilder:
    """
    Generalized prompt builder for Ukrainian dialect/surzhyk translation.

    Works with any dialect by loading rules from a file that contains:
    - System instructions
    - Transformation rules
    - Few-shot examples (embedded in the rules file)
    """

    def __init__(self, rules_path: str | Path):
        """
        Initialize the prompt builder.

        Args:
            rules_path: Path to the rules file containing system prompt,
                       rules, and few-shot examples.
        """
        self.rules_path = Path(rules_path)
        self._system_prompt: str | None = None

    def _load_rules(self) -> str:
        """Load rules from file."""
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")
        return self.rules_path.read_text(encoding="utf-8")

    @property
    def system_prompt(self) -> str:
        """Get or load the system prompt (cached)."""
        if self._system_prompt is None:
            self._system_prompt = self._load_rules()
        return self._system_prompt

    def build_user_prompt(self, sentences: list[str]) -> str:
        """
        Build the user prompt with sentences to translate.

        Args:
            sentences: List of standard Ukrainian sentences to translate.

        Returns:
            Formatted user prompt.
        """
        parts = [
            "Перетвори ці речення на діалект.",
            "Поверни ТІЛЬКИ переклади, по одному на рядок, без нумерації, без оригіналу.",
            "",
        ]

        for i, sentence in enumerate(sentences, 1):
            parts.append(f"{i}. {sentence}")

        return "\n".join(parts)

    def estimate_tokens(self, sentences: list[str]) -> int:
        """
        Rough estimate of token count for the full prompt.

        Uses ~4 chars per token as a rough approximation for Ukrainian text.
        """
        system_len = len(self.system_prompt)
        user_prompt = self.build_user_prompt(sentences)
        user_len = len(user_prompt)
        return (system_len + user_len) // 4

    @property
    def dialect_name(self) -> str:
        """Extract dialect name from rules file path."""
        # e.g., "hutsul_rules_system.txt" -> "hutsul"
        stem = self.rules_path.stem
        for suffix in ["_rules_system", "_rules", "_system"]:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem
