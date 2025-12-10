"""Generalized prompt builder for dialect generation."""

from collections import Counter
import csv
from pathlib import Path
import re


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for Ukrainian text."""
    return re.findall(r"\b\w+\b", text.lower())


def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors (dicts)."""
    common_keys = set(vec1.keys()) & set(vec2.keys())
    if not common_keys:
        return 0.0

    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
    norm1 = sum(v**2 for v in vec1.values()) ** 0.5
    norm2 = sum(v**2 for v in vec2.values()) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class DictionaryMatcher:
    """Finds relevant dictionary entries using TF-IDF cosine similarity."""

    def __init__(self, dictionary_entries: list[tuple[str, str]]):
        """
        Initialize with dictionary entries.

        Args:
            dictionary_entries: List of (ukrainian, hutsul) tuples.
        """
        self.entries = dictionary_entries
        self.entry_vectors: list[dict[str, float]] = []
        self._build_vectors()

    def _build_vectors(self):
        """Build TF vectors for each dictionary entry."""
        for ukrainian, _ in self.entries:
            tokens = _tokenize(ukrainian)
            tf = Counter(tokens)
            max_freq = max(tf.values()) if tf else 1
            self.entry_vectors.append({k: v / max_freq for k, v in tf.items()})

    def find_relevant(self, sentences: list[str], top_k: int = 50, threshold: float = 0.1) -> list[tuple[str, str]]:
        """
        Find dictionary entries relevant to the given sentences.

        Args:
            sentences: List of sentences to match against.
            top_k: Maximum number of entries to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of (ukrainian, hutsul) tuples sorted by relevance.
        """
        if not self.entries:
            return []

        all_tokens: list[str] = []
        for sentence in sentences:
            all_tokens.extend(_tokenize(sentence))

        sentence_tf = Counter(all_tokens)
        max_freq = max(sentence_tf.values()) if sentence_tf else 1
        sentence_vec = {k: v / max_freq for k, v in sentence_tf.items()}

        scored_entries: list[tuple[float, tuple[str, str]]] = []
        for i, entry_vec in enumerate(self.entry_vectors):
            sim = _cosine_similarity(sentence_vec, entry_vec)
            if sim >= threshold:
                scored_entries.append((sim, self.entries[i]))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:top_k]]


class DialectPromptBuilder:
    """
    Generalized prompt builder for Ukrainian dialect/surzhyk translation.

    Works with any dialect by loading rules from a file that contains:
    - System instructions
    - Transformation rules
    - Few-shot examples (embedded in the rules file)
    """

    def __init__(
        self,
        rules_path: str | Path,
        dictionary_path: str | Path | None = None,
        max_dict_entries: int = 50,
    ):
        """
        Initialize the prompt builder.

        Args:
            rules_path: Path to the rules file containing system prompt,
                       rules, and few-shot examples.
            dictionary_path: Optional path to CSV dictionary file with
                            Dialect/Surzhyk and Ukrainian columns.
            max_dict_entries: Maximum dictionary entries to include per batch.
        """
        self.rules_path = Path(rules_path)
        self.dictionary_path = Path(dictionary_path) if dictionary_path else None
        self.max_dict_entries = max_dict_entries
        self._rules_template: str | None = None
        self._dictionary_entries: list[tuple[str, str]] | None = None
        self._dictionary_matcher: DictionaryMatcher | None = None

    def _load_rules(self) -> str:
        """Load rules from file."""
        if not self.rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")
        return self.rules_path.read_text(encoding="utf-8")

    def _load_dictionary_entries(self) -> list[tuple[str, str]]:
        """Load dictionary entries from CSV file."""
        if not self.dictionary_path or not self.dictionary_path.exists():
            return []

        entries = []
        with open(self.dictionary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hutsul = row.get("Hutsul", "").strip()
                ukrainian = row.get("Ukrainian", "").strip()
                if hutsul and ukrainian:
                    entries.append((ukrainian, hutsul))

        return entries

    @property
    def rules_template(self) -> str:
        """Get or load the rules template (cached)."""
        if self._rules_template is None:
            self._rules_template = self._load_rules()
        return self._rules_template

    @property
    def dictionary_entries(self) -> list[tuple[str, str]]:
        """Get or load dictionary entries (cached)."""
        if self._dictionary_entries is None:
            self._dictionary_entries = self._load_dictionary_entries()
        return self._dictionary_entries

    @property
    def dictionary_matcher(self) -> DictionaryMatcher:
        """Get or create dictionary matcher (cached)."""
        if self._dictionary_matcher is None:
            self._dictionary_matcher = DictionaryMatcher(self.dictionary_entries)
        return self._dictionary_matcher

    def _format_dictionary(self, entries: list[tuple[str, str]]) -> str:
        """Format dictionary entries as text."""
        if not entries:
            return "No relevant dictionary entries found."
        return "\n".join(f"- {ukr} → {huts}" for ukr, huts in entries)

    def build_system_prompt(self, sentences: list[str]) -> str:
        """
        Build system prompt with relevant dictionary entries for the batch.

        Args:
            sentences: List of sentences to find relevant dictionary entries for.

        Returns:
            System prompt with relevant dictionary entries injected.
        """
        relevant_entries = self.dictionary_matcher.find_relevant(
            sentences,
            top_k=self.max_dict_entries,
            threshold=0.05,
        )
        dictionary_text = self._format_dictionary(relevant_entries)
        return self.rules_template.format(dictionary=dictionary_text)

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
        system_prompt = self.build_system_prompt(sentences)
        system_len = len(system_prompt)
        user_prompt = self.build_user_prompt(sentences)
        user_len = len(user_prompt)
        return (system_len + user_len) // 4

    @property
    def dialect_name(self) -> str:
        """
        Extract dialect name from rules file path.

        e.g., "hutsul_rules_system.txt" -> "hutsul"
        """
        stem = self.rules_path.stem
        for suffix in ["_rules_system", "_rules", "_system"]:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem
