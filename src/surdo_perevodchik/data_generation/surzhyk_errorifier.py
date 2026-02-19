"""Programmatic Surzhyk errorifier adapted from diploma/unlp/errorification/SurzhErrorifier.

Converts standard Ukrainian text into Surzhyk by replacing authentic Ukrainian
phrases/words with Russian-calqued variants using the surzhyk dictionary.

Two-phase strategy (inspired by SurzhGenerator from diploma/unlp):
  Phase 1 — phrase-level regex substitution (multi-word entries, longest-first).
  Phase 2 — word-level morphological substitution: uses pymorphy2 to match
             inflected forms of single-word entries and inflect the surzhyk
             replacement to the same grammatical form (case, number, gender).
             Falls back to plain regex when pymorphy2 is unavailable.

Key morphological tricks adapted from FormExtractorInflector / SurzhGenerator:
  - Lemma-based matching: index Ukrainian words by their base form so that
    any inflected variant (gen/dat/acc/…) triggers the substitution.
  - Agreement-preserving inflection: `pymorphy2.Parse.inflect()` is called with
    the grammemes extracted from the source token, ensuring the replacement word
    carries the same case/number/gender as the original.
  - Robust fallback: if full inflection fails, the method retries with only the
    case grammeme, then returns the dictionary base form unchanged.
"""

import csv
from pathlib import Path
import re


# Cyrillic word characters including Ukrainian-specific letters and apostrophe
_UA_WORD_RE = re.compile(r"[а-яіїєёА-ЯІЇЄ'ʼ\-]+", re.UNICODE)

# Grammeme groups used for inflection agreement
_CASE_GRAMMEMES = {"nomn", "gent", "datv", "accs", "ablt", "loct", "voct"}
_NUMBER_GRAMMEMES = {"sing", "plur"}


class SurzhykErrorifier:
    """
    Generate Surzhyk text from standard Ukrainian using phrase and word substitution.

    Loads a (Surzhyk, Ukrainian) dictionary and replaces standard Ukrainian
    phrases/words with their Russian-influenced Surzhyk equivalents.

    When pymorphy2 is available, single-word substitutions use full morphological
    analysis: inflected forms of dictionary entries are matched by lemma, and the
    surzhyk replacement is inflected to preserve grammatical agreement (case,
    number, gender).  Multi-word phrases always use regex matching.

    Args:
        dictionary_path: Path to CSV with 'Surzhyk' and 'Ukrainian' columns.
        use_morphology: Whether to enable pymorphy2-based morphological matching
            for single-word pairs (default True; silently degrades if not installed).
    """

    def __init__(self, dictionary_path: str | Path, use_morphology: bool = True):
        self.morph = None
        if use_morphology:
            try:
                import pymorphy2  # noqa: PLC0415

                self.morph = pymorphy2.MorphAnalyzer(lang="uk")
            except Exception:  # ImportError or missing Ukrainian dict
                pass

        # phrase_pairs: (ukrainian_standard, surzhyk) where Ukrainian side is multi-word
        self.phrase_pairs: list[tuple[str, str]] = []
        # single_pairs: (ukrainian_standard, surzhyk) both sides single words
        self.single_pairs: list[tuple[str, str]] = []
        self._load_pairs(Path(dictionary_path))

        # Longest-first within each group ensures longer phrases win over sub-phrases
        self.phrase_pairs.sort(key=lambda p: len(p[0]), reverse=True)
        self.single_pairs.sort(key=lambda p: len(p[0]), reverse=True)

        # Lemma → [(ukrainian_base, surzhyk_base)] for morphological matching
        self._lemma_index: dict[str, list[tuple[str, str]]] = {}
        if self.morph:
            self._build_lemma_index()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_pairs(self, path: Path) -> None:
        """Load (standard_ukrainian, surzhyk) substitution pairs from CSV."""
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                surzhyk = row.get("Surzhyk", "").strip()
                ukrainian = row.get("Ukrainian", "").strip()
                if not surzhyk or not ukrainian or surzhyk == ukrainian:
                    continue
                # Multi-word on the Ukrainian (source) side → phrase matching
                if " " in ukrainian:
                    self.phrase_pairs.append((ukrainian, surzhyk))
                else:
                    self.single_pairs.append((ukrainian, surzhyk))

    # ------------------------------------------------------------------
    # Morphological helpers (pymorphy2-based, adapted from FormExtractorInflector)
    # ------------------------------------------------------------------

    def _get_lemma(self, word: str) -> str:
        """Return the normal (base) form of a Ukrainian word via pymorphy2."""
        if self.morph is None:
            return word.lower()
        parsed = self.morph.parse(word)
        return parsed[0].normal_form if parsed else word.lower()

    def _build_lemma_index(self) -> None:
        """Pre-index single-word pairs by the lemma of the Ukrainian standard form."""
        for ukrainian, surzhyk in self.single_pairs:
            lemma = self._get_lemma(ukrainian)
            self._lemma_index.setdefault(lemma, []).append((ukrainian, surzhyk))

    def _inflect_to_match(self, surzhyk_word: str, source_word: str) -> str:
        """
        Inflect *surzhyk_word* so it carries the same grammatical form as
        *source_word* (the original Ukrainian token being replaced).

        Adapted from FormExtractorInflector.inflect_as_in_sentence():
        - Extract case + number from the source token's parse.
        - Try to inflect the surzhyk word with all collected grammemes.
        - If that fails, retry with just the case grammeme (robust fallback).
        - If both fail, return the dictionary base form unchanged.
        """
        if self.morph is None:
            return surzhyk_word

        src_parsed = self.morph.parse(source_word)
        if not src_parsed:
            return surzhyk_word
        surz_parsed = self.morph.parse(surzhyk_word)
        if not surz_parsed:
            return surzhyk_word

        src_tag = src_parsed[0].tag
        surz_parse = surz_parsed[0]

        # Collect the grammemes we want to copy (case + number)
        grammemes: set[str] = set()
        found_case: str | None = None
        for g in _CASE_GRAMMEMES:
            if g in src_tag.grammemes:
                grammemes.add(g)
                found_case = g
                break
        for g in _NUMBER_GRAMMEMES:
            if g in src_tag.grammemes:
                grammemes.add(g)
                break

        if not grammemes:
            return surzhyk_word

        # Attempt 1: full case+number inflection
        try:
            inflected = surz_parse.inflect(grammemes)
            if inflected:
                return inflected.word
        except Exception:
            pass

        # Attempt 2: case only (FormExtractorInflector fallback trick)
        if found_case:
            try:
                inflected = surz_parse.inflect({found_case})
                if inflected:
                    return inflected.word
            except Exception:
                pass

        return surzhyk_word

    # ------------------------------------------------------------------
    # Pattern helpers (regex-based, for phrase matching)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_phrase_pattern(phrase: str) -> re.Pattern:
        """
        Compile a case-insensitive whole-phrase regex.
        Multi-word phrases use non-word lookaround; single words use \\b.
        """
        escaped = re.escape(phrase)
        if " " in phrase:
            return re.compile(r"(?<![^\W\s])" + escaped + r"(?![^\W\s])", re.IGNORECASE | re.UNICODE)
        return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE | re.UNICODE)

    @staticmethod
    def _preserve_case(original: str, replacement: str) -> str:
        """Copy leading capitalisation from *original* onto *replacement*."""
        if not original or not replacement:
            return replacement
        if original[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    # ------------------------------------------------------------------
    # Core surzhykification
    # ------------------------------------------------------------------

    def surzhykify(self, sentence: str) -> tuple[str, int]:
        """
        Replace standard Ukrainian phrases/words in *sentence* with Surzhyk variants.

        Phase 1: multi-word phrase substitutions (regex, longest-first).
        Phase 2: single-word substitutions.
          - With pymorphy2: lemma-indexed morphological matching + inflection.
          - Without pymorphy2: plain word-boundary regex (matches base form only).

        Returns:
            (modified_sentence, num_substitutions_made)
        """
        result = sentence
        total_subs = 0

        # ── Phase 1: phrase substitutions ──────────────────────────────
        for ukrainian, surzhyk in self.phrase_pairs:
            pattern = self._make_phrase_pattern(ukrainian)

            def _replace_phrase(m: re.Match, s: str = surzhyk) -> str:
                return self._preserve_case(m.group(0), s)

            new_result, n = pattern.subn(_replace_phrase, result)
            if n > 0:
                result = new_result
                total_subs += n

        # ── Phase 2a: morphological word substitutions (pymorphy2 path) ──
        if self.morph and self._lemma_index:
            result, n = self._morphological_substitution(result)
            total_subs += n

        # ── Phase 2b: regex fallback for word substitutions ──────────────
        else:
            for ukrainian, surzhyk in self.single_pairs:
                pattern = self._make_phrase_pattern(ukrainian)

                def _replace_word(m: re.Match, s: str = surzhyk) -> str:
                    return self._preserve_case(m.group(0), s)

                new_result, n = pattern.subn(_replace_word, result)
                if n > 0:
                    result = new_result
                    total_subs += n

        return result, total_subs

    def _morphological_substitution(self, sentence: str) -> tuple[str, int]:
        """
        Word-level substitution with morphological agreement.

        Tokenises *sentence* into Ukrainian word spans, looks each token's lemma
        up in the index, and replaces matching tokens with an appropriately
        inflected surzhyk form.  Replacements are applied in reverse-position
        order so string offsets remain valid.
        """
        matches = list(_UA_WORD_RE.finditer(sentence))
        if not matches:
            return sentence, 0

        replacements: list[tuple[int, int, str]] = []

        for m in matches:
            word = m.group(0)
            # Skip very short tokens to avoid spurious matches (e.g., 'в', 'і')
            if len(word) < 3:
                continue
            lemma = self._get_lemma(word)
            if lemma not in self._lemma_index:
                continue
            # Use the first matching pair (single_pairs already sorted longest-first)
            _, surzhyk_base = self._lemma_index[lemma][0]
            inflected = self._inflect_to_match(surzhyk_base, word)
            replacement = self._preserve_case(word, inflected)
            if replacement.lower() != word.lower():
                replacements.append((m.start(), m.end(), replacement))

        if not replacements:
            return sentence, 0

        result = sentence
        for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
            result = result[:start] + replacement + result[end:]

        return result, len(replacements)

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_pairs(
        self,
        sentences: list[str],
        min_substitutions: int = 1,
    ) -> list[tuple[str, str]]:
        """
        Generate (surzhyk, standard_ukrainian) parallel pairs.

        Only emits a pair when at least *min_substitutions* replacements were
        made, ensuring the surzhyk column genuinely differs from the standard.

        Returns:
            List of (surzhyk_sentence, standard_sentence) tuples.
        """
        result = []
        for sentence in sentences:
            surzhyk_sentence, n = self.surzhykify(sentence)
            if n >= min_substitutions:
                result.append((surzhyk_sentence, sentence))
        return result
