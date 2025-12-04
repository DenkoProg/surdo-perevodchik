"""Data generation module for creating synthetic parallel corpora."""

from surdo_perevodchik.data_generation.corpus_generator import DialectCorpusGenerator
from surdo_perevodchik.data_generation.openrouter_client import OpenRouterClient, OpenRouterConfig
from surdo_perevodchik.data_generation.prompt_builder import DialectPromptBuilder


__all__ = [
    "OpenRouterClient",
    "OpenRouterConfig",
    "DialectPromptBuilder",
    "DialectCorpusGenerator",
]
