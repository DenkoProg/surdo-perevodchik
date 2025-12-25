"""Data generation module for creating synthetic parallel corpora."""

from surdo_perevodchik.data_generation.corpus_generator import DialectCorpusGenerator
from surdo_perevodchik.data_generation.llm_client import LLMClient, LLMConfig, create_llm_client
from surdo_perevodchik.data_generation.local_model_client import LocalModelClient, LocalModelConfig
from surdo_perevodchik.data_generation.openrouter_client import OpenRouterClient, OpenRouterConfig
from surdo_perevodchik.data_generation.prompt_builder import DialectPromptBuilder


__all__ = [
    "LLMClient",
    "LLMConfig",
    "create_llm_client",
    "OpenRouterClient",
    "OpenRouterConfig",
    "LocalModelClient",
    "LocalModelConfig",
    "DialectPromptBuilder",
    "DialectCorpusGenerator",
]
