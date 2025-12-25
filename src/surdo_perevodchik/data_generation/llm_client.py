from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class LLMConfig:
    """Base configuration for LLM clients."""

    model: str = "mistralai/mistral-7b-instruct:free"
    max_tokens: int = 2048
    temperature: float = 0.7
    max_retries: int = 5
    retry_delay: float = 5.0
    timeout: int = 120


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        num_sentences: int = 0,
    ) -> str | None:
        """
        Generate text from prompts.

        Args:
            system_prompt: System message with instructions/rules.
            user_prompt: User message with content to process.
            num_sentences: Number of sentences being translated (for structured output).

        Returns:
            Generated text or None if generation failed.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name/identifier of this client."""
        pass


def create_llm_client(
    provider: Literal["openrouter", "local"] = "openrouter",
    model: str | None = None,
    **kwargs,
) -> LLMClient:
    """
    Factory function to create LLM clients.

    Args:
        provider: "openrouter" for API or "local" for GPU inference.
        model: Model identifier (uses default if not provided).
        **kwargs: Additional config parameters for the specific client.

    Returns:
        Configured LLM client instance.

    Examples:
        # OpenRouter API
        client = create_llm_client("openrouter", model="mistralai/mistral-7b-instruct:free")

        # Local GPU
        client = create_llm_client("local", model="mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True)
    """
    if provider == "openrouter":
        from surdo_perevodchik.data_generation.openrouter_client import OpenRouterClient, OpenRouterConfig

        config = OpenRouterConfig(**kwargs)
        if model:
            config.model = model
        return OpenRouterClient(config)

    elif provider == "local":
        from surdo_perevodchik.data_generation.local_model_client import LocalModelClient, LocalModelConfig

        config = LocalModelConfig(**kwargs)
        if model:
            config.model = model
        return LocalModelClient(config)

    else:
        raise ValueError(f"Unknown provider: {provider}. Choose 'openrouter' or 'local'.")
