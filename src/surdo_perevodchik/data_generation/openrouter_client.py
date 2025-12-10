"""OpenRouter API client for LLM generation."""

from dataclasses import dataclass
import os
import time

from dotenv import load_dotenv
import requests


load_dotenv()


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter API client."""

    api_key: str | None = None
    model: str = "mistralai/mistral-7b-instruct:free"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    max_tokens: int = 2048
    temperature: float = 0.7
    max_retries: int = 5
    retry_delay: float = 5.0
    timeout: int = 120
    use_structured_output: bool = True
    app_name: str = "Surdo Perevodchik"
    app_url: str = "https://github.com/DenkoProg/surdo-perevodchik"


class OpenRouterClient:
    """Client for calling OpenRouter API with retry logic."""

    def __init__(self, config: OpenRouterConfig | None = None):
        self.config = config or OpenRouterConfig()
        self.api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Export it as environment variable or pass in config.")

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.app_url,
            "X-Title": self.config.app_name,
        }

    def _build_translation_schema(self, num_sentences: int) -> dict:
        """Build JSON schema for structured translation output."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "translations",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "translations": {
                            "type": "array",
                            "description": "Array of translated sentences in dialect",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["translations"],
                    "additionalProperties": False,
                },
            },
        }

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        num_sentences: int = 0,
    ) -> dict:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if self.config.use_structured_output and num_sentences > 0:
            payload["response_format"] = self._build_translation_schema(num_sentences)

        return payload

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        num_sentences: int = 0,
    ) -> str | None:
        """
        Send a chat completion request to OpenRouter.

        Args:
            system_prompt: System message with instructions/rules.
            user_prompt: User message with content to process.
            num_sentences: Number of sentences being translated (for structured output).

        Returns:
            Generated text or None if all retries failed.
        """
        headers = self._build_headers()
        payload = self._build_payload(system_prompt, user_prompt, num_sentences)

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )

                if response.status_code == 429:
                    wait_time = self.config.retry_delay * (2**attempt)
                    print(f"Rate limited (429). Waiting {wait_time:.1f}s... (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 500:
                    wait_time = self.config.retry_delay * (attempt + 1)
                    print(f"Server error ({response.status_code}). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f"Unexpected response format: {data}")
                    return None

            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{self.config.max_retries})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                continue

            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                continue

        print("All retries exhausted.")
        return None
