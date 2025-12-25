"""Local model client for running LLMs on GPU."""

from dataclasses import dataclass
import json
import re

from surdo_perevodchik.data_generation.llm_client import LLMClient, LLMConfig


@dataclass
class LocalModelConfig(LLMConfig):
    """Configuration for local model client."""

    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu"
    load_in_8bit: bool = True  # Use 8-bit quantization
    load_in_4bit: bool = False  # Use 4-bit quantization (overrides 8-bit)
    use_flash_attention: bool = False  # Requires flash-attn package
    max_memory: dict | None = None  # e.g., {"cuda:0": "20GiB", "cpu": "30GiB"}


class LocalModelClient(LLMClient):
    """Client for running local LLMs with transformers."""

    def __init__(self, config: LocalModelConfig | None = None):
        super().__init__(config)
        if not isinstance(self.config, LocalModelConfig):
            self.config = LocalModelConfig(**self.config.__dict__)

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer with quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError as e:
            raise ImportError(
                "Local model requires: pip install transformers accelerate bitsandbytes torch"
            ) from e

        print(f"Loading model {self.config.model}...")

        # Configure quantization
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization (NF4)")
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("Using 8-bit quantization")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=False,  # Better compatibility
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "device_map": self.config.device,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory

        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            **model_kwargs,
        )
        self.model.eval()
        print(f"Model loaded on {self.model.device}")

    @property
    def name(self) -> str:
        """Return the name/identifier of this client."""
        quant = "4bit" if self.config.load_in_4bit else "8bit" if self.config.load_in_8bit else "fp16"
        return f"local:{self.config.model}:{quant}"

    def _format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompts for Mistral/Llama chat template."""
        # Try to use the model's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fall back to manual formatting
                pass

        # Manual Mistral-Instruct format
        return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

    def _extract_json_response(self, text: str, num_sentences: int) -> str | None:
        """Try to extract JSON from response if present."""
        # Look for JSON object in response
        json_match = re.search(r'\{[^{}]*"translations"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "translations" in data and isinstance(data["translations"], list):
                    # Return just the JSON object as string
                    return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                pass

        # If no valid JSON, return raw text (parser will handle it)
        return text

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        num_sentences: int = 0,
    ) -> str | None:
        """
        Generate text using local model.

        Args:
            system_prompt: System message with instructions/rules.
            user_prompt: User message with content to process.
            num_sentences: Number of sentences being translated (hint for parsing).

        Returns:
            Generated text or None if generation failed.
        """
        import torch

        try:
            # Format prompt
            prompt = self._format_chat_prompt(system_prompt, user_prompt)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            # Try to extract JSON if present
            return self._extract_json_response(generated_text.strip(), num_sentences)

        except Exception as e:
            print(f"Error during generation: {e}")
            return None

    def __del__(self):
        """Clean up model on deletion."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
