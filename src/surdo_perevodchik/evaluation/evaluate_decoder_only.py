import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from surdo_perevodchik.evaluation.metrics import compute_metrics


try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


SYSTEM_PROMPT = "Ти — перекладач з діалектів та суржику на літературну українську мову."


def format_prompt(source: str, tokenizer, use_system_prompt: bool = True) -> str:
    """Format a source text as a chat prompt for generation."""
    if use_system_prompt:
        user_content = f"{SYSTEM_PROMPT}\n\nПерекладіть: {source}"
    else:
        user_content = f"Перекладіть на літературну українську: {source}"

    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_predictions(
    model_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_k: int = 25,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    use_system_prompt: bool = True,
    use_4bit: bool = False,
    lora_adapter: str = None,
    attn_implementation: str = None,
) -> list[str]:
    """Generate predictions using a decoder-only model."""
    print(f"🔄 Loading model from {model_path}...")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if use_4bit:
        model_kwargs["quantization_config"] = bnb_config
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if lora_adapter:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA models. Install with: pip install peft")
        print(f"🔌 Loading LoRA adapter from {lora_adapter}...")
        model = PeftModel.from_pretrained(model, lora_adapter)

    model.eval()

    input_path = Path(input_file)
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_file)
        if "source" not in df.columns:
            raise ValueError("CSV must have 'source' column")
        sources = df["source"].tolist()
    else:
        with open(input_file, encoding="utf-8") as f:
            sources = [line.strip() for line in f if line.strip()]

    print(f"⚡ Generating predictions for {len(sources)} examples...")

    predictions = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(sources), batch_size)):
            batch_sources = sources[i : i + batch_size]

            prompts = [format_prompt(src, tokenizer, use_system_prompt) for src in batch_sources]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                generated_tokens = output[input_len:]
                decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predictions.append(decoded.strip())

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))

    print(f"✅ Predictions saved to {output_file}")
    return predictions


def run_evaluation(args):
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = args.output_dir / "predictions.txt"
    results_file = args.output_dir / "results.json"

    predictions = generate_predictions(
        model_path=args.model_path,
        input_file=args.test_file,
        output_file=str(predictions_file),
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        use_system_prompt=args.use_system_prompt,
        use_4bit=args.use_4bit,
        lora_adapter=args.lora_adapter,
        attn_implementation=args.attn_implementation,
    )

    print("\n📊 Computing metrics...")
    df = pd.read_csv(args.test_file)
    references = df["target"].tolist()

    references_file = args.output_dir / "references.txt"
    with open(references_file, "w", encoding="utf-8") as f:
        f.write("\n".join(references))

    metrics = compute_metrics(predictions, references)

    for metric, score in metrics.items():
        print(f"{metric:15s}: {score:6.2f}")

    results = {
        "metrics": metrics,
        "config": {
            "model_path": args.model_path,
            "test_file": args.test_file,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": args.do_sample,
        },
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate decoder-only LMs for translation")

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HuggingFace model ID")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV with source/target columns")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Output directory for results")

    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=25, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true", default=True, help="Use sampling")
    parser.add_argument("--no_sample", action="store_false", dest="do_sample", help="Use greedy decoding")

    # Prompt formatting
    parser.add_argument("--use_system_prompt", action="store_true", default=True, help="Include system prompt")
    parser.add_argument("--no_system_prompt", action="store_false", dest="use_system_prompt")

    # Model loading options
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--lora_adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument(
        "--attn_implementation", type=str, default=None, help="Attention impl (flash_attention_2, sdpa)"
    )

    args = parser.parse_args()
    run_evaluation(args)
