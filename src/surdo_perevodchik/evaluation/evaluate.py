import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from surdo_perevodchik.evaluation.metrics import compute_metrics


def generate_predictions(
    model_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 8,
    max_length: int = 128,
    num_beams: int = 4,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    print(f"🔄 Loading model from {model_path}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    input_path = Path(input_file)
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_file)
        if "source" not in df.columns:
            raise ValueError("CSV must have 'source' column")
        sources = df["source"].tolist()
    else:
        with open(input_file, encoding="utf-8") as f:
            sources = [line.strip() for line in f if line.strip()]

    print(f"⚡ Generating predictions for {len(sources)} examples on {device} ({dtype})...")

    predictions = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(sources), batch_size)):
            batch = sources[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded)

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
        max_length=args.max_length,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    print("\n📊 Computing metrics...")
    df = pd.read_csv(args.test_file)
    references = df["target"].tolist()

    metrics = compute_metrics(predictions, references)

    for metric, score in metrics.items():
        print(f"{metric:15s}: {score:6.2f}")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    args = parser.parse_args()
    run_evaluation(args)
