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
):
    """
    Args:
        model_path: Path to trained model
        input_file: CSV file with 'source' column or text file (one sentence per line)
        output_file: Where to save predictions
        batch_size: Batch size for inference
        max_length: Maximum generation length
    """
    print(f"Loading model from {model_path}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
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

    print(f"Generating predictions for {len(sources)} examples...")

    predictions = []
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

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                repetition_penalty=1.2,
                bad_words_ids=[[250099], [250098], [250097]],  # Block <extra_id_0>, <extra_id_1>, <extra_id_2>
            )

        outputs = outputs.cpu()
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print(f"Predictions saved to {output_file}")
    return predictions


def run_evaluation(
    model_path: str,
    test_file: str,
    output_dir: str,
    batch_size: int = 8,
    max_length: int = 128,
):
    """
    Args:
        model_path: Path to trained model
        test_file: CSV with 'source' and 'target' columns
        output_dir: Directory to save predictions and results
        batch_size: Batch size for inference
        max_length: Maximum generation length
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "predictions.txt"
    references_file = output_dir / "references.txt"
    results_file = output_dir / "results.json"

    print(f"Loading test data from {test_file}...")
    df = pd.read_csv(test_file)

    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError("Test CSV must have 'source' and 'target' columns")

    with open(references_file, "w", encoding="utf-8") as f:
        for target in df["target"]:
            f.write(str(target) + "\n")

    predictions = generate_predictions(
        model_path=model_path,
        input_file=test_file,
        output_file=str(predictions_file),
        batch_size=batch_size,
        max_length=max_length,
    )

    print("\nComputing metrics...")
    references = df["target"].tolist()
    metrics = compute_metrics(predictions, references)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in metrics.items():
        print(f"{metric:10s}: {score:6.2f}")
    print("=" * 50)
    print(f"\nResults saved to {results_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="CSV file with 'source' and 'target' columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Directory to save predictions and results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        test_file=args.test_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
