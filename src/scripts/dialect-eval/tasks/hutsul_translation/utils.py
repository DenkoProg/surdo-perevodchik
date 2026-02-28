import datetime
import re

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from loguru import logger as eval_logger
import pandas as pd


def doc_to_visual(doc):
    return []


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = ""
    if lmms_eval_specific_kwargs:
        prompt = lmms_eval_specific_kwargs.get("prompt", "")
    return prompt + doc["source"]


def _chrf(hyp: str, ref: str) -> float:
    """Character F-score (chrF), n=1..6, beta=2 (recall-weighted)."""
    hyp = hyp.strip().lower()
    ref = ref.strip().lower()

    def ngrams(text, n):
        counts: dict[str, int] = {}
        for i in range(len(text) - n + 1):
            ng = text[i : i + n]
            counts[ng] = counts.get(ng, 0) + 1
        return counts

    total = 0.0
    for n in range(1, 7):
        h = ngrams(hyp, n)
        r = ngrams(ref, n)
        matches = sum(min(h[ng], r.get(ng, 0)) for ng in h)
        h_total = max(len(hyp) - n + 1, 0)
        r_total = max(len(ref) - n + 1, 0)
        precision = matches / h_total if h_total else 0.0
        recall = matches / r_total if r_total else 0.0
        beta = 2
        if precision + recall > 0:
            f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        else:
            f = 0.0
        total += f

    return total / 6


def process_results(doc, results):
    reference = doc.get("target", "")
    prediction = results[0].strip() if results else ""
    score = _chrf(prediction, reference)
    return {
        "chrf_score": {
            "reference": reference,
            "prediction": prediction,
            "score": score,
        }
    }


def aggregate_results(results, args=None):
    if not results:
        eval_logger.warning("No results to aggregate")
        return 0.0

    df = pd.DataFrame(results)
    mean_chrf = df["score"].mean()
    total = len(df)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = generate_submission_file(f"hutsul_results_{now}.md", args, subpath="results")

    with open(file_name, "w", encoding="utf-8") as f:
        f.write("# Hutsul Translation Evaluation Results\n\n")
        f.write(f"**chrF (avg):** {mean_chrf:.4f}\n")
        f.write(f"**Total samples:** {total}\n\n")
        f.write("## Sample Predictions\n\n")
        f.write("| Source (dialect) | Reference | Prediction | chrF |\n")
        f.write("|------------------|-----------|------------|------|\n")
        for _, row in df.head(50).iterrows():
            src = re.sub(r"\|", "/", row.get("source", "")) if "source" in df.columns else ""
            ref = re.sub(r"\|", "/", row["reference"])
            pred = re.sub(r"\|", "/", row["prediction"])
            f.write(f"| {src} | {ref} | {pred} | {row['score']:.3f} |\n")

    eval_logger.info(f"chrF: {mean_chrf:.4f} over {total} samples")
    return mean_chrf
