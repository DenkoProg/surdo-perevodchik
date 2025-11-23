from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    file_references = [[ref] for ref in references]

    bleu = corpus_bleu(predictions, file_references)
    chrf = corpus_chrf(predictions, file_references, word_order=2)
    ter = corpus_ter(predictions, file_references)

    return {
        "BLEU": round(bleu.score, 2),
        "chrF++": round(chrf.score, 2),
        "TER": round(ter.score, 2),
    }


def evaluate_predictions(
    predictions_file: str, references_file: str
) -> dict[str, float]:
    with open(predictions_file, encoding="utf-8") as f:
        predictions = [line.strip() for line in f if line.strip()]

    with open(references_file, encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    if len(predictions) != len(references):
        raise ValueError(
            f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    return compute_metrics(predictions, references)
