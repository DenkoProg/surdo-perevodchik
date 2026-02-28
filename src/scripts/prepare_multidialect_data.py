"""Prepare multi-dialect training data.

Reads all dialect CSVs, adds dialect prefixes, performs stratified train/val split,
and outputs train.csv, val.csv, test.csv.
"""

import argparse
from pathlib import Path

import pandas as pd


DIALECT_SOURCES = {
    "hutsul": [
        "data/parallel/hutsul/manual_hutsul_corpus.csv",
        "data/parallel/hutsul/synthetic_hutsul_corpus.csv",
    ],
    "boiko": [
        "data/parallel/boiko/synthetic_boiko_corpus.csv",
    ],
    "transcarpathian": [
        "data/parallel/transcarpathian/synthetic_transcarpathian_corpus.csv",
    ],
    "surzhyk": [
        "data/parallel/surzhyk/synthetic_surzhyk_corpus_llm.csv",
        "data/parallel/surzhyk/programmatic_surzhyk_corpus.csv",
    ],
}

DIALECT_NAMES_UK = {
    "hutsul": "гуцульської",
    "boiko": "бойківської",
    "transcarpathian": "закарпатської",
    "surzhyk": "суржику",
}


def load_all_dialects(base_dir: Path) -> pd.DataFrame:
    parts = []
    for dialect, files in DIALECT_SOURCES.items():
        for f in files:
            path = base_dir / f
            df = pd.read_csv(path)
            df["dialect"] = dialect
            parts.append(df)
            print(f"  {dialect}: {path.name} - {len(df)} rows")
    combined = pd.concat(parts, ignore_index=True)
    print(f"\nTotal: {len(combined)} rows")
    return combined


def add_prefix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source"] = df.apply(
        lambda row: f"Переклади з {DIALECT_NAMES_UK[row['dialect']]}: {row['source']}",
        axis=1,
    )
    return df


def stratified_split(df: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_parts = []
    train_parts = []
    for _, group in df.groupby("dialect"):
        n_val = max(1, int(len(group) * val_size))
        shuffled = group.sample(frac=1, random_state=seed)
        val_parts.append(shuffled.iloc[:n_val])
        train_parts.append(shuffled.iloc[n_val:])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-dialect training data")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/parallel",
        help="Output directory for train/val/eval CSVs",
    )
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--eval_per_dialect",
        type=int,
        default=50,
        help="Number of eval samples per dialect (taken from val set)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = base_dir / args.output_dir

    print("Loading all dialect datasets...")
    df = load_all_dialects(base_dir)

    # Drop rows with missing source or target
    before = len(df)
    df = df.dropna(subset=["source", "target"])
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing values")

    print("\nDialect distribution:")
    for dialect, count in df["dialect"].value_counts().items():
        print(f"  {dialect}: {count}")

    # Add dialect prefixes
    print("\nAdding dialect prefixes...")
    df = add_prefix(df)

    # Stratified split
    print(f"\nSplitting: {1 - args.val_size:.0%} train / {args.val_size:.0%} val (seed={args.seed})...")
    train_df, val_df = stratified_split(df, args.val_size, args.seed)

    # Carve out eval set from val
    eval_parts = []
    val_remaining_parts = []
    for dialect, group in val_df.groupby("dialect"):
        n_eval = min(args.eval_per_dialect, len(group))
        eval_parts.append(group.iloc[:n_eval])
        val_remaining_parts.append(group.iloc[n_eval:])

    eval_df = pd.concat(eval_parts).reset_index(drop=True)
    val_df = pd.concat(val_remaining_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Eval:  {len(eval_df)} rows")

    print("\nTrain dialect distribution:")
    for dialect, count in train_df["dialect"].value_counts().items():
        print(f"  {dialect}: {count}")

    print("\nVal dialect distribution:")
    for dialect, count in val_df["dialect"].value_counts().items():
        print(f"  {dialect}: {count}")

    print("\nEval dialect distribution:")
    for dialect, count in eval_df["dialect"].value_counts().items():
        print(f"  {dialect}: {count}")

    # Save - training CSVs only have source,target (no dialect column)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df[["source", "target"]].to_csv(output_dir / "train.csv", index=False)
    val_df[["source", "target"]].to_csv(output_dir / "val.csv", index=False)
    eval_df[["source", "target", "dialect"]].to_csv(output_dir / "test.csv", index=False)

    print(f"\nSaved to {output_dir}:")
    print(f"  train.csv          ({len(train_df)} rows)")
    print(f"  val.csv            ({len(val_df)} rows)")
    print(f"  test.csv ({len(eval_df)} rows)")

    # Print a few examples
    print("\nSample prefixed rows:")
    for dialect in DIALECT_NAMES_UK:
        sample = train_df[train_df["dialect"] == dialect].iloc[0]
        print(f"\n  [{dialect}]")
        print(f"  source: {sample['source'][:120]}...")
        print(f"  target: {sample['target'][:120]}...")


if __name__ == "__main__":
    main()
