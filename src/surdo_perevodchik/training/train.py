import argparse
import os

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def preprocess(batch, tokenizer, max_length):
    inputs = batch["source"]
    targets = batch["target"]
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("csv", data_files={"all": args.train_file})["all"]
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=args.val_size)
    train_ds = split["train"]
    val_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized_train = train_ds.map(
        lambda batch: preprocess(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        lambda batch: preprocess(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        predict_with_generate=True,
        save_total_limit=2,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/parallel/hutsul_parallel.csv")
    parser.add_argument("--model_name", type=str, default="google/mt5-small")
    parser.add_argument("--output_dir", type=str, default="models/mt5-hutsul-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()
    main(args)
