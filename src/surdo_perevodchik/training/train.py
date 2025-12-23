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
    inputs = tokenizer(
        batch["source"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=batch["target"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )

    inputs["labels"] = labels["input_ids"]
    return inputs


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("csv", data_files={"data": args.train_file})["data"]
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=args.val_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    tokenized_train = split["train"].map(
        lambda x: preprocess(x, tokenizer, args.max_length), batched=True, remove_columns=ds.column_names
    )
    tokenized_val = split["test"].map(
        lambda x: preprocess(x, tokenizer, args.max_length), batched=True, remove_columns=ds.column_names
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        gradient_checkpointing=args.grad_checkpoint,
        optim=args.optim,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
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
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/byt5-large")
    parser.add_argument("--output_dir", type=str, default="models/byt5-hutsul-large")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()
    main(args)
