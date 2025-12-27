import argparse
import os

from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
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


def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    vocab_size = len(tokenizer)
    preds = np.where((preds >= 0) & (preds < vocab_size), preds, tokenizer.pad_token_id)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("csv", data_files={"data": args.train_file})["data"]
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=args.val_size, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if model.generation_config is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.repetition_penalty = args.repetition_penalty
        model.generation_config.no_repeat_ngram_size = args.no_repeat_ngram_size
    else:
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    tokenized_train = split["train"].map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=ds.column_names,
        num_proc=4,
        desc="Tokenizing training data",
    )
    tokenized_val = split["test"].map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=ds.column_names,
        num_proc=4,
        desc="Tokenizing validation data",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    metric = evaluate.load("sacrebleu")

    total_steps = (len(tokenized_train) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = min(1000, int(0.1 * total_steps))

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.grad_checkpoint,
        optim=args.optim,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to=["tensorboard"],
        push_to_hub=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        logging_first_step=True,
        eval_on_start=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metric),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_output = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/umt5-base")
    parser.add_argument("--output_dir", type=str, default="models/umt5-base-hutsul")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty for generation")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Block repetition of n-grams")

    args = parser.parse_args()
    main(args)
