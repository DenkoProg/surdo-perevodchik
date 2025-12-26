import argparse
import os

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments


try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from trl import SFTTrainer

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


SYSTEM_PROMPT = "Ти — перекладач з діалектів та суржику на літературну українську мову."


def format_chat_prompt(example, tokenizer, use_system_prompt: bool = True):
    if use_system_prompt:
        user_content = f"{SYSTEM_PROMPT}\n\nПерекладіть: {example['source']}"
    else:
        user_content = f"Перекладіть на літературну українську: {example['source']}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["target"]},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main(args):
    if not TRL_AVAILABLE:
        raise ImportError("trl is required for decoder-only training. Install with: pip install trl")

    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset("csv", data_files={"data": args.train_file})["data"]
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=args.val_size, seed=42)

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        "trust_remote_code": True,
        "device_map": "auto" if not args.use_4bit else None,
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        if args.use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else "all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    train_dataset = split["train"].map(
        lambda x: format_chat_prompt(x, tokenizer, args.use_system_prompt),
        remove_columns=split["train"].column_names,
        desc="Formatting training data",
    )
    eval_dataset = split["test"].map(
        lambda x: format_chat_prompt(x, tokenizer, args.use_system_prompt),
        remove_columns=split["test"].column_names,
        desc="Formatting validation data",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        optim=args.optim,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        gradient_checkpointing=args.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_checkpoint else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=args.max_length,
        packing=False,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.use_lora:
        trainer.save_model(args.output_dir)
        if args.merge_lora:
            print("🔀 Merging LoRA weights...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(f"{args.output_dir}-merged")
            tokenizer.save_pretrained(f"{args.output_dir}-merged")
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune decoder-only LMs for translation")

    parser.add_argument(
        "--train_file", type=str, required=True, help="Path to training CSV with source/target columns"
    )
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation split ratio")

    parser.add_argument("--model_name", type=str, default="INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0")
    parser.add_argument("--output_dir", type=str, default="models/mamaylm-hutsul")
    parser.add_argument(
        "--attn_implementation", type=str, default=None, help="Attention impl (flash_attention_2, sdpa)"
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--optim", type=str, default="adamw_torch")

    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended for modern GPUs)")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")

    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated target modules")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights after training")

    parser.add_argument("--use_system_prompt", action="store_true", default=True, help="Include system prompt")
    parser.add_argument("--no_system_prompt", action="store_false", dest="use_system_prompt")

    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()
    main(args)
