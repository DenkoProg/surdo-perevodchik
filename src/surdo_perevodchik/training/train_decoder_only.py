import argparse
import os

from datasets import load_dataset
import torch
from transformers import TrainingArguments


try:
    from unsloth import FastModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

try:
    from trl import SFTTrainer

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


def format_chat_prompt(example, tokenizer):
    # source already contains the dialect prefix: "Переклади з гуцульської: ..."
    messages = [
        {"role": "user", "content": example["source"]},
        {"role": "assistant", "content": example["target"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def load_model_unsloth(args):
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=args.use_4bit,
        dtype=None,  # auto-detects bfloat16 / float16
    )

    if args.use_lora:
        target_modules = (
            args.lora_target_modules.split(",")
            if args.lora_target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = FastModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # optimized async checkpointing
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )

    return model, tokenizer


def load_model_transformers(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        "device_map": "auto",
    }
    if args.use_4bit:
        model_kwargs["quantization_config"] = bnb_config

    attn_impl = _resolve_attn_implementation(args.attn_implementation)
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

    return model, tokenizer


def _resolve_attn_implementation(requested):
    if requested is not None:
        return requested
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def main(args):
    # Prevent mid-epoch OOM caused by VRAM fragmentation.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not TRL_AVAILABLE:
        raise ImportError("trl is required. Install with: pip install trl")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.val_file:
        train_ds = load_dataset("csv", data_files={"train": args.train_file})["train"].shuffle(seed=42)
        val_ds = load_dataset("csv", data_files={"val": args.val_file})["val"].shuffle(seed=42)
        split = {"train": train_ds, "test": val_ds}
    else:
        ds = load_dataset("csv", data_files={"data": args.train_file})["data"].shuffle(seed=42)
        split = ds.train_test_split(test_size=args.val_size, seed=42)

    if UNSLOTH_AVAILABLE:
        print("Using Unsloth for optimized training")
        model, tokenizer = load_model_unsloth(args)
    else:
        print("Unsloth not found, falling back to transformers + peft")
        model, tokenizer = load_model_transformers(args)

    train_dataset = split["train"].map(
        lambda x: format_chat_prompt(x, tokenizer),
        remove_columns=split["train"].column_names,
        desc="Formatting training data",
    )
    eval_dataset = split["test"].map(
        lambda x: format_chat_prompt(x, tokenizer),
        remove_columns=split["test"].column_names,
        desc="Formatting validation data",
    )

    # Step-based eval/save avoids epoch-boundary memory spikes.
    eval_strategy = "steps" if args.eval_steps else "epoch"
    save_strategy = "steps" if args.save_steps else "epoch"

    # When using Unsloth, bf16/fp16 are handled internally; pass both False to avoid conflicts.
    bf16 = args.bf16 and not UNSLOTH_AVAILABLE
    fp16 = args.fp16 and not UNSLOTH_AVAILABLE

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        optim=args.optim,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        logging_strategy="steps",
        logging_steps=50,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        # gradient_checkpointing handled by Unsloth's use_gradient_checkpointing="unsloth"
        gradient_checkpointing=args.grad_checkpoint and not UNSLOTH_AVAILABLE,
        gradient_checkpointing_kwargs={"use_reentrant": False} if (args.grad_checkpoint and not UNSLOTH_AVAILABLE) else None,
        dataloader_pin_memory=False,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        dataset_num_proc=2,
        packing=args.packing,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if args.use_lora:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        if args.merge_lora:
            print("Merging LoRA weights...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(f"{args.output_dir}-merged")
            tokenizer.save_pretrained(f"{args.output_dir}-merged")
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune decoder-only LMs for dialect translation")

    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--val_size", type=float, default=0.1)

    parser.add_argument("--model_name", type=str, default="INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0")
    parser.add_argument("--output_dir", type=str, default="models/mamaylm-multidialect")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention impl for fallback (transformers) path. Auto-detected if not set.",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="Optimizer. adamw_8bit is recommended with Unsloth; paged_adamw_8bit for vanilla transformers.",
    )
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing (fallback path only; Unsloth always uses its own)")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--packing", action="store_true", default=True)
    parser.add_argument("--no_packing", action="store_false", dest="packing")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="0 recommended for Gemma-3 based models")
    parser.add_argument("--lora_target_modules", type=str, default=None)
    parser.add_argument("--merge_lora", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    main(args)
