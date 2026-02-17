# Task: Rent Vast.ai GPU for Lapa LLM Fine-Tuning

**Goal**: Find and rent the best performance/price GPU instance on Vast.ai for QLoRA fine-tuning of Lapa LLM (12B, Gemma-3-12B based) on Ukrainian dialect normalization data.

---

## Lapa LLM Specs

| Property | Value |
|----------|-------|
| **Model** | `lapa-llm/lapa-v0.1.2-instruct` (or `lapa-llm/lapa-12b-pt` for base) |
| **Parameters** | 12B |
| **Architecture** | Gemma 3, BF16 |
| **Context** | 128K input / 8K output |
| **License** | Gemma (Apache 2.0 based) |
| **HuggingFace** | https://huggingface.co/lapa-llm |
| **GitHub** | https://github.com/lapa-llm/lapa-llm |

---

## VRAM Requirements (12B model)

| Method | VRAM | Cost Efficiency |
|--------|------|-----------------|
| **QLoRA (4-bit) + Unsloth** | ~16-24 GB | Best — recommended |
| **LoRA (BF16 base)** | ~30-40 GB | Good quality, higher cost |
| **Full fine-tune** | ~80-100+ GB | Multi-GPU, expensive |

---

## Recommended Vast.ai Instances

### Tier 1: Best Value (QLoRA)

| GPU | VRAM | ~$/hr | Use Case |
|-----|------|-------|----------|
| **RTX 3090** | 24 GB | $0.08 | **Best price/perf** — QLoRA 12B fits easily |
| **RTX 4090** | 24 GB | $0.20-0.35 | ~2x faster than 3090, same VRAM |

### Tier 2: LoRA / Larger Batch Sizes

| GPU | VRAM | ~$/hr | Use Case |
|-----|------|-------|----------|
| **A40** | 48 GB | $0.29 | LoRA without quantization, large batches |
| **RTX A6000** | 48 GB | $0.39 | Same 48 GB, slightly faster |

### Tier 3: Full Fine-Tune / Maximum Speed

| GPU | VRAM | ~$/hr | Use Case |
|-----|------|-------|----------|
| **A100 SXM** | 80 GB | $0.67 | Full fine-tune on single card |
| **H100** | 80 GB | $1.55 | 2-3x faster than A100 |

---

## Primary Recommendation

**RTX 3090 (24 GB) at ~$0.08/hr** with QLoRA + Unsloth:

- Unsloth supports Gemma 3, gives ~1.6x speedup + 60% VRAM savings
- A multi-hour training run costs under $1
- Use **interruptible instances** to cut price to ~$0.04/hr
- Sufficient for QLoRA with batch_size=2, grad_accum=4, rank=32

---

## Vast.ai Instance Search Filters

```
VRAM:       ≥ 24 GB
GPU:        RTX 3090 / RTX 4090 (or A40/A6000 for LoRA)
RAM:        ≥ 32 GB
Disk:       ≥ 50 GB, NVMe (>1 GB/s)
CUDA:       ≥ 8.0
Rental:     Interruptible (for savings) or On-Demand (for reliability)
Verified:   Preferred (ISO 27001 / Uptime Tier 3+)
```

---

## Training Setup Checklist

- [ ] Rent Vast.ai instance (RTX 3090 interruptible)
- [ ] Install dependencies: `unsloth`, `transformers`, `peft`, `bitsandbytes`, `wandb`
- [ ] Download model: `lapa-llm/lapa-v0.1.2-instruct` (or `lapa-12b-pt`)
- [ ] Prepare dataset in instruction format (see `tasks.md` Phase 3.1)
- [ ] Configure QLoRA: r=16, alpha=32, target_modules=[q,k,v,o,gate,up,down]
- [ ] Configure Unsloth for Gemma 3 (1.6x speedup, 60% VRAM reduction)
- [ ] Train with checkpointing enabled (critical for interruptible instances)
- [ ] Monitor with W&B
- [ ] Evaluate on held-out dialect test sets

---

## Cost Estimates

| Scenario | GPU | Hours | Cost |
|----------|-----|-------|------|
| Quick experiment (1 epoch) | RTX 3090 interruptible | ~2-3h | ~$0.10-0.15 |
| Full training (3 epochs) | RTX 3090 interruptible | ~6-10h | ~$0.25-0.50 |
| Full training (3 epochs) | RTX 4090 on-demand | ~3-5h | ~$0.60-1.75 |
| Full training (3 epochs) | A100 SXM on-demand | ~2-3h | ~$1.35-2.00 |

---

## Key Links

| Resource | URL |
|----------|-----|
| Vast.ai Search | https://vast.ai/search |
| Lapa LLM (HF) | https://huggingface.co/lapa-llm |
| Lapa LLM (GitHub) | https://github.com/lapa-llm/lapa-llm |
| Unsloth (Gemma 3 support) | https://unsloth.ai/blog/gemma3 |
| Vast.ai CLI Docs | https://vast.ai/docs/cli/commands |
