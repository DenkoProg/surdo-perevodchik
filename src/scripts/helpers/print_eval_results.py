#!/usr/bin/env python3
RESULTS = {
    "MamayLM (baseline, before finetuning)": {
        "boiko_translation":          {"bleu": 21.77, "chrf++": 58.73, "ter": 56.42},
        "hutsul_translation":         {"bleu":  5.06, "chrf++": 16.73, "ter": 50.13},
        "surzhyk_translation":        {"bleu": 49.07, "chrf++": 65.58, "ter": 35.89},
        "transcarpathian_translation":{"bleu": 10.43, "chrf++": 27.90, "ter": 82.21},
    },
    "MamayLM (finetuned, decoder-only)": {
        "boiko_translation":          {"bleu": 51.81, "chrf++": 66.82, "ter": 35.26},
        "hutsul_translation":         {"bleu": 64.84, "chrf++": 74.02, "ter": 12.53},
        "surzhyk_translation":        {"bleu": 78.48, "chrf++": 91.22, "ter": 11.96},
        "transcarpathian_translation":{"bleu":  3.82, "chrf++": 20.93, "ter": 89.69},
    },
    "UMT5-base (finetuned, encoder-decoder)": {
        "boiko_translation":          {"bleu": 29.12, "chrf++": 54.78, "ter": 49.37},
        "hutsul_translation":         {"bleu": 74.68, "chrf++": 86.90, "ter": 12.53},
        "surzhyk_translation":        {"bleu": 89.42, "chrf++": 93.58, "ter":  5.98},
        "transcarpathian_translation":{"bleu":  6.20, "chrf++": 27.45, "ter": 82.21},
    },
}

T, M = 28, 8

def print_table(model_name, tasks):
    print(f"\n{model_name}\n")
    print(f"|{'Tasks':<{T}}|{'bleu':>{M}}|{'chrf++':>{M}}|{'ter':>{M}}|")
    print(f"|{'-'*T}|{'-'*M}:|{'-'*M}:|{'-'*M}:|")
    for task, m in tasks.items():
        print(f"|{task:<{T}}|{m['bleu']:>{M}.2f}|{m['chrf++']:>{M}.2f}|{m['ter']:>{M}.2f}|")
    print()

for model_name, tasks in RESULTS.items():
    print_table(model_name, tasks)
