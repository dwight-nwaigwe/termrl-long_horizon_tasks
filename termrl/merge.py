"""Merge a LoRA adapter into its base model so vLLM (or anything else) can load it as a plain model.

python -m termrl.merge --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/dppo/iter_9/adapter --out runs/dppo/final/merged
"""

from __future__ import annotations

import argparse


def merge(base: str, adapter: str, out: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print(f"Merged model saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(a.base, a.adapter, a.out)


if __name__ == "__main__":
    main()
