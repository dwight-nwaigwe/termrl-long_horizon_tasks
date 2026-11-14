"""LoRA SFT with TRL on prompt/completion data from build_sft.py, then merge the adapter.

The merged model is a normal Hugging Face model folder that vLLM can serve.

Example
-------
python -m termrl.train_sft --model Qwen/Qwen2.5-1.5B-Instruct \
    --data runs/sft/teacher.jsonl --output runs/models/warmstart
# -> runs/models/warmstart/adapter  and  runs/models/warmstart/merged
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local path of the model to fine-tune")
    ap.add_argument("--data", required=True, nargs="+", help="prompt/completion .jsonl file(s)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=8192, help="examples longer than this are dropped")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--report-to", default=os.environ.get("REPORT_TO", "none"), help='"wandb" to log to Weights & Biases')
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out = Path(args.output)
    adapter_dir, merged_dir = out / "adapter", out / "merged"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "completion")])

    def length(ex):
        text = tok.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False)
        return {"n_tokens": len(tok(text, add_special_tokens=False)["input_ids"])}

    ds = ds.map(length)
    before = len(ds)
    ds = ds.filter(lambda ex: ex["n_tokens"] <= args.max_length).remove_columns("n_tokens")
    print(f"Training examples: {len(ds)} (dropped {before - len(ds)} longer than {args.max_length} tokens)")
    if len(ds) == 0:
        raise SystemExit("No training examples left.")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")

    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        max_length=args.max_length,
        report_to=args.report_to,
        seed=args.seed,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    if args.no_merge:
        return
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
