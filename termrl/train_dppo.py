"""One DPPO update on a batch of rollouts (called by iterate.py with algorithm: dppo).

The policy is the base model plus a LoRA adapter that persists across iterations
(together with its AdamW state). Each call:

  1. groups rollouts by task, A = reward - group mean (groups with no variance are skipped)
  2. turns every assistant turn into a sample: prompt ids, generated ids, rollout log-probs
  3. splits the samples into --updates mini-batches and takes one optimizer step per
     mini-batch with the DPPO loss (see termrl/dppo.py), log-probs from an fp32 LM head
  4. saves the adapter + optimizer state to --adapter-out

Example (normally run by iterate.py):
python -m termrl.train_dppo --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --rollouts runs/dppo/iter_0/train_rollouts.jsonl --adapter-out runs/dppo/iter_0/adapter
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .dppo import build_samples, dppo_mask_and_ratio, group_advantages
from .rollout import read_jsonl


def enable_checkpointing(model) -> None:
    for name in ("gradient_checkpointing_enable", "activation_checkpointing_enable"):
        fn = getattr(model, name, None)
        if fn is not None:
            try:
                fn(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                fn()
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--rollouts", required=True, nargs="+")
    ap.add_argument("--adapter-in", default=None, help="adapter (+ optimizer.pt) from the previous iteration")
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--divergence", choices=["tv", "kl"], default="tv")
    ap.add_argument("--delta", type=float, default=None, help="trust-region threshold (default 0.15 for tv, 0.05 for kl)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--updates", type=int, default=4, help="optimizer steps per batch of rollouts")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--no-thinking", action="store_true", help="match a vLLM server called with enable_thinking=False")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    delta = args.delta if args.delta is not None else (0.15 if args.divergence == "tv" else 0.05)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.adapter_out)
    records = [r for f in args.rollouts for r in read_jsonl(f)]
    adv_records, gstats = group_advantages(records)
    print(f"Rollouts: {len(records)} episodes, {gstats}")
    if not adv_records:
        print("No task had both successes and failures: nothing to learn from this batch. Adapter unchanged.")
        return

    tok = AutoTokenizer.from_pretrained(args.base_model)
    chat_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    samples, sstats = build_samples(adv_records, tok, args.max_length, chat_kwargs)
    print(f"Samples: {sstats}")
    if not samples:
        print("No usable samples. Adapter unchanged.")
        return

    device = "cuda"
    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
    base.config.use_cache = False
    if args.adapter_in:
        model = PeftModel.from_pretrained(base, args.adapter_in, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                                target_modules="all-linear", task_type="CAUSAL_LM"))
    enable_checkpointing(model)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    if args.adapter_in and (Path(args.adapter_in) / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(Path(args.adapter_in) / "optimizer.pt", map_location=device))
        for g in opt.param_groups:
            g["lr"] = args.lr

    # Log-probs from an fp32 LM head (bf16 logits make the ratio noisy; tmax also uses an fp32 head).
    inner = model.get_base_model()
    backbone = inner.get_decoder() if hasattr(inner, "get_decoder") else inner.model
    w32 = inner.lm_head.weight.detach().float()
    b32 = inner.lm_head.bias.detach().float() if getattr(inner.lm_head, "bias", None) is not None else None

    def token_logprobs(s: dict) -> "torch.Tensor":
        ids = torch.tensor([s["prompt"] + s["completion"]], device=device)
        n = len(s["completion"])
        hidden = backbone(input_ids=ids).last_hidden_state[0, -n - 1 : -1]  # positions predicting the completion
        logits = torch.nn.functional.linear(hidden.float(), w32, b32)
        return torch.log_softmax(logits, dim=-1).gather(-1, ids[0, -n:, None]).squeeze(-1)

    # Turns without rollout log-probs fall back to the current policy as the anchor.
    missing = [s for s in samples if s["mu"] is None]
    if missing:
        print(f"Computing anchor log-probs for {len(missing)} turns without rollout log-probs")
        model.eval()
        with torch.no_grad():
            for s in missing:
                s["mu"] = token_logprobs(s).tolist()

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_up = max(1, min(args.updates, len(samples)))
    batches = [samples[i::n_up] for i in range(n_up)]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = (out_dir / "dppo_log.jsonl").open("w")
    model.train()
    t0 = time.time()
    for step, batch in enumerate(batches):
        n_tok = sum(len(s["completion"]) for s in batch)
        opt.zero_grad(set_to_none=True)
        tot = {"loss": 0.0, "masked": 0.0, "ratio": 0.0, "abs_logp_diff": 0.0, "div": 0.0}
        for s in batch:
            logp = token_logprobs(s)
            mu = torch.tensor(s["mu"], device=device, dtype=torch.float32)
            if mu.shape != logp.shape:
                continue
            ratio, mask, d = dppo_mask_and_ratio(
                logp, mu, s["adv"], delta, args.divergence, torch.exp, torch.log, torch.clamp
            )
            loss = -(mask * ratio * s["adv"]).sum() / n_tok
            loss.backward()
            with torch.no_grad():
                tot["loss"] += loss.item()
                tot["masked"] += (1 - mask).sum().item()
                tot["ratio"] += ratio.detach().sum().item()
                tot["abs_logp_diff"] += (logp.detach() - mu).abs().sum().item()
                tot["div"] += d.detach().sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
        opt.step()
        row = {
            "step": step, "sequences": len(batch), "tokens": n_tok, "loss": tot["loss"],
            "masked_frac": tot["masked"] / n_tok, "mean_ratio": tot["ratio"] / n_tok,
            "mean_abs_logp_diff": tot["abs_logp_diff"] / n_tok, "mean_divergence": tot["div"] / n_tok,
            "grad_norm": grad_norm, "seconds": round(time.time() - t0, 1),
        }
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(json.dumps(row), flush=True)
        if step == 0 and row["mean_abs_logp_diff"] > 0.1:
            print("WARNING: training and rollout log-probs differ a lot before any update. The prompt the "
                  "trainer rebuilds may not match what vLLM saw (check no_thinking / chat template).", flush=True)
    log_f.close()

    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    torch.save(opt.state_dict(), out_dir / "optimizer.pt")
    (out_dir / "batch_stats.json").write_text(json.dumps({**gstats, **sstats}))
    print(f"Adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
