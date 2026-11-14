"""Expert iteration (rejection-sampling fine-tuning) on one GPU pod.

Each iteration:
  1. serve the current model with vLLM
  2. evaluate it on the held-out tasks
  3. sample k attempts per training task
  4. stop vLLM (frees the GPU)
  5. LoRA-SFT the current model on its own successful attempts, merge -> next model

This is the simplest RL-style loop with a binary reward: it only learns from
successes (like REINFORCE with zero weight on failures). tmax uses DPPO, which
also learns from failures and needs a separate learner/inference setup.

Usage:  python -m termrl.iterate configs/default.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .build_sft import build
from .llm import make_llm
from .rollout import print_summary, run_rollouts, summarize
from .sandbox import UserPool
from .tasks import load_tasks
from .vllm_server import VLLMServer


def log_metrics(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def collect(cfg: dict, model_path: str, it_dir: Path, train_tasks, eval_tasks, pool, do_train_rollouts: bool) -> dict:
    v = cfg["vllm"]
    server = VLLMServer(
        model_path, port=v["port"], version=v["version"], gpu_memory_utilization=v["gpu_memory_utilization"],
        max_model_len=v["max_model_len"], tensor_parallel_size=v["tensor_parallel_size"],
        extra_args=v.get("extra_args", []), log_path=it_dir / "vllm.log",
    )
    llm = make_llm(server.base_url, server.served_name, no_thinking=cfg.get("no_thinking", False))
    ep = dict(max_turns=cfg["max_turns"], max_tokens=cfg["max_tokens"], command_timeout=cfg["command_timeout"], model_name=model_path)
    out = {}
    with server:
        if eval_tasks:
            recs = asyncio.run(run_rollouts(eval_tasks, llm, pool, k=cfg["eval_k"], out_path=it_dir / "eval.jsonl", temperature=cfg["eval_temperature"], **ep))
            out["eval"] = summarize(recs)
            print_summary(out["eval"], f"{it_dir.name} eval")
        if do_train_rollouts:
            recs = asyncio.run(run_rollouts(train_tasks, llm, pool, k=cfg["k"], out_path=it_dir / "train_rollouts.jsonl", temperature=cfg["temperature"], **ep))
            out["train"] = summarize(recs)
            print_summary(out["train"], f"{it_dir.name} train rollouts")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics = run_dir / "metrics.jsonl"

    train_tasks = load_tasks(*cfg["train_tasks"])
    eval_tasks = load_tasks(*cfg["eval_tasks"]) if cfg.get("eval_tasks") else []
    overlap = {t.id for t in train_tasks} & {t.id for t in eval_tasks}
    if overlap:
        sys.exit(f"train and eval tasks overlap: {sorted(overlap)[:5]}")
    print(f"{len(train_tasks)} train tasks, {len(eval_tasks)} eval tasks")
    pool = UserPool(cfg["concurrency"])

    model_path = cfg["base_model"]
    start_it = 0
    # Resume: skip iterations whose merged model already exists.
    for i in range(cfg["iterations"]):
        merged = run_dir / f"iter_{i}" / "model" / "merged"
        if (merged / "config.json").exists():
            model_path, start_it = str(merged), i + 1
    if start_it:
        print(f"Resuming from iteration {start_it} with {model_path}")

    for i in range(start_it, cfg["iterations"]):
        it_dir = run_dir / f"iter_{i}"
        it_dir.mkdir(exist_ok=True)
        t0 = time.time()
        print(f"\n######## Iteration {i}: {model_path}", flush=True)
        res = collect(cfg, model_path, it_dir, train_tasks, eval_tasks, pool, do_train_rollouts=True)

        stats = build(
            [it_dir / "train_rollouts.jsonl"], it_dir / "sft.jsonl",
            max_per_task=cfg["sft"]["max_per_task"], skip_easy=cfg["sft"].get("skip_easy", True),
        )
        log_metrics(metrics, {"iteration": i, "model": model_path, **res, "sft": stats})
        if stats["examples"] == 0:
            if stats["skipped_easy_tasks"]:
                print(f"Nothing new to learn: {stats['skipped_easy_tasks']} tasks are solved every time and the rest never. "
                      "Add harder tasks, or set sft.skip_easy: false.")
            else:
                print("No successful trajectories to learn from. Try easier tasks, a warm start, or a bigger model.")
            return

        s = cfg["sft"]
        cmd = [
            sys.executable, "-m", "termrl.train_sft",
            "--model", model_path, "--data", str(it_dir / "sft.jsonl"), "--output", str(it_dir / "model"),
            "--epochs", str(s["epochs"]), "--lr", str(s["lr"]), "--batch-size", str(s["batch_size"]),
            "--grad-accum", str(s["grad_accum"]), "--max-length", str(s["max_length"]),
            "--lora-r", str(s["lora_r"]), "--lora-alpha", str(s["lora_alpha"]),
        ]
        print("Training:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        model_path = str(it_dir / "model" / "merged")
        print(f"Iteration {i} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Final evaluation of the last model.
    final_dir = run_dir / "final"
    final_dir.mkdir(exist_ok=True)
    if eval_tasks:
        res = collect(cfg, model_path, final_dir, train_tasks, eval_tasks, pool, do_train_rollouts=False)
        log_metrics(metrics, {"iteration": "final", "model": model_path, **res})
    print(f"\nFinal model: {model_path}\nMetrics: {metrics}")


if __name__ == "__main__":
    main()
