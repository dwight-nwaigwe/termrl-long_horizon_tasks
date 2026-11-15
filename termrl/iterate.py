"""The RL loop on one GPU pod. Two algorithms, chosen with `algorithm:` in the config.

expert_iteration (configs/default.yaml)
  Each iteration: serve the current model, attempt every training task k times,
  LoRA-SFT on the model's own successes, merge -> next model. Learns only from successes.

dppo (configs/dppo.yaml), the algorithm tmax uses
  Each iteration: serve base model + current LoRA adapter, attempt a batch of tasks
  k times while recording each token's rollout log-prob, then take a few DPPO
  optimizer steps on the adapter. Learns from successes *and* failures:
  advantage = reward - mean reward of the task's k attempts. The adapter and its
  optimizer state carry over between iterations; it's merged into the base at the end.

Both: vLLM is stopped before training so they can share one GPU; re-running the
same command resumes after the last finished iteration.

Usage:  python -m termrl.iterate configs/default.yaml   |   configs/dppo.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
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


def make_server(cfg: dict, model: str, it_dir: Path, lora_path: str | None = None, rl: bool = False) -> VLLMServer:
    v = cfg["vllm"]
    return VLLMServer(
        model, port=v["port"], version=v["version"], gpu_memory_utilization=v["gpu_memory_utilization"],
        max_model_len=v["max_model_len"], tensor_parallel_size=v["tensor_parallel_size"],
        extra_args=v.get("extra_args", []), log_path=it_dir / "vllm.log",
        lora_path=lora_path, max_lora_rank=cfg.get("dppo", {}).get("lora_r", 64), plain_sampling=rl,
    )


def collect(cfg: dict, server: VLLMServer, it_dir: Path, train_tasks, eval_tasks, pool, logprobs: bool = False) -> dict:
    llm = make_llm(server.base_url, server.served_name, no_thinking=cfg.get("no_thinking", False), logprobs=logprobs)
    ep = dict(max_turns=cfg["max_turns"], max_tokens=cfg["max_tokens"], command_timeout=cfg["command_timeout"], model_name=str(it_dir))
    out = {}
    with server:
        if eval_tasks:
            recs = asyncio.run(run_rollouts(eval_tasks, llm, pool, k=cfg["eval_k"], out_path=it_dir / "eval.jsonl", temperature=cfg["eval_temperature"], **ep))
            out["eval"] = summarize(recs)
            print_summary(out["eval"], f"{it_dir.name} eval")
        if train_tasks:
            recs = asyncio.run(run_rollouts(train_tasks, llm, pool, k=cfg["k"], out_path=it_dir / "train_rollouts.jsonl", temperature=cfg["temperature"], **ep))
            out["train"] = summarize(recs)
            print_summary(out["train"], f"{it_dir.name} train rollouts")
    return out


def last_done(run_dir: Path, iterations: int) -> tuple[int, dict]:
    start, state = 0, {}
    for i in range(iterations):
        f = run_dir / f"iter_{i}" / "done.json"
        if f.exists():
            start, state = i + 1, json.loads(f.read_text())
    return start, state


# ----------------------------------------------------------------------------- expert iteration
def run_expert_iteration(cfg, run_dir, metrics, train_tasks, eval_tasks, pool) -> str:
    model_path = cfg["base_model"]
    start_it, state = last_done(run_dir, cfg["iterations"])
    if start_it:
        model_path = state["model"]
        print(f"Resuming from iteration {start_it} with {model_path}")

    for i in range(start_it, cfg["iterations"]):
        it_dir = run_dir / f"iter_{i}"
        it_dir.mkdir(exist_ok=True)
        t0 = time.time()
        print(f"\n######## Iteration {i}: {model_path}", flush=True)
        res = collect(cfg, make_server(cfg, model_path, it_dir), it_dir, train_tasks, eval_tasks, pool)

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
            return model_path

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
        (it_dir / "done.json").write_text(json.dumps({"model": model_path}))
        print(f"Iteration {i} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return model_path


# ----------------------------------------------------------------------------- DPPO
def task_batch(order: list, i: int, n: int) -> list:
    n = min(n, len(order))
    return [order[(i * n + j) % len(order)] for j in range(n)]


def run_dppo(cfg, run_dir, metrics, train_tasks, eval_tasks, pool) -> str:
    d = cfg["dppo"]
    base = cfg["base_model"]
    order = list(train_tasks)
    random.Random(cfg.get("seed", 0)).shuffle(order)
    eval_every = cfg.get("eval_every", 5)

    start_it, state = last_done(run_dir, cfg["iterations"])
    adapter = state.get("adapter")
    if start_it:
        print(f"Resuming from iteration {start_it} with adapter {adapter}")

    for i in range(start_it, cfg["iterations"]):
        it_dir = run_dir / f"iter_{i}"
        it_dir.mkdir(exist_ok=True)
        t0 = time.time()
        batch = task_batch(order, i, d["tasks_per_iter"])
        do_eval = bool(eval_tasks) and i % eval_every == 0
        print(f"\n######## DPPO iteration {i}: adapter={adapter}, {len(batch)} tasks x {cfg['k']}", flush=True)
        res = collect(cfg, make_server(cfg, base, it_dir, lora_path=adapter, rl=True), it_dir,
                      batch, eval_tasks if do_eval else [], pool, logprobs=True)

        out = it_dir / "adapter"
        cmd = [
            sys.executable, "-m", "termrl.train_dppo",
            "--base-model", base, "--rollouts", str(it_dir / "train_rollouts.jsonl"), "--adapter-out", str(out),
            "--divergence", d["divergence"], "--lr", str(d["lr"]), "--updates", str(d["updates"]),
            "--max-length", str(d["max_length"]), "--lora-r", str(d["lora_r"]), "--lora-alpha", str(d["lora_alpha"]),
            "--seed", str(i),
        ]
        if d.get("delta") is not None:
            cmd += ["--delta", str(d["delta"])]
        if adapter:
            cmd += ["--adapter-in", adapter]
        if cfg.get("no_thinking"):
            cmd += ["--no-thinking"]
        print("Training:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        if (out / "adapter_config.json").exists():
            adapter = str(out)
        else:
            print("No update this iteration (no task had both successes and failures).")
        log_steps = [json.loads(l) for l in (out / "dppo_log.jsonl").read_text().splitlines()] if (out / "dppo_log.jsonl").exists() else []
        log_metrics(metrics, {"iteration": i, "adapter": adapter, **res, "dppo_steps": log_steps})
        (it_dir / "done.json").write_text(json.dumps({"adapter": adapter}))
        print(f"Iteration {i} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    if not adapter:
        print("No adapter was trained.")
        return base
    final_merged = run_dir / "final" / "merged"
    marker = run_dir / "final" / "merged_from.txt"
    if not (final_merged / "config.json").exists() or not marker.exists() or marker.read_text() != adapter:
        shutil.rmtree(final_merged, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "termrl.merge", "--base", base, "--adapter", adapter, "--out", str(final_merged)], check=True)
        marker.write_text(adapter)
    return str(final_merged)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    algo = cfg.get("algorithm", "expert_iteration")
    if algo not in ("expert_iteration", "dppo"):
        sys.exit(f"unknown algorithm {algo!r}; use expert_iteration or dppo")

    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics = run_dir / "metrics.jsonl"

    train_tasks = load_tasks(*cfg["train_tasks"])
    eval_tasks = load_tasks(*cfg["eval_tasks"]) if cfg.get("eval_tasks") else []
    overlap = {t.id for t in train_tasks} & {t.id for t in eval_tasks}
    if overlap:
        sys.exit(f"train and eval tasks overlap: {sorted(overlap)[:5]}")
    print(f"{algo}: {len(train_tasks)} train tasks, {len(eval_tasks)} eval tasks")
    pool = UserPool(cfg["concurrency"])

    runner = run_dppo if algo == "dppo" else run_expert_iteration
    model_path = runner(cfg, run_dir, metrics, train_tasks, eval_tasks, pool)

    final_dir = run_dir / "final"
    final_dir.mkdir(exist_ok=True)
    done_marker = final_dir / "eval_done.txt"
    tag = f"{model_path}@{cfg['iterations']}"
    if eval_tasks and not (done_marker.exists() and done_marker.read_text() == tag):
        (final_dir / "eval.jsonl").unlink(missing_ok=True)
        # Same sampling settings as the per-iteration evals, so the numbers are comparable.
        res = collect(cfg, make_server(cfg, model_path, final_dir, rl=(algo == "dppo")), final_dir, [], eval_tasks, pool)
        log_metrics(metrics, {"iteration": "final", "model": model_path, **res})
        done_marker.write_text(tag)
    print(f"\nFinal model: {model_path}\nMetrics: {metrics}")


if __name__ == "__main__":
    main()
