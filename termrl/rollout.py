"""Run k episodes per task against an OpenAI-compatible endpoint and report pass@k.

Examples
--------
# Evaluate the policy served by scripts/serve.sh
python -m termrl.rollout --tasks tasks/seed --k 4 --out runs/baseline/seed.jsonl

# Collect teacher trajectories for a warm-start SFT
python -m termrl.rollout --tasks tasks/generated --k 2 --out runs/teacher.jsonl \
    --base-url "$TEACHER_BASE_URL" --model "$TEACHER_MODEL" --api-key-env TEACHER_API_KEY

# Re-print the summary of an existing file
python -m termrl.rollout --summarize runs/baseline/seed.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

from .agent import ChatModel, run_episode
from .sandbox import UserPool
from .tasks import Task, load_tasks


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarize(records: list[dict]) -> dict:
    by_task: dict[str, list[float]] = defaultdict(list)
    finishes = Counter()
    turns = []
    for r in records:
        finishes[r.get("finish")] += 1
        if r.get("reward") is None:
            continue
        by_task[r["task_id"]].append(r["reward"])
        turns.append(r.get("n_turns", 0))
    if not by_task:
        return {"n_tasks": 0, "n_episodes": 0, "finishes": dict(finishes)}
    n_min = min(len(v) for v in by_task.values())
    out = {
        "n_tasks": len(by_task),
        "n_episodes": sum(len(v) for v in by_task.values()),
        "pass@1": sum(sum(v) / len(v) for v in by_task.values()) / len(by_task),
        "solved_any": sum(1 for v in by_task.values() if sum(v) > 0) / len(by_task),
        "solved_all": sum(1 for v in by_task.values() if sum(v) == len(v)) / len(by_task),
        "mean_turns": sum(turns) / len(turns),
        "finishes": dict(finishes),
    }
    if n_min > 1:
        out[f"pass@{n_min}"] = sum(pass_at_k(len(v), int(sum(v)), n_min) for v in by_task.values()) / len(by_task)
    return out


def print_summary(s: dict, title: str = "Summary") -> None:
    print(f"\n== {title} ==")
    for k, v in s.items():
        print(f"  {k:>12}: {v:.3f}" if isinstance(v, float) else f"  {k:>12}: {v}")


async def run_rollouts(
    tasks: list[Task],
    llm: ChatModel,
    pool: UserPool,
    *,
    k: int,
    out_path: str | Path,
    resume: bool = True,
    **episode_kwargs,
) -> list[dict]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(out_path) if resume else []
    done = {(r["task_id"], r["sample_idx"]) for r in existing if r.get("finish") != "llm_error"}
    jobs = [(t, i) for t in tasks for i in range(k) if (t.id, i) not in done]
    if not resume and out_path.exists():
        out_path.unlink()
    print(f"{len(jobs)} episodes to run ({len(done)} already done), concurrency {pool.size}", flush=True)

    lock = asyncio.Lock()
    results: list[dict] = [r for r in existing if (r["task_id"], r["sample_idx"]) in done]
    counter = {"n": 0, "ok": 0}
    t0 = time.time()

    async def one(task: Task, idx: int):
        try:
            rec = await run_episode(task, llm, pool, sample_idx=idx, **episode_kwargs)
        except Exception as e:  # noqa: BLE001
            rec = {"task_id": task.id, "sample_idx": idx, "reward": None, "finish": "crash", "error": repr(e), "messages": []}
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            results.append(rec)
            counter["n"] += 1
            counter["ok"] += int(rec.get("reward") == 1.0)
            n = counter["n"]
            if n % 10 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} episodes, {counter['ok']} solved, {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*(one(t, i) for t, i in jobs))
    return results


def add_llm_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--base-url", default=os.environ.get("POLICY_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("POLICY_MODEL", "policy"))
    ap.add_argument("--api-key-env", default=None, help="name of the env var holding the API key")
    ap.add_argument("--no-thinking", action="store_true", help="send enable_thinking=False (Qwen3-style models)")
    ap.add_argument("--max-tokens-field", default="max_tokens", help="use max_completion_tokens for newer OpenAI models")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", help="task directories")
    ap.add_argument("--out", help="output .jsonl")
    ap.add_argument("--k", type=int, default=4, help="episodes per task")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--concurrency", type=int, default=16, help="parallel episodes (needs that many sandbox users)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=2048, help="per reply")
    ap.add_argument("--command-timeout", type=float, default=60)
    ap.add_argument("--fresh", action="store_true", help="overwrite --out instead of resuming")
    ap.add_argument("--summarize", metavar="JSONL", help="only print the summary of an existing file")
    add_llm_args(ap)
    args = ap.parse_args()

    if args.summarize:
        print_summary(summarize(read_jsonl(args.summarize)), args.summarize)
        return
    if not args.tasks or not args.out:
        ap.error("--tasks and --out are required")

    from .llm import make_llm

    tasks = load_tasks(*args.tasks)[: args.limit]
    llm = make_llm(args.base_url, args.model, args.api_key_env, args.no_thinking, max_tokens_field=args.max_tokens_field)
    records = asyncio.run(
        run_rollouts(
            tasks, llm, UserPool(args.concurrency),
            k=args.k, out_path=args.out, resume=not args.fresh,
            max_turns=args.max_turns, temperature=args.temperature, max_tokens=args.max_tokens,
            command_timeout=args.command_timeout, model_name=args.model,
        )
    )
    print_summary(summarize(records), args.out)


if __name__ == "__main__":
    main()
