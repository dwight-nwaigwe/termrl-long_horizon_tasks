"""Turn successful rollouts into an SFT dataset.

Each successful trajectory becomes one example per assistant turn, in TRL's
conversational prompt/completion format, so loss is only on the model's replies:

    {"prompt": [system, user, assistant, user, ...], "completion": [{"role": "assistant", ...}]}

Example
-------
python -m termrl.build_sft runs/teacher.jsonl --out runs/sft/teacher.jsonl --max-per-task 2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .rollout import read_jsonl


def build(
    rollout_files: list[str | Path],
    out: str | Path,
    max_per_task: int = 2,
    skip_easy: bool = False,
    require_submit: bool = True,
) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for f in rollout_files:
        for r in read_jsonl(f):
            if r.get("reward") is not None:
                by_task[r["task_id"]].append(r)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_traj = n_examples = n_skipped_easy = 0
    with out.open("w") as f:
        for tid, recs in sorted(by_task.items()):
            wins = [r for r in recs if r["reward"] == 1.0 and (r["finish"] == "submitted" or not require_submit)]
            if not wins:
                continue
            if skip_easy and len(wins) == len(recs) and len(recs) > 1:
                n_skipped_easy += 1
                continue
            # Prefer short, clean trajectories.
            wins.sort(key=lambda r: (r.get("n_format_errors", 0), r.get("n_turns", 0)))
            for r in wins[:max_per_task]:
                n_traj += 1
                msgs, meta = r["messages"], r.get("turn_meta", [])
                a_idx = 0
                for i, m in enumerate(msgs):
                    if m["role"] != "assistant":
                        continue
                    bad = a_idx < len(meta) and meta[a_idx].get("format_error")
                    a_idx += 1
                    if bad:
                        continue
                    f.write(json.dumps({"prompt": msgs[:i], "completion": [m], "task_id": tid}) + "\n")
                    n_examples += 1
    stats = {"tasks_seen": len(by_task), "trajectories": n_traj, "examples": n_examples, "skipped_easy_tasks": n_skipped_easy}
    print(f"SFT data -> {out}: {stats}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts", nargs="+", help="rollout .jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-task", type=int, default=2)
    ap.add_argument("--skip-easy", action="store_true", help="drop tasks the model already solves every time")
    ap.add_argument("--allow-unsubmitted", action="store_true", help="also keep passing episodes that never sent <submit/>")
    args = ap.parse_args()
    build(args.rollouts, args.out, args.max_per_task, args.skip_easy, not args.allow_unsubmitted)


if __name__ == "__main__":
    main()
