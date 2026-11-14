"""Synthesize terminal tasks with a teacher LLM, then keep only tasks that validate.

Like tmax, tasks are diversified by sampling a domain, a skill, a persona and a
difficulty for each one. Every generated task is executed in the sandbox:
setup must work, the verifier must FAIL on the initial state and PASS after the
teacher's reference solution. Rejected tasks go to <out>/_rejected/ with a reason.

Example
-------
export TEACHER_BASE_URL=https://api.openai.com/v1 TEACHER_MODEL=gpt-4.1 TEACHER_API_KEY=sk-...
python -m termrl.generate_tasks --n 300 --out tasks/generated \
    --base-url "$TEACHER_BASE_URL" --model "$TEACHER_MODEL" --api-key-env TEACHER_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import yaml

from .llm import make_llm
from .rollout import add_llm_args
from .sandbox import UserPool
from .tasks import load_task, validate_task

DOMAINS = [
    "log analysis", "CSV and JSON data wrangling", "git repository workflows", "debugging a small Python project",
    "writing reusable bash scripts", "organizing files and directories", "text processing and reports",
    "SQLite databases", "archives and backups", "editing configuration files (INI/YAML/TOML/env)",
    "Makefiles and builds", "scheduled-job and automation scripts", "file permissions and secrets hygiene",
    "scientific or numeric data processing", "web server config and access logs (offline)",
    "package/dependency manifests", "unit-testing and fixing failing tests", "data validation and cleaning",
]
SKILLS = [
    "grep/sed/awk", "find and xargs", "python3 standard-library scripting", "git", "jq", "sqlite3",
    "tar/gzip/zip", "regular expressions", "sort/uniq/join/comm", "bash scripting with arguments and error handling",
    "make", "chmod and file modes", "diff and patch", "date and time arithmetic", "checksums (sha256sum/md5sum)",
]
PERSONAS = [
    "a data analyst in a hurry", "a junior sysadmin", "a PhD student cleaning experiment output",
    "an open-source maintainer", "an SRE on call at 3am", "a bookkeeper at a small business",
    "a game modder", "a teacher automating grading", "a security engineer doing an audit",
    "a hobbyist photographer", "a backend developer", "a technical writer managing docs",
]
DIFFICULTY = {
    "easy": "solvable with 1-3 commands by someone comfortable with the shell",
    "medium": "needs 4-10 commands combining several tools, or a short script",
    "hard": "needs investigation, several steps and a script of 30+ lines or careful debugging",
}

PROMPT = """You design training tasks for an AI agent that operates a Linux bash shell as an UNPRIVILEGED user (no root, no sudo, no apt, no Docker). Available tools: coreutils, bash, python3 (standard library only), git, sqlite3, jq, tar, gzip, zip/unzip, make, diffutils, bc. Assume there is NO internet access.

Create ONE task with these properties:
- Domain: {domain}
- Main skill: {skill}
- Written by: {persona}
- Difficulty: {difficulty} ({difficulty_desc})

Return ONLY a JSON object (no markdown fences) with these string keys:
"slug": short kebab-case name for the task.
"instruction": the request as the persona would write it. It must name exact files, formats and locations so the result can be checked automatically. Relative paths are relative to the agent's starting directory. Do not mention setup, verification or the reference solution.
"setup_sh": bash script run as the agent user in its starting directory BEFORE the agent starts. It creates all input files. It must be deterministic (fixed random seeds, no current dates or times) and self-contained.
"solution_sh": bash script that correctly solves the task starting from the state setup_sh creates.
"verify_sh": bash script run in the starting directory AFTER the agent finishes. Exit 0 only if the task is solved, non-zero otherwise, printing a short reason on failure. It must fail on the initial state. Never trust files the agent could edit to compute the expected answer: recompute expected values independently (e.g. regenerate the input data inside verify_sh with the same seed). Accept any correct solution, not only the reference one (ignore trailing whitespace, accept any order unless the instruction requires one). Use `python3 -I` for Python-based checks."""


def sample_spec(rng: random.Random) -> dict:
    diff = rng.choices(list(DIFFICULTY), weights=[3, 4, 3])[0]
    return {
        "domain": rng.choice(DOMAINS),
        "skill": rng.choice(SKILLS),
        "persona": rng.choice(PERSONAS),
        "difficulty": diff,
        "difficulty_desc": DIFFICULTY[diff],
    }


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in reply")
    raw = text[start : end + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Teachers often forget to double backslashes in shell/regex code (e.g. "\." or "\d").
        # Keep valid JSON escapes, double every other backslash, and try again.
        fixed = re.sub(
            r'\\(u[0-9a-fA-F]{4}|.)',
            lambda m: m.group(0) if m.group(1)[0] in '"\\/bfnrt' or len(m.group(1)) == 5 else "\\\\" + m.group(1),
            raw,
            flags=re.S,
        )
        obj = json.loads(fixed)
    for key in ("slug", "instruction", "setup_sh", "solution_sh", "verify_sh"):
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise ValueError(f"missing or empty key {key!r}")
    return obj


def write_task(obj: dict, spec: dict, out: Path) -> Path:
    slug = re.sub(r"[^a-z0-9-]+", "-", obj["slug"].lower()).strip("-")[:50] or "task"
    digest = hashlib.sha1(obj["instruction"].encode()).hexdigest()[:8]
    tid = f"{slug}-{digest}"
    d = out / tid
    d.mkdir(parents=True, exist_ok=True)
    meta = {"id": tid, "instruction": obj["instruction"].strip(), "tags": [spec["difficulty"], spec["domain"], spec["skill"]]}
    (d / "task.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100))
    (d / "setup.sh").write_text(obj["setup_sh"].rstrip() + "\n")
    (d / "solution.sh").write_text(obj["solution_sh"].rstrip() + "\n")
    (d / "verify.sh").write_text(obj["verify_sh"].rstrip() + "\n")
    return d


async def generate(args) -> None:
    out = Path(args.out)
    rejected = out / "_rejected"
    out.mkdir(parents=True, exist_ok=True)
    llm = make_llm(args.base_url, args.model, args.api_key_env, args.no_thinking, max_tokens_field=args.max_tokens_field)
    pool = UserPool(args.users)
    sem = asyncio.Semaphore(args.concurrency)
    stats = {"ok": 0, "bad_json": 0, "invalid": 0}

    async def one(i: int):
        rng = random.Random(args.seed * 1_000_003 + i)
        spec = sample_spec(rng)
        obj = None
        async with sem:
            for attempt in range(1 + args.retries):
                try:
                    reply = await llm.chat([{"role": "user", "content": PROMPT.format(**spec)}], temperature=args.temperature, max_tokens=args.max_tokens)
                    obj = parse_json(reply.text)
                    break
                except Exception as e:  # noqa: BLE001
                    last = "retrying" if attempt < args.retries else "giving up"
                    print(f"[{i}] generation failed ({last}): {e!r}"[:300], flush=True)
        if obj is None:
            stats["bad_json"] += 1
            return
        d = write_task(obj, spec, out)
        ok, why = await validate_task(load_task(d), pool)
        if ok:
            stats["ok"] += 1
            print(f"[{i}] OK      {d.name}", flush=True)
        else:
            stats["invalid"] += 1
            rejected.mkdir(exist_ok=True)
            dest = rejected / d.name
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(d), dest)
            (dest / "rejected_reason.txt").write_text(why)
            print(f"[{i}] REJECT  {d.name}: {why[:150]}", flush=True)

    await asyncio.gather(*(one(i) for i in range(args.start, args.start + args.n)))
    print(f"\nDone: {stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="number of generation attempts")
    ap.add_argument("--start", type=int, default=0, help="first index (use to extend a corpus)")
    ap.add_argument("--out", default="tasks/generated")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8, help="parallel teacher requests")
    ap.add_argument("--users", type=int, default=8, help="sandbox users for validation")
    ap.add_argument("--retries", type=int, default=2, help="extra teacher requests when a reply can't be parsed")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    add_llm_args(ap)
    ap.set_defaults(
        base_url=os.environ.get("TEACHER_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("TEACHER_MODEL"),
        api_key_env="TEACHER_API_KEY",
    )
    args = ap.parse_args()
    if not args.model:
        ap.error("set --model or the TEACHER_MODEL environment variable")
    asyncio.run(generate(args))


if __name__ == "__main__":
    main()
