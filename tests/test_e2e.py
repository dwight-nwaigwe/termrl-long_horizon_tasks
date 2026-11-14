"""End-to-end tests without a GPU: a scripted fake model drives the real agent loop and sandbox.

Needs root and sandbox users:  python -m termrl.sandbox setup --users 4
Run:                            python -m pytest -q tests
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

from termrl.agent import parse_action, run_episode
from termrl.build_sft import build
from termrl.llm import Reply
from termrl.rollout import run_rollouts, summarize
from termrl.sandbox import UserPool, session, user_exists
from termrl.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.geteuid() != 0 or not user_exists("termrl3"), reason="needs root + sandbox users")


class ScriptedLLM:
    """Replies with a fixed list of messages, in order (per conversation)."""

    def __init__(self, script_for_task):
        self.script_for_task = script_for_task

    async def chat(self, messages, temperature=1.0, max_tokens=2048):
        task_instruction = messages[1]["content"]
        n_assistant = sum(m["role"] == "assistant" for m in messages)
        script = self.script_for_task(task_instruction)
        return Reply(text=script[min(n_assistant, len(script) - 1)], finish_reason="stop")


def test_parse_action():
    assert parse_action("ok\n<bash>\nls -la\n</bash>") == ("bash", "ls -la")
    assert parse_action("<think>use <bash>x</bash></think>done <submit/>") == ("submit", "")
    assert parse_action("<bash>a</bash><bash>b</bash>")[0] == "error"
    assert parse_action("<bash>a</bash> <submit/>")[0] == "error"
    assert parse_action("I would run ls")[0] == "error"


def test_shell_state_and_isolation():
    async def go():
        pool = UserPool(2)
        async with session(pool) as sb:
            await sb.run("mkdir sub && cd sub && export X=42")
            r = await sb.run("pwd; echo $X; whoami")
            assert r.output.split()[0].endswith("/sub") and r.output.split()[1] == "42"
            assert r.output.split()[2].startswith("termrl")
            r = await sb.run("cat /root/.bashrc", timeout=5)
            assert r.exit_code != 0  # can't read root's files
            r = await sb.run("sleep 10", timeout=1)
            assert r.timed_out
            r = await sb.run("yes | head -c 100000")
            assert "omitted" in r.output and len(r.output) < 9000

    asyncio.run(go())


def test_reference_solutions_score_1_and_naive_scores_0(tmp_path):
    tasks = load_tasks(ROOT / "tasks/seed")
    by_instr = {t.instruction: t for t in tasks}

    def solver(instr):
        return ["I'll do it.\n<bash>\n" + by_instr[instr].solution + "</bash>", "Looks right.\n<submit/>"]

    out = tmp_path / "rollouts.jsonl"
    recs = asyncio.run(run_rollouts(tasks, ScriptedLLM(solver), UserPool(4), k=2, out_path=out, max_turns=5))
    s = summarize(recs)
    assert s["pass@1"] == 1.0, [(r["task_id"], r["verify_output"]) for r in recs if r["reward"] != 1.0]
    assert s["finishes"] == {"submitted": len(tasks) * 2}

    # SFT data: one example per assistant turn of each kept trajectory.
    stats = build([out], tmp_path / "sft.jsonl", max_per_task=1)
    assert stats["trajectories"] == len(tasks) and stats["examples"] == 2 * len(tasks)
    ex = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    assert ex["prompt"][0]["role"] == "system" and ex["completion"][0]["role"] == "assistant"

    # Naive answers must fail (the seed tasks contain traps).
    naive = {
        "count-errors": "grep -c ERROR logs/app.log > error_count.txt",
        "top-customer-sqlite": "python3 -c \"import sqlite3;print(sqlite3.connect('shop.db').execute('select c.name from customers c join orders o on o.customer_id=c.id group by c.id order by sum(amount_cents) desc limit 1').fetchone()[0])\" > top_customer.txt",
        "find-large-files": "(cd data && find . -type f -size +1000k | sed 's|^./||' | sort) > large_files.txt",
    }
    for tid, cmd in naive.items():
        task = next(t for t in tasks if t.id == tid)
        rec = asyncio.run(run_episode(task, ScriptedLLM(lambda _: [f"<bash>\n{cmd}\n</bash>", "<submit/>"]), UserPool(1), max_turns=4))
        assert rec["reward"] == 0.0, tid


def test_format_errors_and_max_turns():
    task = next(t for t in load_tasks(ROOT / "tasks/seed") if t.id == "count-errors")
    rec = asyncio.run(run_episode(task, ScriptedLLM(lambda _: ["I think I should count."]), UserPool(1), max_turns=3))
    assert rec["finish"] == "max_turns" and rec["n_format_errors"] == 3 and rec["reward"] == 0.0
