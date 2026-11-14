"""The agent loop: model writes one command per turn, we run it, show the output."""

from __future__ import annotations

import re
import time
from typing import Any, Protocol

from .llm import ContextTooLong, Reply
from .sandbox import CommandResult, UserPool, session
from .tasks import Task, prepare, verify

SYSTEM_PROMPT = """You are an expert Linux user. You solve tasks by running commands in a bash shell as an unprivileged user (no root, no sudo).

On every turn, think briefly about what to do next, then give exactly ONE shell command (it may span several lines) inside tags:
<bash>
your command here
</bash>

You will then see its exit code and output. The current directory and exported variables persist between commands. Each command times out after {timeout} seconds. Do not start interactive programs (vim, nano, less, top, python REPL); use non-interactive alternatives such as cat, sed, heredocs or python3 -c.

Check your work before finishing. When the task is complete, reply with <submit/> and no command."""

BASH_RE = re.compile(r"<bash>\s*\n?(.*?)\n?\s*</bash>", re.S)
SUBMIT_RE = re.compile(r"<submit\s*/?>")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class ChatModel(Protocol):
    async def chat(self, messages: list[dict], temperature: float = ..., max_tokens: int = ...) -> Reply: ...


def parse_action(text: str) -> tuple[str, str]:
    """Return ("bash", cmd) | ("submit", "") | ("error", message_for_model)."""
    visible = THINK_RE.sub("", text)
    cmds = BASH_RE.findall(visible)
    submitted = bool(SUBMIT_RE.search(visible))
    if submitted and cmds:
        return "error", "Reply with either one <bash> command or <submit/>, not both."
    if submitted:
        return "submit", ""
    if len(cmds) == 1 and cmds[0].strip():
        return "bash", cmds[0]
    if len(cmds) > 1:
        return "error", "Give exactly one <bash>...</bash> block per reply."
    return "error", "No command found. Put one command inside <bash>...</bash>, or reply <submit/> if you are done."


def format_observation(res: CommandResult) -> str:
    status = f"Exit code: {res.exit_code}"
    if res.timed_out:
        status += " (command timed out and was killed)"
    out = res.output if res.output.strip() else "(no output)"
    return f"{status}\nOutput:\n{out}"


async def run_episode(
    task: Task,
    llm: ChatModel,
    pool: UserPool,
    *,
    sample_idx: int = 0,
    max_turns: int = 20,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    command_timeout: float = 60,
    model_name: str = "",
) -> dict[str, Any]:
    t0 = time.time()
    max_turns = task.max_turns or max_turns
    cmd_timeout = task.command_timeout or command_timeout
    record: dict[str, Any] = {"task_id": task.id, "sample_idx": sample_idx, "model": model_name}

    async with session(pool, command_timeout=cmd_timeout) as sb:
        setup = await prepare(sb, task)
        if setup.exit_code != 0:
            record.update(reward=None, finish="setup_failed", error=setup.output[-1000:], messages=[], turn_meta=[])
            return record

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(timeout=int(cmd_timeout))},
            {"role": "user", "content": task.instruction},
        ]
        turn_meta: list[dict] = []  # one entry per assistant message
        finish = "max_turns"
        error = None
        for _ in range(max_turns):
            try:
                reply = await llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
            except ContextTooLong:
                finish = "context_length"
                break
            except Exception as e:  # noqa: BLE001  (server down etc.: don't count as a model failure)
                finish, error = "llm_error", repr(e)
                break
            messages.append({"role": "assistant", "content": reply.text})
            kind, payload = parse_action(reply.text)
            if reply.finish_reason == "length" and kind != "submit":
                kind, payload = "error", "Your reply was cut off. Keep replies short: brief reasoning, then one command."
            turn_meta.append({"kind": kind, "format_error": kind == "error"})
            if kind == "submit":
                finish = "submitted"
                break
            if kind == "error":
                obs = payload
            else:
                obs = format_observation(await sb.run(payload))
            messages.append({"role": "user", "content": obs})

        passed, vres = await verify(sb, task)

    record.update(
        reward=None if finish == "llm_error" else float(passed),
        finish=finish,
        error=error,
        n_turns=len(turn_meta),
        n_format_errors=sum(m["format_error"] for m in turn_meta),
        verify_output=vres.output[-1000:],
        seconds=round(time.time() - t0, 1),
        messages=messages,
        turn_meta=turn_meta,
    )
    return record
