"""Task format, loading and validation.

A task is a directory containing:

    task.yaml     id, instruction, optional max_turns / command_timeout / tags
    setup.sh      (optional) creates the starting files; runs as the agent user in its home dir
    solution.sh   reference solution, used only to validate the task
    verify.sh     exits 0 iff the task is solved; runs after the agent finishes

Directories whose name starts with "_" (e.g. _rejected) are ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .sandbox import CommandResult, Sandbox, UserPool, session


@dataclass
class Task:
    id: str
    path: Path
    instruction: str
    verify: str
    setup: str = ""
    solution: str | None = None
    max_turns: int | None = None
    command_timeout: float | None = None
    tags: list[str] = field(default_factory=list)


def load_task(path: Path) -> Task:
    meta = yaml.safe_load((path / "task.yaml").read_text()) or {}

    def read(name: str) -> str | None:
        p = path / name
        return p.read_text() if p.exists() else None

    verify = read("verify.sh")
    if verify is None:
        raise ValueError(f"{path}: missing verify.sh")
    return Task(
        id=str(meta.get("id", path.name)),
        path=path,
        instruction=str(meta["instruction"]).strip(),
        verify=verify,
        setup=read("setup.sh") or "",
        solution=read("solution.sh"),
        max_turns=meta.get("max_turns"),
        command_timeout=meta.get("command_timeout"),
        tags=list(meta.get("tags", [])),
    )


def load_tasks(*roots: str | Path) -> list[Task]:
    tasks: dict[str, Task] = {}
    for root in roots:
        root = Path(root)
        for yml in sorted(root.rglob("task.yaml")):
            if any(part.startswith("_") for part in yml.relative_to(root).parts):
                continue
            t = load_task(yml.parent)
            if t.id in tasks:
                raise ValueError(f"duplicate task id {t.id!r}: {tasks[t.id].path} and {t.path}")
            tasks[t.id] = t
    return sorted(tasks.values(), key=lambda t: t.id)


async def prepare(sb: Sandbox, task: Task, timeout: float = 180) -> CommandResult:
    if not task.setup.strip():
        return CommandResult(0, "")
    return await sb.run_script(task.setup, timeout=timeout)


async def verify(sb: Sandbox, task: Task, timeout: float = 180) -> tuple[bool, CommandResult]:
    await sb.kill_all()  # stop agent background jobs so they can't interfere with the check
    res = await sb.run_script(task.verify, timeout=timeout, private=True)
    return (res.exit_code == 0 and not res.timed_out), res


async def validate_task(task: Task, pool: UserPool) -> tuple[bool, str]:
    """A valid task: setup succeeds, verify FAILS before solving, PASSES after the reference solution."""
    if not task.solution:
        return False, "no solution.sh"
    async with session(pool) as sb:
        res = await prepare(sb, task)
        if res.exit_code != 0:
            return False, f"setup failed (exit {res.exit_code}): {res.output[-500:]}"
        ok, _ = await verify(sb, task)
        if ok:
            return False, "verify passes before the task is solved"
    async with session(pool) as sb:
        await prepare(sb, task)
        sol = await sb.run_script(task.solution, timeout=300)
        if sol.exit_code != 0:
            return False, f"solution failed (exit {sol.exit_code}): {sol.output[-500:]}"
        ok, vres = await verify(sb, task)
        if not ok:
            return False, f"verify fails after reference solution: {vres.output[-500:]}"
    return True, "ok"


async def validate_many(tasks: list[Task], pool: UserPool, move_invalid_to: Path | None = None) -> list[tuple[Task, bool, str]]:
    async def one(t: Task):
        try:
            ok, why = await validate_task(t, pool)
        except Exception as e:  # noqa: BLE001
            ok, why = False, f"error: {e!r}"
        print(f"[{'PASS' if ok else 'FAIL'}] {t.id}: {why}", flush=True)
        if not ok and move_invalid_to is not None:
            move_invalid_to.mkdir(parents=True, exist_ok=True)
            dest = move_invalid_to / t.path.name
            shutil.move(str(t.path), dest)
            (dest / "rejected_reason.txt").write_text(why)
        return t, ok, why

    results = await asyncio.gather(*(one(t) for t in tasks))
    n_ok = sum(ok for _, ok, _ in results)
    print(f"\n{n_ok}/{len(results)} tasks valid")
    return list(results)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate task directories")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("dirs", nargs="+")
    v.add_argument("--users", type=int, default=8, help="sandbox users to use (= parallelism)")
    v.add_argument("--move-invalid", type=Path, default=None, help="move invalid tasks into this dir")
    args = ap.parse_args()
    tasks = load_tasks(*args.dirs)
    print(f"Validating {len(tasks)} tasks")

    async def run():
        results = await validate_many(tasks, UserPool(args.users), args.move_invalid)
        return all(ok for _, ok, _ in results)

    raise SystemExit(0 if asyncio.run(run()) else 1)


if __name__ == "__main__":
    main()
