"""Container-free sandbox for agent episodes.

A RunPod pod is already a Docker container, so we cannot start Docker, Podman or
Apptainer containers inside it. Instead every concurrent episode gets:

* its own unprivileged Linux user from a pool (termrl0 .. termrlN-1),
* a private, empty working directory (mode 700) that is the agent's $HOME,
* a clean environment (no API keys or HF tokens leak in),
* `no_new_privs`, a process limit and a file-size limit,
* a per-command timeout that kills the whole process group.

Shell state (current directory and exported variables) persists between the
agent's commands, like an interactive shell, but each command runs in a fresh
bash process so timeouts and background jobs are easy to handle.

This is weaker isolation than a container: the agent can read world-readable
files on the pod and can use the network. Run it on a disposable pod and keep
secrets and task solutions out of reach (scripts/runpod_setup.sh does chmod 700
on /workspace and on the repo).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

SANDBOX_ROOT = Path(os.environ.get("TERMRL_SANDBOX_ROOT", "/sandbox"))
USER_PREFIX = os.environ.get("TERMRL_USER_PREFIX", "termrl")
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Runs as the sandbox user. Restores cwd + exported vars, runs the command in the
# current shell (so `cd` and `export` persist), then saves state again.
WRAPPER = r"""
cd "$(cat "$TERMRL_STATE/cwd" 2>/dev/null)" 2>/dev/null || cd "$HOME"
[ -f "$TERMRL_STATE/env" ] && source "$TERMRL_STATE/env" >/dev/null 2>&1
ulimit -u {nproc} 2>/dev/null
ulimit -f {fsize_kb} 2>/dev/null
source "$TERMRL_STATE/cmd"
__termrl_rc=$?
pwd > "$TERMRL_STATE/cwd" 2>/dev/null
export -p | grep -v -E '^declare -x (PWD|OLDPWD|SHLVL|_|TERMRL_STATE)=' > "$TERMRL_STATE/env" 2>/dev/null
exit $__termrl_rc
"""


@dataclass
class CommandResult:
    exit_code: int
    output: str
    timed_out: bool = False


def user_exists(name: str) -> bool:
    try:
        pwd.getpwnam(name)
        return True
    except KeyError:
        return False


def user_names(n: int) -> list[str]:
    return [f"{USER_PREFIX}{i}" for i in range(n)]


def setup_host(n_users: int) -> None:
    """Create sandbox users and directories. Needs root. Safe to re-run."""
    if os.geteuid() != 0:
        sys.exit("termrl.sandbox setup must run as root (RunPod pods run as root by default).")
    for name in user_names(n_users):
        if not user_exists(name):
            subprocess.run(["useradd", "-M", "-d", "/nonexistent", "-s", "/bin/bash", name], check=True)
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(SANDBOX_ROOT, 0o711)
    for sub, mode in [("work", 0o711), ("state", 0o711), ("verify", 0o711), ("out", 0o700)]:
        d = SANDBOX_ROOT / sub
        d.mkdir(exist_ok=True)
        os.chmod(d, mode)
    print(f"Sandbox ready: {n_users} users ({USER_PREFIX}0..{USER_PREFIX}{n_users - 1}), root {SANDBOX_ROOT}")


class UserPool:
    """Hands out one sandbox user per concurrent episode."""

    def __init__(self, n: int):
        names = user_names(n)
        missing = [u for u in names if not user_exists(u)]
        if missing:
            raise RuntimeError(
                f"{len(missing)} sandbox users are missing (e.g. {missing[0]}). "
                f"Run: python -m termrl.sandbox setup --users {n}"
            )
        self.size = n
        self._names = names
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[str] | None = None

    def _q(self) -> asyncio.Queue[str]:
        # asyncio queues are bound to one event loop; rebuild for each asyncio.run().
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop, self._queue = loop, asyncio.Queue()
            for u in self._names:
                self._queue.put_nowait(u)
        return self._queue

    async def acquire(self) -> str:
        return await self._q().get()

    def release(self, user: str) -> None:
        self._q().put_nowait(user)


def _read_head_tail(path: str, max_bytes: int) -> str:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size <= max_bytes:
            data = f.read()
            text = data.decode("utf-8", "replace")
        else:
            half = max_bytes // 2
            head = f.read(half)
            f.seek(size - half)
            tail = f.read(half)
            text = (
                head.decode("utf-8", "replace")
                + f"\n\n[... {size - 2 * half} bytes of output omitted ...]\n\n"
                + tail.decode("utf-8", "replace")
            )
    return ANSI_RE.sub("", text).replace("\r\n", "\n")


class Sandbox:
    def __init__(
        self,
        user: str,
        command_timeout: float = 60,
        max_output_bytes: int = 8000,
        nproc: int = 512,
        fsize_kb: int = 2 * 1024 * 1024,
    ):
        self.user = user
        pw = pwd.getpwnam(user)
        self.uid, self.gid = pw.pw_uid, pw.pw_gid
        self.command_timeout = command_timeout
        self.max_output_bytes = max_output_bytes
        self.wrapper = WRAPPER.format(nproc=nproc, fsize_kb=fsize_kb)
        name = f"{user}-{uuid.uuid4().hex[:10]}"
        self.workdir = SANDBOX_ROOT / "work" / name
        self.statedir = SANDBOX_ROOT / "state" / name
        self.tmpdir = self.statedir / "tmp"
        self.outdir = SANDBOX_ROOT / "out" / name

    # ---------- lifecycle ----------
    async def start(self) -> None:
        await self.kill_all()
        for d in (self.workdir, self.statedir, self.tmpdir):
            d.mkdir(parents=True)
            os.chown(d, self.uid, self.gid)
            os.chmod(d, 0o700)
        self.outdir.mkdir(parents=True)
        os.chmod(self.outdir, 0o700)

    async def close(self) -> None:
        await self.kill_all()
        for d in (self.workdir, self.statedir, self.outdir):
            await asyncio.to_thread(shutil.rmtree, d, True)
        # Remove anything the user left in shared temp dirs.
        proc = await asyncio.create_subprocess_exec(
            "find", "/tmp", "/var/tmp", "/dev/shm", "-xdev", "-user", self.user, "-delete",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        await proc.wait()

    async def kill_all(self) -> None:
        """Kill every process owned by this sandbox user (incl. background jobs)."""
        for _ in range(20):
            proc = await asyncio.create_subprocess_exec(
                "pkill", "-KILL", "-u", self.user,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            rc = await proc.wait()
            if rc == 1:  # no processes matched
                return
            await asyncio.sleep(0.1)

    # ---------- execution ----------
    def _env(self, with_state: bool) -> dict[str, str]:
        env = {
            "HOME": str(self.workdir),
            "USER": self.user,
            "LOGNAME": self.user,
            "SHELL": "/bin/bash",
            "PATH": SAFE_PATH,
            "TERM": "dumb",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TMPDIR": str(self.tmpdir),
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "DEBIAN_FRONTEND": "noninteractive",
            "PYTHONUNBUFFERED": "1",
        }
        if with_state:
            env["TERMRL_STATE"] = str(self.statedir)
        return env

    async def _exec(self, argv: list[str], env: dict[str, str], cwd: Path, timeout: float) -> CommandResult:
        fd, out_path = tempfile.mkstemp(prefix="out-", dir=self.outdir)
        try:
            proc = await asyncio.create_subprocess_exec(
                "setpriv",
                f"--reuid={self.uid}",
                f"--regid={self.gid}",
                "--clear-groups",
                "--no-new-privs",
                *argv,
                stdin=subprocess.DEVNULL,
                stdout=fd,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(cwd),
                start_new_session=True,
            )
            timed_out = False
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout)
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
                rc = 124
        finally:
            os.close(fd)
        try:
            output = _read_head_tail(out_path, self.max_output_bytes)
        finally:
            os.unlink(out_path)
        return CommandResult(exit_code=rc, output=output, timed_out=timed_out)

    async def run(self, command: str, timeout: float | None = None) -> CommandResult:
        """Run one agent command with persistent shell state."""
        cmd_file = self.statedir / "cmd"
        cmd_file.write_text(command.rstrip("\n") + "\n")
        os.chmod(cmd_file, 0o644)
        return await self._exec(
            ["bash", "--noprofile", "--norc", "-c", self.wrapper],
            env=self._env(with_state=True),
            cwd=self.workdir,
            timeout=timeout or self.command_timeout,
        )

    async def run_script(self, script: str, timeout: float = 120, private: bool = False) -> CommandResult:
        """Run a whole bash script (task setup / solution / verifier) as the sandbox user.

        private=True writes the script to a fresh directory created *now*, so the
        agent (whose processes have been killed) can't have tampered with it.
        """
        base = SANDBOX_ROOT / "verify" if private else self.statedir
        d = Path(tempfile.mkdtemp(prefix="script-", dir=base))
        try:
            os.chown(d, self.uid, self.gid)
            os.chmod(d, 0o700)
            path = d / "script.sh"
            path.write_text(script)
            os.chmod(path, 0o644)
            env = self._env(with_state=False)
            env["WORKDIR"] = str(self.workdir)
            return await self._exec(["bash", "--noprofile", "--norc", str(path)], env=env, cwd=self.workdir, timeout=timeout)
        finally:
            await asyncio.to_thread(shutil.rmtree, d, True)


@asynccontextmanager
async def session(pool: UserPool, **kwargs):
    user = await pool.acquire()
    sb = Sandbox(user, **kwargs)
    try:
        await sb.start()
        yield sb
    finally:
        try:
            await sb.close()
        finally:
            pool.release(user)


async def _selftest(n_users: int) -> None:
    pool = UserPool(n_users)
    async with session(pool) as sb:
        for cmd in ["whoami", "cd /tmp && export FOO=bar", "pwd; echo $FOO", "env | grep -c TOKEN || true", "sleep 5"]:
            res = await sb.run(cmd, timeout=2)
            print(f"$ {cmd}\n{res.output.rstrip()}\n[exit {res.exit_code}{' timeout' if res.timed_out else ''}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sandbox host setup and self-test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="create sandbox users and directories (root)")
    s.add_argument("--users", type=int, default=32)
    t = sub.add_parser("selftest", help="run a few commands in a sandbox")
    t.add_argument("--users", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "setup":
        setup_host(args.users)
    else:
        asyncio.run(_selftest(args.users))


if __name__ == "__main__":
    main()
