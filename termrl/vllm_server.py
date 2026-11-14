"""Start/stop a vLLM OpenAI-compatible server as a subprocess (used by iterate.py)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


class VLLMServer:
    def __init__(
        self,
        model: str,
        port: int = 8000,
        served_name: str = "policy",
        version: str = "0.19.1",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 16384,
        tensor_parallel_size: int = 1,
        extra_args: list[str] | None = None,
        log_path: str | Path = "vllm.log",
        startup_timeout: float = 1800,
    ):
        self.model, self.port, self.served_name = model, port, served_name
        self.cmd = [
            "uvx", "--from", f"vllm=={version}", "vllm", "serve", model,
            "--port", str(port),
            "--served-model-name", served_name,
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--tensor-parallel-size", str(tensor_parallel_size),
            *(extra_args or []),
        ]
        self.log_path = Path(log_path)
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def _ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                return self.served_name in json.loads(r.read())["data"][0]["id"]
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> "VLLMServer":
        if self._ready():
            raise RuntimeError(f"Something is already serving on port {self.port}; stop it first.")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w")
        print(f"Starting vLLM: {' '.join(self.cmd)}\n  log: {self.log_path}", flush=True)
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {self.proc.returncode}; see {self.log_path}")
            if self._ready():
                print(f"vLLM ready after {time.time() - t0:.0f}s", flush=True)
                return self
            time.sleep(5)
        self.stop()
        raise TimeoutError(f"vLLM not ready after {self.startup_timeout}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        time.sleep(5)  # give the driver a moment to release GPU memory
        print("vLLM stopped", flush=True)

    def __enter__(self) -> "VLLMServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
