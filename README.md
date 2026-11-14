# termrl

Train a small open model to use a Linux terminal, on **one RunPod GPU pod**.

termrl follows the same idea as [tmax](https://github.com/hamishivi/tmax), simplified so it fits what a RunPod pod can do:

| | tmax | termrl |
|---|---|---|
| Task sandbox | Docker / Podman / Apptainer containers, or Daytona | **No containers.** Each episode runs as its own unprivileged Linux user in a private directory |
| Task data | LLM-generated tasks with verifiers (TMax-15K) | LLM-generated tasks with verifiers (`generate_tasks.py`), plus 13 hand-written seed tasks |
| Training | SFT, then DPPO RL on multi-node clusters (Beaker) | Optional SFT warm start, then **expert iteration** (learn from your own successful attempts) on 1 GPU |
| Serving / eval | vLLM + Harbor | vLLM + a built-in agent loop, reporting pass@k |

Why no containers: a RunPod pod *is* a Docker container, and you can't start Docker, Podman or Apptainer containers inside it. termrl isolates episodes with Linux users instead (details under [How the sandbox works](#how-the-sandbox-works)).

---

## 1. Create the pod

* **GPU:** 1× L40S / A6000 (48 GB) is plenty for 0.5B–3B models. Use 1× A100 / H100 (80 GB) for 7B models.
* **Template:** any recent RunPod PyTorch template (Ubuntu). termrl builds its own Python env with `uv`, so the template's torch version doesn't matter.
* **Storage:** attach a **network volume** (100 GB or more) mounted at `/workspace`. Models, caches and runs live there and survive pod restarts. Anything outside `/workspace` is wiped when the pod stops.
* **vCPUs:** episodes run on the CPU. Set `concurrency` (default 16) to about the number of vCPUs.
* **Environment variables / Secrets:** `TEACHER_BASE_URL`, `TEACHER_MODEL`, `TEACHER_API_KEY` (any OpenAI-compatible API, used to generate tasks), plus `HF_TOKEN` if you use gated models. See `.env.example`.

## 2. Install

From the pod's web terminal or SSH (as root, the default):

```bash
cd /workspace
git clone <your fork of this repo> termrl     # or upload the zip and unzip it here
cd termrl
bash scripts/runpod_setup.sh
source /etc/profile.d/termrl.sh
```

The setup script:
* installs the tools agents use in tasks (git, jq, sqlite3, make, ...);
* installs `uv` and the Python env;
* creates 32 sandbox users;
* locks `/workspace`, the repo and `/root` so sandbox users can't read your secrets or the task solutions;
* runs a self-test and validates the seed tasks.

Re-run it after a pod restart: users and apt packages live outside `/workspace`. It's quick the second time.

## 3. Baseline: how good is the model before training?

```bash
tmux new -d -s vllm 'bash scripts/serve.sh Qwen/Qwen2.5-1.5B-Instruct'   # first start downloads vLLM + the model
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done     # wait until ready (logs: tmux attach -t vllm)

uv run python -m termrl.rollout --tasks tasks/seed --k 4 --out runs/baseline/seed.jsonl
```

This prints pass@1, pass@4, how often episodes ended by submitting vs. running out of turns, and so on. Every full conversation is saved in the `.jsonl`, which is the best way to see *why* the model fails.

Stop the server (`tmux kill-session -t vllm`) before you train: training needs the GPU memory.

## 4. Generate training tasks with a teacher model

```bash
uv run python -m termrl.generate_tasks --n 300 --out tasks/generated
```

Each task comes from a randomly drawn domain × skill × persona × difficulty, and then gets **executed**. A task is kept only if:
* its setup runs;
* its verifier fails before the task is solved;
* its verifier passes after the teacher's reference solution.

Expect a fair share of tasks to be rejected; the reasons are in `tasks/generated/_rejected/*/rejected_reason.txt`. Add more with `--start 300 --n 300`.

Tips:
* Use a strong teacher (a frontier API model). Weak teachers write verifiers that are broken or too easy.
* Read 10–20 generated tasks before training on them. Verifiers that can be fooled teach the model to fool them.
* For newer OpenAI models, add `--max-tokens-field max_completion_tokens`.

## 5. (Optional but recommended) warm start from teacher demonstrations

Small models often solve almost nothing at first, which leaves expert iteration nothing to learn from. Fix that by first imitating the teacher (the same idea as tmax's "solve at pass@k, then SFT" stage):

```bash
# teacher attempts the tasks (k=2 each)
uv run python -m termrl.rollout --tasks tasks/generated --k 2 --out runs/teacher.jsonl \
    --base-url "$TEACHER_BASE_URL" --model "$TEACHER_MODEL" --api-key-env TEACHER_API_KEY

# keep successful attempts -> SFT data -> LoRA fine-tune + merge
uv run python -m termrl.build_sft runs/teacher.jsonl --out runs/sft/teacher.jsonl --max-per-task 2
uv run python -m termrl.train_sft --model Qwen/Qwen2.5-1.5B-Instruct \
    --data runs/sft/teacher.jsonl --output runs/models/warmstart
```

Then set `base_model: runs/models/warmstart/merged` in `configs/default.yaml`.

## 6. Expert iteration (the RL loop)

```bash
tmux new -s train 'uv run python -m termrl.iterate configs/default.yaml 2>&1 | tee runs/iterate.log'
```

Each iteration:
1. Serves the current model with vLLM.
2. Evaluates it on the held-out seed tasks.
3. Makes `k` attempts at each training task.
4. Stops vLLM.
5. LoRA-fine-tunes the model on its own successful attempts and merges the result.

The merged model is the next iteration's starting point. Metrics go to `runs/expert_iter/metrics.jsonl`. If the pod restarts, re-running the command resumes after the last finished iteration.

What to watch: eval `pass@1` should rise over iterations, and train `solved_any` tells you how much signal there is. If `solved_any` is near 0, warm-start first or use easier tasks. If it's near 1, generate harder tasks.

## Task format

```
tasks/seed/count-errors/
  task.yaml     # id, instruction, optional max_turns / command_timeout / tags
  setup.sh      # creates the starting files (runs as the agent user, in its home dir)
  solution.sh   # reference solution, only used for validation
  verify.sh     # exit 0 = solved; runs after the agent finishes
```

Write your own tasks the same way, then check them:

```bash
uv run python -m termrl.tasks validate tasks/mine
```

Rules for good verifiers:
* Recompute expected answers independently: regenerate the input data with the same seed rather than reading files the agent could edit.
* Accept any correct solution, not just the reference one.
* Use `python3 -I` so files in the agent's directory can't hijack Python imports.

## How the sandbox works

* Each concurrent episode borrows a user (`termrl0`…`termrl31`) and gets an empty private home directory (mode 700).
* Commands run under `setpriv` with `no_new_privs`, a clean environment (no API keys or HF token), a process limit, a file-size limit and a timeout that kills the whole process group.
* The current directory and exported variables persist between the agent's commands, like a real shell.
* After the agent finishes, all of its processes are killed, and only then is the verifier copied in and run. The agent never sees the verifier or the solution.
* Afterwards the user's files and processes are cleaned up and the user goes back to the pool.

**Limits compared with containers:**
* Tasks can't need root or `apt install`. The generator prompt tells the teacher this.
* The agent can read world-readable system files and **has network access**.
* Agents share the pod's kernel and CPU.

Use a dedicated pod, and keep your secrets in environment variables, not in files.

## Repo layout

```
termrl/sandbox.py         per-episode users, command execution
termrl/tasks.py           task loading + validation
termrl/agent.py           prompt, action parsing, episode loop
termrl/rollout.py         k attempts per task, pass@k summary
termrl/generate_tasks.py  teacher-generated tasks + validation
termrl/build_sft.py       successful trajectories -> prompt/completion SFT data
termrl/train_sft.py       TRL + PEFT LoRA training, adapter merge
termrl/vllm_server.py     start/stop vLLM
termrl/iterate.py         expert-iteration loop
configs/default.yaml      settings for iterate.py
scripts/runpod_setup.sh   one-time pod setup
scripts/serve.sh          serve a model with vLLM
tasks/seed/               13 hand-written held-out tasks
tests/test_e2e.py         full loop with a scripted fake model (no GPU needed)
```

Run the tests with `uv run pytest -q tests`. They need root and the sandbox users.

## Troubleshooting

* **`sandbox users are missing`**: re-run `bash scripts/runpod_setup.sh`. This happens after every pod restart.
* **vLLM won't start**: read `runs/.../vllm.log` or the tmux window. CUDA errors usually mean the pod's driver is too old for that vLLM version. Pick a pod with a newer CUDA version (RunPod lets you filter by it) or set `VLLM_VERSION` / `vllm.version` to an older release.
* **Out of memory while training**: lower `sft.max_length` or `lora_r`, or use a smaller model. Make sure vLLM was stopped.
* **Qwen3-style models**: set `no_thinking: true` (or pass `--no-thinking`) unless you want `<think>` reasoning, which needs a larger `max_tokens` and `max_model_len`.

## Next steps beyond this repo

* Learn from failures too: replace expert iteration with GRPO-style updates (advantage = reward − mean reward of the group). The rollout files already hold `k` samples per task, which is exactly what GRPO groups need.
* Scale out: run several rollout workers or pods against one vLLM server, or a server with tensor parallelism.
