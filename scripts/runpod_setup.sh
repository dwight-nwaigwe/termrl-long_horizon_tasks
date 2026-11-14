#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (runs as root). Safe to re-run after a pod restart.
#   bash scripts/runpod_setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
N_USERS="${N_USERS:-32}"
WS="${WORKSPACE:-/workspace}"

echo "== System packages (tools the agent can use inside tasks)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  ca-certificates curl git python3 procps util-linux tmux jq sqlite3 \
  make bc zip unzip diffutils file less >/dev/null

echo "== Caches on the network volume (survive pod restarts)"
mkdir -p "$WS/.cache/uv" "$WS/.cache/huggingface"
cat > /etc/profile.d/termrl.sh <<EOF
export UV_CACHE_DIR=$WS/.cache/uv
export HF_HOME=$WS/.cache/huggingface
export PATH=\$HOME/.local/bin:\$PATH
EOF
grep -q termrl.sh ~/.bashrc 2>/dev/null || echo "source /etc/profile.d/termrl.sh" >> ~/.bashrc
source /etc/profile.d/termrl.sh

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== Python environment"
cd "$REPO_DIR"
uv sync --extra train --extra dev

echo "== Sandbox users and directories"
uv run python -m termrl.sandbox setup --users "$N_USERS"

echo "== Hide secrets and task solutions from the sandbox users"
chmod 700 "$WS" 2>/dev/null || true
chmod -R go-rwx "$REPO_DIR"
chmod 700 /root 2>/dev/null || true

echo "== Self-test"
uv run python -m termrl.sandbox selftest
uv run python -m termrl.tasks validate tasks/seed --users 8

cat <<EOF

Setup complete. Next:
  source /etc/profile.d/termrl.sh
  tmux new -s vllm 'bash scripts/serve.sh Qwen/Qwen2.5-1.5B-Instruct'
  uv run python -m termrl.rollout --tasks tasks/seed --k 4 --out runs/baseline/seed.jsonl
EOF
