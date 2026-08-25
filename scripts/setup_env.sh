#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -x .venv/bin/python ]]; then
  python_candidate=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_candidate="$candidate"
      break
    fi
  done
  if [[ -z "$python_candidate" ]]; then
    echo "Python 3.11 or 3.12 is required." >&2
    exit 1
  fi
  "$python_candidate" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install ".[dev]"
# Conda-based Python distributions may skip setuptools' hidden editable-install .pth file.
# Force-refresh the local package so the console command cannot retain an older source copy.
.venv/bin/python -m pip install --force-reinstall --no-deps .
.venv/bin/python -m pytest
.venv/bin/python -c \
  "from pathlib import Path; from garl_trading.config import load_config; c=load_config('configs/default.toml'); print(f'Environment ready; device={c.models.device}; rebalance_threshold={c.execution.rebalance_threshold}; source={Path(__import__(\"garl_trading\").__file__).resolve()}')"
