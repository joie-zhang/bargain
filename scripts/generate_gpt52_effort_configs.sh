#!/bin/bash
# Generate GPT-5.2 reasoning_effort sweep configs.

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BASE_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" "$BASE_DIR/scripts/generate_gpt52_effort_configs.py" "$@"
