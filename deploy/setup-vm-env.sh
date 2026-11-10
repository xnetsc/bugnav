#!/usr/bin/env bash
# Recreate Muse's persistent Python env for bugnav work on a fresh VM.
# Persistent location (survives VM replacement): ~/workspace/envs/bugnav-venv
# NEVER build throwaway venvs in /tmp - they vanish with the VM.
set -euo pipefail
VENV="$HOME/workspace/envs/bugnav-venv"
mkdir -p "$HOME/workspace/envs"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -r "$HOME/workspace/t80s-bugnav/requirements.txt"
echo "env ready: $VENV"
"$VENV/bin/python" -c "import deebot_client, bugnav" 2>/dev/null || \
  PYTHONPATH="$HOME/workspace/t80s-bugnav" "$VENV/bin/python" -c "import deebot_client, bugnav; print('imports OK')"
