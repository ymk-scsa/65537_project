#!/usr/bin/env bash
# WSL 上でプロジェクトの venv を有効化し、python -u で引数を実行する。
# Windows の bat から wsl -d Ubuntu-22.04 経由で呼ばれる想定。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${WSL_VENV:-${ROOT}/.venv}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found: ${VENV}" >&2
  echo "Setup example:" >&2
  echo "  cd ${ROOT}" >&2
  echo "  python3 -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  echo "  # GPU torch example:" >&2
  echo "  # pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${ROOT}"
exec python -u "$@"
