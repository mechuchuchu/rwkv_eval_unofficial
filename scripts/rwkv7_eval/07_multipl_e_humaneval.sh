#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "multipl_e_humaneval" "multipl_e_humaneval:olmo3base" "codex_universal" "$@"

