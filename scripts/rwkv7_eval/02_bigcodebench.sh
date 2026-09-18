#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "bigcodebench" "bigcodebench:olmo3base" "codex_universal" "$@"

