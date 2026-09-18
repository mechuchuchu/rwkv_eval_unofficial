#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "deepseek_leetcode" "deepseek_leetcode:olmo3base" "codex_universal" "$@"

