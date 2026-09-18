#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "mmlu_humanities" "mmlu:humanities:mc:olmo3base" "default" "$@"

