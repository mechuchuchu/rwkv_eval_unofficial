#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "hellaswag_rc" "hellaswag:rc:olmo3base" "default" "$@"

