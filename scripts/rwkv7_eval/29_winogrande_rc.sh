#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "winogrande_rc" "winogrande:rc:olmo3base" "default" "$@"

