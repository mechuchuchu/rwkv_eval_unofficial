#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "olmo_3_eval_mc_non_stem" "olmobase:mcqa_non_stem" "default" "$@"

