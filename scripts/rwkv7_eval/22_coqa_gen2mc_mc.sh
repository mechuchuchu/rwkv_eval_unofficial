#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "coqa_gen2mc_mc" "coqa:mc:olmo3base" "default" "$@"

