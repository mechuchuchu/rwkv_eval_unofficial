#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "jeopardy_gen2mc_mc" "jeopardy:mc:olmo3base" "default" "$@"

