#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
run_eval "squad_gen2mc_mc" "squad:mc:olmo3base" "default" "$@"

