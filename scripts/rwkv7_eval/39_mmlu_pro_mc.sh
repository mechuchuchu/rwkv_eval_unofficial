#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
unsupported_eval "MMLU Pro MC" "No MMLU Pro task or suite is registered in this olmo-eval checkout."

