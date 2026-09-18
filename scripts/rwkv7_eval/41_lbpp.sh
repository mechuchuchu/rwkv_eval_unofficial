#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
unsupported_eval "LBPP" "No LBPP task or suite is registered in this olmo-eval checkout."

