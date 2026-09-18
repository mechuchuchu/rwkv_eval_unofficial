#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
failed=()
total=0

for script in "${SCRIPT_DIR}"/[0-9][0-9]_*.sh; do
    [[ -f "${script}" ]] || continue
    total=$((total + 1))
    name="$(basename "${script}")"
    echo
    echo "===== [${total}] ${name} ====="
    if "${script}" "$@"; then
        echo "===== ${name}: OK ====="
    else
        status=$?
        failed+=("${name} (exit ${status})")
        echo "===== ${name}: FAILED (exit ${status}); continuing =====" >&2
    fi
done

echo
echo "Completed ${total} benchmark wrappers."
if (( ${#failed[@]} > 0 )); then
    echo "Failed or unavailable:"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi

