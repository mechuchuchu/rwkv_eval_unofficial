"""Tests for the RWKV evaluation launcher command."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
COMMON = ROOT / "scripts" / "rwkv7_eval" / "_common.sh"


def test_launcher_uses_the_rwkv_compatible_transformers_overlay(tmp_path: Path) -> None:
    """The launcher pins the cache API required by the bundled RWKV snapshot."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "args"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "${CAPTURE_FILE}"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    model_path = tmp_path / "model"
    model_path.mkdir()
    output_root = tmp_path / "results"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CAPTURE_FILE": str(capture),
            "MODEL_PATH": str(model_path),
            "OUTPUT_ROOT": str(output_root),
            "LIMIT": "2",
        }
    )

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; run_eval arc_mc "arc:mc:olmo3base" "default"',
            "bash",
            str(COMMON),
        ],
        check=True,
        env=environment,
    )

    assert capture.read_text(encoding="utf-8").splitlines()[:3] == [
        "run",
        "--with",
        "transformers>=5.15,<5.16",
    ]
