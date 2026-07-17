"""RWKV-7 G1x prompt rendering rules."""

from __future__ import annotations

import json
import re
from typing import Any

ROLE_STOPS = ("\n\nUser:", "\n\nSystem:", "\n\nAssistant:")


def clean_text(text: str) -> str:
    """Match the normalization prescribed by RWKV7-G1x-templates.txt."""
    return re.sub(r"\n{2,}", "\n", text.replace("\r\n", "\n")).strip()


def _tool_system_prompt(tools: list[dict[str, Any]]) -> str:
    rendered = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return f"Tools:\n{rendered}\nReturn only a JSON function call."


def render_g1x(
    messages: list[dict[str, Any]],
    *,
    think_mode: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Serialize OpenAI messages using the G1-series textual chat template."""
    parts: list[str] = []
    if tools:
        parts.append(f"System: {_tool_system_prompt(tools)}")

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            parts.append(f"System: {clean_text(str(content))}")
        elif role == "user":
            parts.append(f"User: {clean_text(str(content))}")
        elif role == "assistant":
            parts.append(f"Assistant: {str(content).strip()}")
        elif role == "tool":
            parts.append(f"User: Function output:\n{clean_text(str(content))}")
        else:
            raise ValueError(f"unsupported G1x role: {role!r}")

    prefixes = {
        None: "Assistant:",
        "none": "Assistant:",
        "think": "Assistant: <think",
        "fake": "Assistant: <think>\n</think",
        "fake_compact": "Assistant: <think></think",
        "json": "Assistant: ```json",
    }
    try:
        prefix = prefixes[think_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported think_mode: {think_mode!r}") from exc
    parts.append(prefix)
    return "\n\n".join(parts)


def merge_stops(request_stops: list[str] | None, *, chat: bool) -> list[str]:
    stops = list(request_stops or ())
    if chat:
        for stop in ROLE_STOPS:
            if stop not in stops:
                stops.append(stop)
    return stops
