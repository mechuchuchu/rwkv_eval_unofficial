"""Small dependency-free reader for the RWKV world tokenizer vocabulary."""

from __future__ import annotations

import ast
from pathlib import Path


class _TrieNode:
    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.token_id: int | None = None


class RWKVTokenizer:
    def __init__(self, vocab_path: str | Path) -> None:
        self.id_to_bytes: dict[int, bytes] = {}
        self.root = _TrieNode()
        with Path(vocab_path).open("r", encoding="utf-8") as vocab:
            for line in vocab:
                first = line.index(" ")
                last = line.rindex(" ")
                token_id = int(line[:first])
                value = ast.literal_eval(line[first:last])
                token = value.encode("utf-8") if isinstance(value, str) else value
                if not isinstance(token, bytes) or len(token) != int(line[last:]):
                    raise ValueError(f"invalid RWKV vocabulary row: {line!r}")
                self.id_to_bytes[token_id] = token
                node = self.root
                for byte in token:
                    node = node.children.setdefault(byte, _TrieNode())
                node.token_id = token_id

    def encode(self, text: str) -> list[int]:
        source = text.encode("utf-8")
        result: list[int] = []
        start = 0
        while start < len(source):
            node = self.root
            found: tuple[int, int] | None = None
            cursor = start
            while cursor < len(source) and source[cursor] in node.children:
                node = node.children[source[cursor]]
                cursor += 1
                if node.token_id is not None:
                    found = cursor, node.token_id
            if found is None:
                raise ValueError(f"RWKV vocabulary cannot encode byte at offset {start}")
            start, token_id = found
            result.append(token_id)
        return result

    def decode_bytes(self, token_ids: list[int]) -> bytes:
        return b"".join(self.id_to_bytes[token_id] for token_id in token_ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.decode_bytes(token_ids).decode("utf-8", errors="replace")
