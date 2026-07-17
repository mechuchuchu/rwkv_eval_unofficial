"""Dependency-free OpenAI/vLLM-compatible HTTP surface for OLMo Eval."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backend import AlbatrossBackend, Backend, GenerationResult, SamplingConfig
from .g1x import merge_stops, render_g1x


class APIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class AdapterApp:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def get(self, path: str) -> tuple[int, dict[str, Any]]:
        if path in ("/health", "/healthz"):
            return 200, {"status": "ok"}
        if path == "/v1/models":
            return 200, {
                "object": "list",
                "data": [{
                    "id": self.backend.model_name,
                    "object": "model",
                    "owned_by": "rwkv",
                    "max_model_len": self.backend.max_model_len,
                }],
            }
        raise APIError(404, f"unknown endpoint: {path}")

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self._validate_model(body.get("model"))
        if path == "/tokenize":
            prompt = body.get("prompt", "")
            if not isinstance(prompt, str):
                raise APIError(400, "tokenize.prompt must be a string")
            tokens = self.backend.encode(
                prompt, add_special_tokens=bool(body.get("add_special_tokens", True))
            )
            return 200, {"tokens": tokens, "count": len(tokens), "max_model_len": self.backend.max_model_len}
        if path == "/detokenize":
            tokens = body.get("tokens")
            if not isinstance(tokens, list) or not all(isinstance(x, int) for x in tokens):
                raise APIError(400, "detokenize.tokens must be an integer array")
            return 200, {"prompt": self.backend.decode(tokens)}
        if path == "/v1/completions":
            return 200, self._completions(body)
        if path == "/v1/chat/completions":
            return 200, self._chat_completions(body)
        raise APIError(404, f"unknown endpoint: {path}")

    def _validate_model(self, requested: Any) -> None:
        if requested and requested != self.backend.model_name:
            raise APIError(404, f"unknown model {requested!r}; expected {self.backend.model_name!r}")

    @staticmethod
    def _stops(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value
        raise APIError(400, "stop must be a string or an array of strings")

    def _sampling(self, body: dict[str, Any], *, chat: bool) -> SamplingConfig:
        extra = body.get("extra_body") or {}
        stops = merge_stops(self._stops(body.get("stop")), chat=chat)
        return SamplingConfig(
            max_tokens=max(0, int(body.get("max_tokens", 512))),
            temperature=max(0.0, float(body.get("temperature", 0.0))),
            top_p=min(1.0, max(0.0, float(body.get("top_p", 1.0)))),
            top_k=body.get("top_k", extra.get("top_k")),
            stop=tuple(stops),
        )

    def _prompt_ids(self, prompt: Any, *, add_special_tokens: bool) -> list[int]:
        if isinstance(prompt, str):
            return self.backend.encode(prompt, add_special_tokens=add_special_tokens)
        if isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
            return prompt
        raise APIError(400, "prompt must be a string or one integer token array")

    @staticmethod
    def _completion_logprobs(result: GenerationResult) -> dict[str, Any]:
        offsets: list[int] = []
        offset = 0
        for token in result.token_texts:
            offsets.append(offset)
            offset += len(token)
        return {
            "tokens": result.token_texts,
            "token_logprobs": result.token_logprobs,
            "top_logprobs": result.top_logprobs,
            "text_offset": offsets,
        }

    def _completions(self, body: dict[str, Any]) -> dict[str, Any]:
        add_special = bool(body.get("add_special_tokens", False))
        prompt_ids = self._prompt_ids(body.get("prompt", ""), add_special_tokens=add_special)
        prompt_lp = body.get("prompt_logprobs")
        created = int(time.time())
        if prompt_lp is not None:
            values = self.backend.prompt_logprobs(prompt_ids, max(1, int(prompt_lp)))
            return {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": created,
                "model": self.backend.model_name,
                "choices": [{"index": 0, "text": "", "finish_reason": "length", "prompt_logprobs": values}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": 0, "total_tokens": len(prompt_ids)},
            }

        config = self._sampling(body, chat=False)
        n = max(1, int(body.get("n", 1)))
        results = [self.backend.generate(prompt_ids, config) for _ in range(n)]
        completion_tokens = sum(len(result.token_ids) for result in results)
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": created,
            "model": self.backend.model_name,
            "choices": [
                {
                    "index": index,
                    "text": result.text,
                    "finish_reason": result.finish_reason,
                    "logprobs": self._completion_logprobs(result),
                }
                for index, result in enumerate(results)
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_ids) + completion_tokens,
            },
        }

    def _chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise APIError(400, "messages must be an array")
        extra = body.get("extra_body") or {}
        template_kwargs = extra.get("chat_template_kwargs") or {}
        think_mode = template_kwargs.get("think_mode")
        if body.get("tools") and think_mode is None:
            think_mode = "json"
        prompt = render_g1x(messages, think_mode=think_mode, tools=body.get("tools"))
        prompt_ids = self.backend.encode(prompt, add_special_tokens=True)
        config = self._sampling(body, chat=True)
        n = max(1, int(body.get("n", 1)))
        results = [self.backend.generate(prompt_ids, config) for _ in range(n)]
        completion_tokens = sum(len(result.token_ids) for result in results)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.backend.model_name,
            "choices": [
                {
                    "index": index,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                    "logprobs": {
                        "content": [
                            {
                                "token": token,
                                "logprob": logprob,
                                "bytes": list(token.encode("utf-8")),
                                "top_logprobs": [],
                            }
                            for token, logprob in zip(result.token_texts, result.token_logprobs)
                        ]
                    },
                }
                for index, result in enumerate(results)
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_ids) + completion_tokens,
            },
        }


def make_handler(app: AdapterApp):
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._write(*app.get(self.path.split("?", 1)[0]))
            except APIError as exc:
                self._write(exc.status, {"error": {"message": str(exc), "type": "invalid_request_error"}})

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise APIError(400, "JSON body must be an object")
                self._write(*app.post(self.path.split("?", 1)[0], body))
            except APIError as exc:
                self._write(exc.status, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._write(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[rwkv-olmo] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="RWKV-7 .pth checkpoint")
    parser.add_argument("--served-model-name", default="rwkv7-g1x")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--engine-dir", default=None)
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--wkv", choices=("fp16", "fp32io16"), default="fp32io16")
    parser.add_argument("--emb", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--bos-policy", choices=("always", "special"), default="always")
    args = parser.parse_args()

    backend = AlbatrossBackend(
        args.model,
        model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        engine_dir=args.engine_dir,
        vocab_path=args.vocab,
        wkv=args.wkv,
        emb=args.emb,
        always_add_bos=args.bos_policy == "always",
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(AdapterApp(backend)))
    print(f"[rwkv-olmo] serving {backend.model_name} at http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
