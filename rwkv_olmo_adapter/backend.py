"""Backend protocol and the concrete Albatross faster3a implementation."""

from __future__ import annotations

import importlib
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .tokenizer import RWKVTokenizer


@dataclass(frozen=True)
class SamplingConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    token_texts: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    finish_reason: str
    prompt_tokens: int


class Backend(Protocol):
    model_name: str
    max_model_len: int
    eos_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def generate(self, prompt_ids: list[int], config: SamplingConfig) -> GenerationResult: ...
    def prompt_logprobs(self, token_ids: list[int], top_k: int) -> list[dict[str, dict[str, float | int]] | None]: ...


class AlbatrossBackend:
    """Thin correctness-first wrapper around faster3a_2605.RWKV7."""

    def __init__(
        self,
        model_path: str,
        *,
        model_name: str = "rwkv7-g1x",
        max_model_len: int = 8192,
        engine_dir: str | Path | None = None,
        vocab_path: str | Path | None = None,
        wkv: str = "fp32io16",
        emb: str = "cpu",
        always_add_bos: bool = True,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        engine = Path(engine_dir) if engine_dir else root / "faster3a_2605"
        vocab = Path(vocab_path) if vocab_path else root.parent / "RWKV-LM" / "RWKV-v7" / "rwkv_vocab_v20230424.txt"
        if not vocab.exists():
            fallback = root / "faster2_251201" / "reference" / "rwkv_vocab_v20230424.txt"
            vocab = fallback

        sys.path.insert(0, str(engine))
        v3a = importlib.import_module("rwkv7_fast_v3a")
        v3a.MODEL_PATH = model_path
        v3a.WKV_MODE = wkv
        v3a.EMB_DEVICE = emb
        v3a.RKV_MODE = "off"
        v3a.CMIX_SPARSE = "no-fc"
        v3a.LOWRANK_WEIGHT = "both"
        v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
        v3a.load_extensions(wkv)

        self._torch = importlib.import_module("torch")
        self._v3a = v3a
        self._model = v3a.RWKV7()
        self._tokenizer = RWKVTokenizer(vocab)
        self._lock = threading.Lock()
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.eos_token_id = 0
        self.always_add_bos = always_add_bos

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        tokens = self._tokenizer.encode(text)
        # RemoteTokenizer cannot discover RWKV's BOS ID, while official G1
        # evaluation requires token 0 at the start of each fresh prompt.
        should_add_bos = self.always_add_bos or add_special_tokens
        return ([self.eos_token_id] + tokens) if should_add_bos else tokens

    def decode(self, token_ids: list[int]) -> str:
        visible = [token for token in token_ids if token != self.eos_token_id]
        return self._tokenizer.decode(visible)

    def _token_tensor(self, token_ids: list[int]):
        device = "cpu" if self._model.emb_cpu else self._v3a.first_device()
        return self._torch.tensor(token_ids, dtype=self._torch.long, device=device).view(1, -1)

    def _sample(self, logits, config: SamplingConfig) -> tuple[int, float, dict[str, float]]:
        torch = self._torch
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        k = min(config.top_k or logits.numel(), logits.numel())
        if config.temperature <= 0:
            token_id = int(torch.argmax(logits).item())
        else:
            values, indices = torch.topk(logits.float(), k=k, sorted=True)
            probs = torch.softmax(values / config.temperature, dim=-1)
            if config.top_p < 1.0:
                cumulative = torch.cumsum(probs, dim=-1)
                cutoff = int(torch.searchsorted(cumulative, torch.tensor(config.top_p, device=cumulative.device)).item()) + 1
                probs, indices = probs[:cutoff], indices[:cutoff]
                probs = probs / probs.sum()
            token_id = int(indices[torch.multinomial(probs, 1)].item())
        top_values, top_ids = torch.topk(log_probs, k=min(5, log_probs.numel()))
        top = {self.decode([int(i)]): float(v) for i, v in zip(top_ids.tolist(), top_values.tolist())}
        return token_id, float(log_probs[token_id].item()), top

    def generate(self, prompt_ids: list[int], config: SamplingConfig) -> GenerationResult:
        if not prompt_ids:
            prompt_ids = [self.eos_token_id]
        prompt_ids = prompt_ids[-self.max_model_len :]
        with self._lock, self._torch.inference_mode():
            state = self._model.zero_state(1)
            logits = self._model.forward(self._token_tensor(prompt_ids), state).view(-1)
            generated: list[int] = []
            token_texts: list[str] = []
            token_logprobs: list[float] = []
            top_logprobs: list[dict[str, float]] = []
            text = ""
            finish_reason = "length"
            for _ in range(min(config.max_tokens, self.max_model_len - len(prompt_ids))):
                token_id, logprob, top = self._sample(logits, config)
                if token_id == self.eos_token_id:
                    finish_reason = "stop"
                    break
                generated.append(token_id)
                token_texts.append(self.decode([token_id]))
                token_logprobs.append(logprob)
                top_logprobs.append(top)
                decoded = self.decode(generated)
                hits = [(decoded.find(stop), stop) for stop in config.stop if stop and stop in decoded]
                if hits:
                    offset, _ = min(hits, key=lambda item: item[0])
                    text = decoded[:offset]
                    finish_reason = "stop"
                    break
                text = decoded
                logits = self._model.forward(self._token_tensor([token_id]), state).view(-1)
        return GenerationResult(generated, text, token_texts, token_logprobs, top_logprobs, finish_reason, len(prompt_ids))

    def prompt_logprobs(self, token_ids: list[int], top_k: int) -> list[dict[str, dict[str, float | int]] | None]:
        if not token_ids:
            return []
        torch = self._torch
        with self._lock, torch.inference_mode():
            state = self._model.zero_state(1)
            logits = self._model.forward_all_logits(self._token_tensor(token_ids), state).squeeze(0).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            result: list[dict[str, dict[str, float | int]] | None] = [None]
            for position, chosen_id in enumerate(token_ids[1:]):
                row = log_probs[position]
                values, ids = torch.topk(row, k=min(max(1, top_k), row.numel()))
                entry = {
                    str(int(token_id)): {"logprob": float(value), "rank": rank}
                    for rank, (token_id, value) in enumerate(zip(ids.tolist(), values.tolist()), 1)
                }
                key = str(chosen_id)
                if key not in entry:
                    rank = int((row > row[chosen_id]).sum().item()) + 1
                    entry[key] = {"logprob": float(row[chosen_id].item()), "rank": rank}
                result.append(entry)
            return result
