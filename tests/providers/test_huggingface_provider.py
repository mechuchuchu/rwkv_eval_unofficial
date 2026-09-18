"""Unit tests for HuggingFaceProvider generate-kwargs construction."""

from types import SimpleNamespace
from typing import Any

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.huggingface import HuggingFaceProvider


@pytest.fixture
def provider() -> HuggingFaceProvider:
    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))
    return instance


def test_finite_max_tokens_passes_through(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=512), prompt_len=100)
    assert kwargs["max_new_tokens"] == 512


def test_uncapped_reserves_room_after_prompt(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=2000)
    assert kwargs["max_new_tokens"] == 2048 - 2000


def test_uncapped_with_no_prompt_uses_full_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None))
    assert kwargs["max_new_tokens"] == 2048


def test_uncapped_floors_at_one_when_prompt_exceeds_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=5000)
    assert kwargs["max_new_tokens"] == 1


class _FakeBatchEncoding(dict[str, Any]):
    def to(self, device: Any) -> "_FakeBatchEncoding":
        return _FakeBatchEncoding(
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in self.items()
            }
        )


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(
        self, texts: list[str], *, padding: bool, return_tensors: str
    ) -> _FakeBatchEncoding:
        import torch

        assert padding is True
        assert return_tensors == "pt"
        encoded = [self.encode(text, add_special_tokens=False) for text in texts]
        width = max(len(row) for row in encoded)
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, token_ids in enumerate(encoded):
            offset = width - len(token_ids)
            input_ids[row, offset:] = torch.tensor(token_ids)
            attention_mask[row, offset:] = 1
        return _FakeBatchEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) % 20 + 1 for char in text]

    def decode(self, token: Any, *, skip_special_tokens: bool) -> str:
        if hasattr(token, "tolist"):
            token = token.tolist()
        if isinstance(token, list):
            return "".join(f"t{int(item)}" for item in token)
        return f"t{int(token)}"


class _FakeModel:
    generation_config = SimpleNamespace(eos_token_id=99)

    def __init__(self) -> None:
        import torch

        self.generate_batch_sizes: list[int] = []
        self.forward_batch_sizes: list[int] = []
        self.forward_attention_masks: list[Any] = []
        self.config = SimpleNamespace(max_position_embeddings=32)
        self._torch = torch

    def generate(self, input_ids: Any, attention_mask: Any, **kwargs: Any) -> Any:
        torch = self._torch
        self.generate_batch_sizes.append(input_ids.shape[0])
        assert attention_mask.shape == input_ids.shape
        assert kwargs["max_new_tokens"] == 2
        generated = torch.full((input_ids.shape[0], 2), 7, dtype=torch.long)
        return torch.cat([input_ids, generated], dim=1)

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        torch = self._torch
        self.forward_batch_sizes.append(input_ids.shape[0])
        self.forward_attention_masks.append(attention_mask.detach().clone())
        vocab = torch.arange(32, dtype=torch.float32, device=input_ids.device)
        logits = vocab.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


@pytest.fixture
def batch_provider() -> HuggingFaceProvider:
    pytest.importorskip("torch")
    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.model = _FakeModel()
    instance.tokenizer = _FakeTokenizer()
    instance.device = instance.model._torch.device("cpu")
    instance.is_multimodal = False
    instance.batch_size = 2
    return instance


def test_generate_batches_requests_and_scores_each_batch(
    batch_provider: HuggingFaceProvider,
) -> None:
    requests = [
        LMRequest(request_type=RequestType.COMPLETION, prompt="a"),
        LMRequest(request_type=RequestType.COMPLETION, prompt="long"),
        LMRequest(request_type=RequestType.COMPLETION, prompt="z"),
    ]

    outputs = batch_provider.generate(requests, SamplingParams(max_tokens=2))

    model = batch_provider.model
    assert model.generate_batch_sizes == [2, 1]
    assert model.forward_batch_sizes == [2, 1]
    assert batch_provider.tokenizer.padding_side == "left"
    assert [len(request_outputs) for request_outputs in outputs] == [1, 1, 1]
    assert all(
        output.logprobs is not None for request_outputs in outputs for output in request_outputs
    )


def test_logprobs_batches_flattened_continuations_and_restores_request_order(
    batch_provider: HuggingFaceProvider,
) -> None:
    requests = [
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="a",
            continuations=("x", "y"),
        ),
        LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="long",
            continuations=("z",),
        ),
    ]

    outputs = batch_provider.logprobs(requests)

    model = batch_provider.model
    assert model.forward_batch_sizes == [2, 1]
    assert [len(request_outputs) for request_outputs in outputs] == [2, 1]
    assert [output.text for output in outputs[0]] == ["x", "y"]
    assert [output.text for output in outputs[1]] == ["z"]
    assert all(output.logprobs for request_outputs in outputs for output in request_outputs)
