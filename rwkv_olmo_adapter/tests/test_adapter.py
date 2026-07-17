from rwkv_olmo_adapter.backend import GenerationResult, SamplingConfig
from rwkv_olmo_adapter.g1x import clean_text, render_g1x
from rwkv_olmo_adapter.server import AdapterApp


class FakeBackend:
    model_name = "rwkv7-g1x"
    max_model_len = 8192
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens=True):
        tokens = list(text.encode())
        return [0, *tokens] if add_special_tokens else tokens

    def decode(self, token_ids):
        return bytes(x for x in token_ids if x).decode(errors="replace")

    def generate(self, prompt_ids, config):
        return GenerationResult(
            token_ids=[79, 75], text="OK", token_texts=["O", "K"],
            token_logprobs=[-0.1, -0.2], top_logprobs=[{"O": -0.1}, {"K": -0.2}],
            finish_reason="stop", prompt_tokens=len(prompt_ids),
        )

    def prompt_logprobs(self, token_ids, top_k):
        return [None, *[
            {str(token_id): {"logprob": -0.25, "rank": 1}}
            for token_id in token_ids[1:]
        ]]


def test_g1x_rendering_and_cleaning():
    assert clean_text(" a\r\n\r\n\r\nb ") == "a\nb"
    assert render_g1x([
        {"role": "system", "content": " concise "},
        {"role": "user", "content": "hello\n\n\nworld"},
    ]) == "System: concise\n\nUser: hello\nworld\n\nAssistant:"


def test_tokenize_contract():
    status, payload = AdapterApp(FakeBackend()).post(
        "/tokenize", {"model": "rwkv7-g1x", "prompt": "A", "add_special_tokens": True}
    )
    assert status == 200
    assert payload["tokens"] == [0, 65]


def test_completion_contract():
    status, payload = AdapterApp(FakeBackend()).post(
        "/v1/completions",
        {"model": "rwkv7-g1x", "prompt": "hi", "max_tokens": 2, "logprobs": 1},
    )
    assert status == 200
    assert payload["choices"][0]["text"] == "OK"
    assert payload["choices"][0]["logprobs"]["token_logprobs"] == [-0.1, -0.2]


def test_prompt_logprobs_contract():
    status, payload = AdapterApp(FakeBackend()).post(
        "/v1/completions",
        {"model": "rwkv7-g1x", "prompt": [0, 65, 66], "max_tokens": 1,
         "prompt_logprobs": 5, "add_special_tokens": False},
    )
    assert status == 200
    assert payload["choices"][0]["prompt_logprobs"] == [
        None,
        {"65": {"logprob": -0.25, "rank": 1}},
        {"66": {"logprob": -0.25, "rank": 1}},
    ]


def test_chat_uses_g1x_and_role_stops():
    backend = FakeBackend()
    original_generate = backend.generate
    captured = {}

    def generate(prompt_ids, config):
        captured["prompt"] = backend.decode(prompt_ids)
        captured["stops"] = config.stop
        return original_generate(prompt_ids, config)

    backend.generate = generate
    status, payload = AdapterApp(backend).post(
        "/v1/chat/completions",
        {"model": "rwkv7-g1x", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert status == 200
    assert captured["prompt"] == "User: hello\n\nAssistant:"
    assert "\n\nUser:" in captured["stops"]
    assert payload["choices"][0]["message"]["content"] == "OK"
