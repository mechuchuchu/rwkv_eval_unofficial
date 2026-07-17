# RWKV–OLMo Eval adapter

This package exposes an Albatross RWKV-7 G1 checkpoint through the subset of
the vLLM OpenAI API used by `olmo-eval`: model discovery, tokenization,
completions, chat completions, and vLLM `prompt_logprobs`.

The first version prioritizes evaluation parity: GPU calls and multiple samples
are processed sequentially. The separate `Backend` protocol allows a dynamic
batch scheduler to replace this later without changing the HTTP API.

## Start

From `/workspace/Albatross`:

```bash
python -m rwkv_olmo_adapter \
  --model /dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth \
  --served-model-name rwkv7-g1x \
  --host 127.0.0.1 --port 8000 --max-model-len 8192
```

Defaults use the accuracy-oriented `fp32io16` WKV path and the vocabulary at
`/workspace/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt`.
The default `--bos-policy always` adds RWKV token `0` to every fresh prompt,
matching the official RWKV evaluation scripts even though OLMo Eval's remote
tokenizer cannot discover a BOS ID.

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rwkv7-g1x","prompt":"User: Hello\n\nAssistant:","max_tokens":16,"temperature":0}'
```

## Connect OLMo Eval

Harness overrides must occur after `--harness` and before the task:

```bash
cd /workspace/olmo-eval
uv run olmo-eval run \
  --harness default \
  -o provider.kind=vllm_server \
  -o provider.base_url=http://127.0.0.1:8000/v1 \
  -o provider.model=rwkv7-g1x \
  -o provider.max_model_len=8192 \
  -o provider.max_concurrency=1 \
  -o provider.kwargs.completion_use_prompt_token_ids=true \
  -o 'metrics.collect_vllm_server=false' \
  -m rwkv7-g1x \
  -t mmlu_abstract_algebra:mc -o limit=20 \
  --save-predictions --save-requests
```

`mmlu:mc` validates log-likelihood transport, but exact parity with the official
RWKV evaluator still needs an OLMo Eval G1x MMLU formatter matching
`faster3a_2605/eval_mmlu.py`.

Chat prompts use the official G1x textual template. They stop on complete role
boundaries such as `\n\nUser:`, not on a bare double newline.

## Tests

Contract tests use a fake backend and require no CUDA:

```bash
cd /workspace/Albatross
python -m pytest rwkv_olmo_adapter/tests -q
```
