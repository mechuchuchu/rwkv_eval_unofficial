# Evaluating RWKV-7 with Albatross and OLMo Eval

This guide evaluates an RWKV-7 G1-series checkpoint with:

- **Albatross** as the RWKV inference engine; **Please note that the repository name has been changed from 'albatross' to 'rwkv eval unofficial'.**
- **RWKV–OLMo adapter** as a vLLM-compatible HTTP bridge; and
- **OLMo Eval** as the task, scoring, and result-management framework.

The examples below use the checkpoint already present on this machine:

```text
/workspace/rwkv7-g1h-1.5b-20260710-ctx10240.pth
```

The adapter serves it as `rwkv7-g1h-1.5b` with a maximum context length of
10,240 tokens.

## 1. Architecture

```text
OLMo Eval
  │
  │ OpenAI/vLLM-compatible HTTP
  │ /v1/models, /tokenize, /detokenize,
  │ /v1/completions, /v1/chat/completions
  ▼
RWKV–OLMo adapter
  │
  ├─ G1x chat rendering
  ├─ RWKV tokenization and BOS handling
  ├─ stop-sequence handling
  └─ vLLM prompt_logprobs conversion
  │
  ▼
Albatross faster3a_2605
  │
  ▼
RWKV-7 G1 checkpoint
```

There are two evaluation paths:

1. **Generation:** OLMo Eval sends a completion or chat request and scores the
   generated text.
2. **Log-likelihood:** OLMo Eval sends token IDs and requests vLLM-style
   `prompt_logprobs`. This is used by multiple-choice and ranking tasks.

## 2. Files used in this workspace

```text
/workspace/
├── rwkv7-g1h-1.5b-20260710-ctx10240.pth
├── Albatross/
│   ├── faster3a_2605/
│   └── rwkv_olmo_adapter/
└── olmo-eval/
```

Important adapter files:

```text
rwkv_olmo_adapter/
├── backend.py       # Albatross model wrapper and prompt log probabilities
├── g1x.py           # Official G1x text-template renderer
├── server.py        # OpenAI/vLLM-compatible HTTP API
├── tokenizer.py     # RWKV world tokenizer
└── tests/           # CUDA-free API contract tests
```

## 3. Check the environment

Confirm that the GPU and checkpoint are visible:

```bash
nvidia-smi
ls -lh /workspace/rwkv7-g1h-1.5b-20260710-ctx10240.pth
```

Confirm PyTorch CUDA support:

```bash
source /venv/main/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("wheel CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

The first adapter startup compiles Albatross CUDA extensions. This may take
several minutes. Later starts can reuse the extension cache.

## 4. Install OLMo Eval

OLMo Eval's default dependency groups include local vLLM and large CUDA wheels.
They are unnecessary when inference is provided by the external Albatross
server. A smaller installation is:

```bash
cd /workspace/olmo-eval
uv sync --frozen --no-default-groups --extra clients
```

Use the full repository default only if local vLLM evaluation is also needed:

```bash
uv sync --frozen
```

All subsequent OLMo Eval commands in this guide use `uv run --no-sync` so they
do not unexpectedly change the environment:

```bash
uv run --no-sync olmo-eval --help
```

## 5. Start the Albatross adapter

Open terminal 1:

```bash
source /venv/main/bin/activate
cd /workspace/Albatross

python -m rwkv_olmo_adapter \
  --model /workspace/rwkv7-g1h-1.5b-20260710-ctx10240.pth \
  --served-model-name rwkv7-g1h-1.5b \
  --max-model-len 10240 \
  --host 127.0.0.1 \
  --port 8000 \
  --wkv fp32io16 \
  --emb cpu \
  --bos-policy always
```

Recommended evaluation defaults:

| Option | Recommended value | Reason |
|---|---:|---|
| `--wkv` | `fp32io16` | More accurate recurrent state path |
| `--emb` | `cpu` | Saves GPU memory |
| `--bos-policy` | `always` | Matches official RWKV evaluation scripts |
| `--host` | `127.0.0.1` | Keeps the unauthenticated API private |

`--bos-policy always` is important. OLMo Eval's remote tokenizer cannot discover
RWKV's BOS ID, while official RWKV evaluation prepends token `0` to every fresh
prompt. The adapter compensates for that limitation.

For faster exploratory runs, `--wkv fp16` can be tested, but final reported
scores should use `fp32io16` unless parity tests show no meaningful difference.

## 6. Smoke-test the server

Open terminal 2.

### Health and model discovery

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

The model response should include:

```json
{
  "id": "rwkv7-g1h-1.5b",
  "max_model_len": 10240
}
```

### Tokenizer round trip

```bash
curl -s http://127.0.0.1:8000/tokenize \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rwkv7-g1h-1.5b",
    "prompt": "User: Hello\n\nAssistant:",
    "add_special_tokens": false
  }' | python -m json.tool
```

The first token should be `0` under the default BOS policy.

### Deterministic completion

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rwkv7-g1h-1.5b",
    "prompt": "User: What is RWKV?\n\nAssistant:",
    "max_tokens": 64,
    "temperature": 0,
    "stop": ["\n\nUser:", "\n\nSystem:", "\n\nAssistant:"]
  }' | python -m json.tool
```

### G1x chat completion

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rwkv7-g1h-1.5b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is RWKV?"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }' | python -m json.tool
```

The adapter renders those messages as:

```text
System: Answer concisely.

User: What is RWKV?

Assistant:
```

Chat generation stops on complete role boundaries such as `\n\nUser:`. A bare
double newline is deliberately not a stop because it is also a normal paragraph
separator.

### Prompt log probabilities

First obtain tokens:

```bash
curl -s http://127.0.0.1:8000/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"model":"rwkv7-g1h-1.5b","prompt":"The answer is A"}'
```

Then send that integer array as `prompt`:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rwkv7-g1h-1.5b",
    "prompt": [0, 123, 456],
    "max_tokens": 1,
    "temperature": 0,
    "prompt_logprobs": 5,
    "add_special_tokens": false
  }' | python -m json.tool
```

Replace the example token IDs with IDs returned by `/tokenize`. The response's
`prompt_logprobs` list must have the same length as the input token array. Its
first element is `null`; each later element scores that token from the previous
position's logits.

## 7. Inspect tasks before evaluation

From `/workspace/olmo-eval`:

```bash
cd /workspace/olmo-eval

uv run --no-sync olmo-eval suite inspect mmlu
uv run --no-sync olmo-eval task inspect mmlu_abstract_algebra:mc --request -n 1
```

Inspection is useful for checking:

- prompt wording;
- few-shot examples;
- continuation choices;
- whether the task uses generation or log-likelihood; and
- whether the task expects a chat or completion request.

## 8. Run a small log-likelihood evaluation

Start with one MMLU subject and a small limit:

```bash
cd /workspace/olmo-eval

uv run --no-sync olmo-eval run \
  --harness default \
  -o provider.kind=vllm_server \
  -o provider.base_url=http://127.0.0.1:8000/v1 \
  -o provider.model=rwkv7-g1h-1.5b \
  -o provider.max_model_len=10240 \
  -o provider.max_concurrency=1 \
  -o provider.kwargs.completion_use_prompt_token_ids=true \
  -o 'metrics.collect_vllm_server=false' \
  -m rwkv7-g1h-1.5b \
  -t mmlu_abstract_algebra:mc \
  -o limit=20 \
  -O /workspace/results/rwkv7-g1h-1.5b/mmlu-smoke \
  --save-predictions \
  --save-requests \
  --inspect-request \
  --inspect-response
```

Important settings:

- `provider.kind=vllm_server` selects OLMo Eval's vLLM-compatible HTTP client.
- `provider.base_url=.../v1` prevents OLMo Eval from launching local vLLM.
- `completion_use_prompt_token_ids=true` preserves exact RWKV tokenization for
  completion requests.
- `provider.max_concurrency=1` matches the adapter's correctness-first serialized
  GPU execution.
- `metrics.collect_vllm_server=false` prevents polling a vLLM `/metrics` endpoint
  that this adapter does not expose.

If this succeeds, remove `limit=20` or run the complete suite:

```bash
uv run --no-sync olmo-eval run \
  --harness default \
  -o provider.kind=vllm_server \
  -o provider.base_url=http://127.0.0.1:8000/v1 \
  -o provider.model=rwkv7-g1h-1.5b \
  -o provider.max_model_len=10240 \
  -o provider.max_concurrency=1 \
  -o provider.kwargs.completion_use_prompt_token_ids=true \
  -o 'metrics.collect_vllm_server=false' \
  -m rwkv7-g1h-1.5b \
  -t mmlu \
  -O /workspace/results/rwkv7-g1h-1.5b/mmlu
```

## 9. Run generation evaluations

The same provider configuration works for generation tasks. First inspect the
task, then run a small deterministic sample:

```bash
uv run --no-sync olmo-eval task inspect gsm8k --request -n 1

uv run --no-sync olmo-eval run \
  --harness default \
  -o provider.kind=vllm_server \
  -o provider.base_url=http://127.0.0.1:8000/v1 \
  -o provider.model=rwkv7-g1h-1.5b \
  -o provider.max_model_len=10240 \
  -o provider.max_concurrency=1 \
  -o provider.kwargs.completion_use_prompt_token_ids=true \
  -o 'metrics.collect_vllm_server=false' \
  -m rwkv7-g1h-1.5b \
  -t gsm8k \
  -o limit=20 \
  -O /workspace/results/rwkv7-g1h-1.5b/gsm8k-smoke \
  --save-predictions --save-requests
```

Do not assume that every task's default prompt is ideal for G1x. Inspect saved
requests and confirm whether the task should use raw completion formatting or
the G1x chat template.

## 10. G1x reasoning modes

The official G1x template supports these assistant prefixes:

| Mode | Prefix |
|---|---|
| Normal | `Assistant:` |
| Thinking | `Assistant: <think` |
| Fast fake thinking | `Assistant: <think>\n</think` |
| Compact fake thinking | `Assistant: <think></think` |

For direct chat API calls, pass the mode through `chat_template_kwargs`:

```json
{
  "model": "rwkv7-g1h-1.5b",
  "messages": [{"role": "user", "content": "Solve this problem."}],
  "extra_body": {
    "chat_template_kwargs": {
      "think_mode": "think"
    }
  }
}
```

Supported adapter values are `think`, `fake`, `fake_compact`, and `json`.
The apparently incomplete tags are intentional prefixes from the official G1x
template; do not automatically append the closing `>`.

## 11. Exact parity with the official RWKV MMLU evaluator

An OLMo Eval `mmlu:mc` score is not automatically identical to the score from
`Albatross/faster3a_2605/eval_mmlu.py`. The inference logits may be identical
while the benchmark protocol differs.

The official RWKV MMLU prompt is:

```text
User: You are a very talented expert in {subject}. Answer this question:
{question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Assistant: The answer is
```

It then compares the next-token logits for the single tokens `" A"`, `" B"`,
`" C"`, and `" D"`.

For defensible parity testing:

1. Run `faster3a_2605/eval_mmlu.py` as the reference.
2. Use the same dataset revision, split, ordering, and choice-shuffling policy.
3. Use `fp32io16`, BOS token `0`, and the exact prompt above.
4. Add a dedicated OLMo Eval `mmlu:g1x` formatter/variant reproducing that
   prompt and one-token choice scoring.
5. Compare per-instance predictions, not only aggregate accuracy.
6. Accept the OLMo Eval integration only after logits or choice log-probability
   differences are within the selected numeric tolerance.

Until that variant exists, treat `mmlu:mc` as an integration and comparative
benchmark, not as exact reproduction of the official RWKV MMLU number.

## 12. Result handling

Always use a separate output directory for each checkpoint and evaluation
configuration:

```text
/workspace/results/
└── rwkv7-g1h-1.5b/
    ├── mmlu-smoke/
    ├── mmlu/
    └── gsm8k-smoke/
```

Retain at least:

- aggregate metrics;
- per-instance predictions;
- serialized requests;
- checkpoint filename and hash;
- adapter command line;
- OLMo Eval commit;
- Albatross commit;
- WKV mode and BOS policy; and
- sampling parameters and random seeds.

Record hashes and commits with:

```bash
sha256sum /workspace/rwkv7-g1h-1.5b-20260710-ctx10240.pth
git -C /workspace/Albatross rev-parse HEAD
git -C /workspace/olmo-eval rev-parse HEAD
```

## 13. Recommended evaluation progression

Use this order to avoid wasting time on a full suite with a protocol error:

1. Run the adapter's CUDA-free contract tests.
2. Start the real checkpoint and test `/health` and `/v1/models`.
3. Verify tokenizer round trips and leading BOS `0`.
4. Run one deterministic completion.
5. Test a manually tokenized `prompt_logprobs` request.
6. Run one MMLU subject with `limit=20`.
7. Inspect saved requests and predictions.
8. Compare a few examples against direct Albatross logits.
9. Run the full MMLU suite.
10. Add generation and reasoning benchmarks.
11. Implement and validate `mmlu:g1x` before claiming official parity.

Run the adapter contract tests without CUDA:

```bash
cd /workspace/Albatross
python -m pytest rwkv_olmo_adapter/tests -q
```

If `pytest` is not installed in `/venv/main`, either install it or run the tests
from an environment that already includes it. The adapter itself has no FastAPI
or Uvicorn dependency; its HTTP server uses the Python standard library.

## 14. Troubleshooting

### OLMo Eval tries to start vLLM

The provider has no effective external URL. Confirm that the harness overrides
appear immediately after `--harness` and that the URL ends in `/v1`:

```text
provider.kind=vllm_server
provider.base_url=http://127.0.0.1:8000/v1
```

### OLMo Eval polls `/metrics`

Set:

```text
metrics.collect_vllm_server=false
```

### Model not found

The request model must exactly equal `--served-model-name`. Check:

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

### Incorrect or shifted log-likelihood scores

Check all of the following:

- token `0` is present exactly once at the beginning;
- `/tokenize` and Albatross use the same vocabulary;
- `prompt_logprobs` has one entry per prompt token;
- its first entry is `null`;
- logits at position `i-1` score token `i`; and
- the chosen token is included even when it is outside the requested top-k.

### Responses stop after one paragraph

Do not use bare `\n\n` as a stop sequence. Use complete G1x role boundaries:

```text
\n\nUser:
\n\nSystem:
\n\nAssistant:
```

### CUDA out of memory

- Keep `--emb cpu`.
- Use `provider.max_concurrency=1`.
- Reduce context length and generation length.
- Ensure no other process occupies the GPU.
- Avoid starting local vLLM alongside Albatross.

### First startup is slow

Albatross compiles custom CUDA extensions on first use. Watch the server
terminal for compilation errors and allow the initial build to finish.

### Numerical differences between runs

- Use `--wkv fp32io16` for final evaluation.
- Use `temperature=0` for deterministic generation.
- Pin task variants, dataset revisions, seeds, and repository commits.
- Compare per-instance log probabilities before comparing aggregate metrics.

## 15. Current adapter limitations

The current adapter is a correctness-first implementation:

- GPU requests are serialized with a lock;
- multiple samples (`n > 1`) are generated sequentially;
- continuous batching is not implemented yet;
- streaming responses are not implemented;
- vLLM's Prometheus `/metrics` endpoint is not implemented; and
- exact official MMLU parity still requires an OLMo Eval G1x task variant.

These limitations affect throughput and benchmark protocol coverage, but the
backend/API boundary allows a dynamic Albatross scheduler to be added later
without changing OLMo Eval commands.
