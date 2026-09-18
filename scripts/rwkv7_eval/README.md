# RWKV-7 OLMo Eval wrappers

This directory contains execution wrappers that connect the local Hugging Face
snapshot `RWKV/RWKV7-G1j-1.5B-20260831` to the tasks and suites in
`olmo-eval`. The model is loaded through the Transformers-based `hf` provider
with `trust_remote_code=true` and `bfloat16` by default.

## Setup

The wrappers default to the model path available on this instance.

```bash
source /venv/main/bin/activate
cd /workspace/olmo-eval

export MODEL_PATH=/workspace/.hf_home/hub/models--RWKV--RWKV7-G1j-1.5B-20260831/snapshots/2c18b29ab7fbece25ff6112281eea0fa41fcb30f
export HF_HOME=/workspace/.hf_home
```

Override the model path or output directory with environment variables:

```bash
export MODEL_PATH=/path/to/model
export OUTPUT_ROOT=/workspace/results/my-run
export DTYPE=bfloat16
export BATCH_SIZE=8
```

Text requests are processed in batches by the Hugging Face provider. Lower
`BATCH_SIZE` if a benchmark's prompts or generation lengths exceed available
GPU memory.

The launcher adds a `transformers>=5.15,<5.16` overlay because the bundled
RWKV-7 model code requires the multi-state cache API introduced in Transformers
5.15. This keeps the repository's default vLLM environment unchanged.

## Running evaluations

Run a single benchmark, for example ARC MC:

```bash
./scripts/rwkv7_eval/10_arc_mc.sh
```

For an initial smoke test, set `LIMIT=2`. For a suite, the limit is applied to
each child task in the suite.

```bash
LIMIT=2 ./scripts/rwkv7_eval/10_arc_mc.sh
```

Run all wrappers in order with:

```bash
./scripts/rwkv7_eval/run_all.sh
```

`run_all.sh` continues after individual benchmark failures, prints a failure
list at the end, and exits with code 1 if anything failed. Each benchmark
writes `metrics.json`, `predictions/`, and `requests/` under
`$OUTPUT_ROOT/<benchmark-name>/`.

Additional CLI options are forwarded to every wrapper:

```bash
LIMIT=2 ./scripts/rwkv7_eval/run_all.sh --dry-run
```

## Benchmark mapping

| Benchmark | Wrapper | olmo-eval task/suite | Harness |
|---|---|---|---|
| Olmo 3-Eval Math | `01_olmo3_eval_math.sh` | `olmobase:math` | default |
| BigCodeBench | `02_bigcodebench.sh` | `bigcodebench:olmo3base` | codex_universal |
| HumanEval | `03_humaneval.sh` | `humaneval:olmo3base` | codex_universal |
| DeepSeek LeetCode | `04_deepseek_leetcode.sh` | `deepseek_leetcode:olmo3base` | codex_universal |
| DS 1000 | `05_ds1000.sh` | `ds1000:olmo3base` | codex_universal |
| MBPP | `06_mbpp.sh` | `mbpp:olmo3base` | codex_universal |
| MultiPL HumanEval | `07_multipl_e_humaneval.sh` | `multipl_e_humaneval:olmo3base` | codex_universal |
| MultiPL MBPP | `08_multipl_e_mbpp.sh` | `multipl_e_mbpp:olmo3base` | codex_universal |
| Olmo 3-Eval Code | `09_olmo3_eval_code.sh` | `olmobase:code` | codex_universal |
| ARC MC | `10_arc_mc.sh` | `arc:mc:olmo3base` | default |
| MMLU STEM | `11_mmlu_stem.sh` | `mmlu:stem:mc:olmo3base` | default |
| MedMCQA MC | `12_medmcqa_mc.sh` | `medmcqa:mc:olmo3base` | default |
| MedQA MC | `13_medqa_mc.sh` | `medqa_en:mc:olmo3base` | default |
| SciQ MC | `14_sciq_mc.sh` | `sciq:mc:olmo3base` | default |
| Olmo 3-Eval MC_STEM | `15_olmo3_eval_mc_stem.sh` | `olmobase:mcqa_stem` | default |
| MMLU Humanities | `16_mmlu_humanities.sh` | `mmlu:humanities:mc:olmo3base` | default |
| MMLU Social Sci. | `17_mmlu_social_sciences.sh` | `mmlu:social_sciences:mc:olmo3base` | default |
| MMLU Other | `18_mmlu_other.sh` | `mmlu:other:mc:olmo3base` | default |
| CSQA MC | `19_csqa_mc.sh` | `csqa:mc_olmo3base` | default |
| PIQA MC | `20_piqa_mc.sh` | `piqa:mc_olmo3base` | default |
| SocialIQA MC | `21_socialiqa_mc.sh` | `socialiqa:mc_olmo3base` | default |
| CoQA Gen2MC MC | `22_coqa_gen2mc_mc.sh` | `coqa:mc:olmo3base` | default |
| DROP Gen2MC MC | `23_drop_gen2mc_mc.sh` | `drop:mc:olmo3base` | default |
| Jeopardy Gen2MC MC | `24_jeopardy_gen2mc_mc.sh` | `jeopardy:mc:olmo3base` | default |
| NaturalQs Gen2MC MC | `25_naturalqs_gen2mc_mc.sh` | `naturalqs:mc:olmo3base` | default |
| SQuAD Gen2MC MC | `26_squad_gen2mc_mc.sh` | `squad:mc:olmo3base` | default |
| Olmo 3-Eval MC_Non-STEM | `27_olmo3_eval_mc_non_stem.sh` | `olmobase:mcqa_non_stem` | default |
| HellaSwag RC | `28_hellaswag_rc.sh` | `hellaswag:rc:olmo3base` | default |
| Winogrande RC | `29_winogrande_rc.sh` | `winogrande:rc:olmo3base` | default |
| LAMBADA | `30_lambada.sh` | `lambada:olmo3base` | default |
| Basic Skills | `31_basic_skills.sh` | `basic_skills:rc:olmo3base` | default |
| DROP | `32_drop.sh` | `drop:gen:olmo3base` | default |
| Jeopardy | `33_jeopardy.sh` | `jeopardy:gen:olmo3base` | default |
| NaturalQs | `34_naturalqs.sh` | `naturalqs:gen:olmo3base` | default |
| SQuAD | `35_squad.sh` | `squad:gen:olmo3base` | default |
| CoQA | `36_coqa.sh` | `coqa:gen:olmo3base` | default |
| Olmo 3-Eval GenQA | `37_olmo3_eval_genqa.sh` | `olmobase:gen` | default |

## Benchmarks unavailable in this checkout

The following wrappers preserve the requested benchmark names and report a
clear error because the corresponding tasks or suites are not registered in
the current `olmo-eval` checkout:

- `38_bbh.sh`: BBH
- `39_mmlu_pro_mc.sh`: MMLU Pro MC
- `40_deepmind_math.sh`: DeepMind Math
- `41_lbpp.sh`: LBPP

They are intentionally not mapped to substitute tasks. Add the benchmark
implementation or an external task package, then fill in the mapping in each
wrapper.

## Code benchmark requirements

Code benchmarks use the Docker sandbox configured by the `codex_universal`
harness. If Docker sandbox execution is unavailable, the code wrappers may
fail during sandbox setup or scoring even though their task mappings and
prompts are valid. Prepare the sandbox runtime separately before running them.
