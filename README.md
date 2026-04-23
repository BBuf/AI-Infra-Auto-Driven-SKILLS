# AI-Infra-Auto-Driven-SKILLS

Agent skills for SGLang/vLLM/TensorRT-LLM development, profiling, and production incident triage.

## Structure

```
skills/
├── model-optimization/            # model-family optimization handbook series
│   ├── model-pr-diff-dossier/     # shared per-PR dossier production standard
│   ├── sglang/                    # SGLang model-family skills
│   └── vllm/                      # reserved for future vLLM model-family skills
├── llm-serving-auto-benchmark/    # framework-neutral serving benchmark playbook
│   ├── SKILL.md
│   ├── agents/
│   ├── configs/cookbook-llm/
│   ├── references/
│   └── scripts/                   # validate_cookbook_configs.py, compare_benchmark_results.py
├── llm-torch-profiler-analysis/   # unified torch-profiler triage for SGLang / vLLM / TensorRT-LLM
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/                   # analyze_llm_torch_profile.py (preferred) + helpers + host runners
├── sglang-prod-incident-triage/   # replay-first debug flow for SGLang serving
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/                   # incident_artifact_tool.py, replay_trusted_request_dump.py
├── h100/                          # operator skill for the h100_sglang host
│   └── SKILL.md
└── h100-sglang-diffusion/         # h100 operator skill with diffusion-specific overrides
    └── SKILL.md
```

Run each skill's `ls` to see its exact current file set; this overview is a
high-level map, not a line-level inventory.

Model PR histories are framework-scoped:

```
model-pr-optimization-history/
├── sglang/
│   ├── model-skill-pr-dossier-quality-scan-2026-04-23.md
│   ├── deepseek-v3-r1/
│   ├── qwen3-core/
│   ├── glm45/
│   └── ...
└── vllm/
    └── README.md
```

## Placeholders

The `h100` and `h100-sglang-diffusion` skills document a concrete remote
environment (SSH alias `h100_sglang`, container `sglang_bbuf`, repo paths
`/sgl-workspace/sglang` and `/data/bbuf/repos/sglang`) because they are the
operator's own runbooks. Only secret-shaped values are templated with
placeholders that you must replace before running:

| Placeholder       | Meaning                                                 |
| ----------------- | ------------------------------------------------------- |
| `<your-hf-token>` | Hugging Face access token (never commit the real value) |

When adapting these skills to a different host/container/repo layout, copy the
SKILL and replace the concrete SSH alias, Docker name, and workspace path in
one pass rather than introducing generic `<...>` placeholders that drift out of
sync.

## Model Optimization Skills

The model optimization handbook series keeps shared production rules at the `model-optimization/` root, then splits framework-specific model-family skills by serving framework. This keeps SGLang and future vLLM model dossiers organized without duplicating common requirements.

SGLang model optimization skills live under `skills/model-optimization/sglang/`:

- `sglang-deepseek-v3-r1-optimization`
- `sglang-deepseek-v31-optimization`
- `sglang-deepseek-v32-optimization`
- `sglang-kimi-k2-k25-optimization`
- `sglang-minimax-m2-series-optimization`
- `sglang-qwen3-core-optimization`
- `sglang-qwen3-next-optimization`
- `sglang-qwen35-optimization`
- `sglang-qwen36-optimization`
- `sglang-qwen3-coder-optimization`
- `sglang-qwen-vlm-omni-asr-optimization`
- `sglang-qwen-image-optimization`
- `sglang-glm45-optimization`
- `sglang-glm46-glm47-optimization`
- `sglang-glm5-glm51-optimization`
- `sglang-glm-vlm-ocr-optimization`

The shared `skills/model-optimization/model-pr-diff-dossier/` skill records the mandatory production standard for model PR histories: read every PR diff and write motivation, implementation, code excerpt, and validation/risk.

Future vLLM model optimization skills should live under `skills/model-optimization/vllm/` and follow the same per-PR dossier standard.

## Model PR Optimization History

The `model-pr-optimization-history/` directory is framework-scoped.

SGLang bilingual model evolution notes live under `model-pr-optimization-history/sglang/`:

- `deepseek-v3-r1`
- `deepseek-v31`
- `deepseek-v32`
- `kimi`
- `minimax`
- `qwen3-core`
- `qwen3-next`
- `qwen35`
- `qwen36`
- `qwen3-coder`
- `qwen-vlm-omni-asr`
- `qwen-image`
- `glm45`
- `glm46-glm47`
- `glm5-glm51`
- `glm-vlm-ocr`

Cross-family audits sit next to those directories (not as a family of their own):

- `model-pr-optimization-history/sglang/model-skill-pr-dossier-quality-scan-2026-04-23.md`

Future vLLM model histories should live under `model-pr-optimization-history/vllm/`.

## Install

Copy the desired skill directory into your local skill path:

```bash
cp -r skills/sglang-prod-incident-triage <agent-skill-dir>/sglang-prod-incident-triage
cp -r skills/llm-torch-profiler-analysis <agent-skill-dir>/llm-torch-profiler-analysis
cp -r skills/model-optimization/sglang/sglang-qwen3-core-optimization <agent-skill-dir>/sglang-qwen3-core-optimization
```
