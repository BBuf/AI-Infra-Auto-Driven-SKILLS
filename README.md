# AI-Infra-Auto-Driven-SKILLS

Agent skills for SGLang/vLLM/TensorRT-LLM development, profiling, and
production incident triage.

## Structure

```text
skills/
├── model-optimization/            # model-family optimization handbook series
│   ├── model-pr-diff-dossier/     # shared per-PR dossier production standard
│   ├── sglang/                    # SGLang model-family skills
│   └── vllm/                      # vLLM model-family skills
├── llm-serving-auto-benchmark/    # framework-neutral serving benchmark playbook
│   ├── SKILL.md
│   ├── agents/
│   ├── configs/cookbook-llm/
│   ├── references/
│   └── scripts/
├── llm-torch-profiler-analysis/   # unified torch-profiler triage for SGLang / vLLM / TensorRT-LLM
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/
├── model-architecture-diagram/    # return upstream model structure diagrams or generate fallback SVGs
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/
├── sglang-prod-incident-triage/   # replay-first debug flow for SGLang serving
│   ├── SKILL.md
│   ├── agents/
│   ├── references/
│   └── scripts/
├── h100/                          # operator skill for the h100_sglang host
│   └── SKILL.md
└── h100-sglang-diffusion/         # h100 operator skill with diffusion-specific overrides
    └── SKILL.md
```

Run each skill's `ls` to see its exact current file set; this overview is a
high-level map, not a line-level inventory.

Model PR histories are framework-scoped:

```text
model-pr-optimization-history/
├── sglang/
│   ├── model-skill-pr-dossier-quality-scan-2026-04-23.md
│   ├── model-skill-pr-dossier-quality-scan-2026-04-24.md
│   ├── deepseek-v3-r1/
│   ├── qwen3-core/
│   └── ...
└── vllm/
    ├── deepseek-v3-r1/
    ├── qwen3-core/
    └── ...
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

The model optimization handbook series keeps shared production rules at the
`model-optimization/` root, then splits framework-specific model-family skills
by serving framework.

SGLang model optimization skills live under `skills/model-optimization/sglang/`:

- `sglang-deepseek-v3-r1-optimization`
- `sglang-deepseek-v31-optimization`
- `sglang-deepseek-v32-optimization`
- `sglang-deepseek-v4-optimization`
- `sglang-glm-vlm-ocr-optimization`
- `sglang-glm45-optimization`
- `sglang-glm46-glm47-optimization`
- `sglang-glm5-glm51-optimization`
- `sglang-hunyuan3-preview-optimization`
- `sglang-kimi-optimization`
- `sglang-ltx23-hq-optimization`
- `sglang-minimax-optimization`
- `sglang-mixtral-quark-int4fp8-moe-optimization`
- `sglang-moss-vl-optimization`
- `sglang-qwen-image-optimization`
- `sglang-qwen-vlm-omni-asr-optimization`
- `sglang-qwen3-coder-optimization`
- `sglang-qwen3-core-optimization`
- `sglang-qwen3-next-optimization`
- `sglang-qwen35-optimization`
- `sglang-qwen36-optimization`
- `sglang-z-image-turbo-optimization`
- `sglang-ernie45-optimization`
- `sglang-gemma4-optimization`
- `sglang-gpt-oss-optimization`
- `sglang-intern-s1-optimization`
- `sglang-internvl35-optimization`
- `sglang-llama4-optimization`
- `sglang-mimo-v2-flash-optimization`
- `sglang-mistral-small-4-optimization`
- `sglang-nemotron-super-optimization`
- `sglang-step35-optimization`

vLLM model optimization skills live under `skills/model-optimization/vllm/`:

- `vllm-deepseek-v3-r1-optimization`
- `vllm-deepseek-v31-optimization`
- `vllm-deepseek-v32-optimization`
- `vllm-deepseek-v4-optimization`
- `vllm-glm-vlm-ocr-optimization`
- `vllm-glm45-optimization`
- `vllm-glm46-glm47-optimization`
- `vllm-glm5-glm51-optimization`
- `vllm-hunyuan3-preview-optimization`
- `vllm-kimi-optimization`
- `vllm-ltx23-hq-optimization`
- `vllm-minimax-optimization`
- `vllm-mixtral-quark-int4fp8-moe-optimization`
- `vllm-moss-vl-optimization`
- `vllm-qwen-image-optimization`
- `vllm-qwen-vlm-omni-asr-optimization`
- `vllm-qwen3-coder-optimization`
- `vllm-qwen3-core-optimization`
- `vllm-qwen3-next-optimization`
- `vllm-qwen35-optimization`
- `vllm-qwen36-optimization`
- `vllm-z-image-turbo-optimization`
- `vllm-ernie45-optimization`
- `vllm-gemma4-optimization`
- `vllm-gpt-oss-optimization`
- `vllm-intern-s1-optimization`
- `vllm-internvl35-optimization`
- `vllm-llama4-optimization`
- `vllm-mimo-v2-flash-optimization`
- `vllm-mistral-small-4-optimization`
- `vllm-nemotron-super-optimization`
- `vllm-step35-optimization`

The shared `skills/model-optimization/model-pr-diff-dossier/` skill records the
mandatory production standard for model PR histories: read every PR diff and
write motivation, implementation, code excerpt, and validation/risk.

## Model PR Optimization History

The `model-pr-optimization-history/` directory is framework-scoped.

SGLang bilingual model evolution notes live under
`model-pr-optimization-history/sglang/`:

- `deepseek-v3-r1`
- `deepseek-v31`
- `deepseek-v32`
- `deepseek-v4`
- `glm-vlm-ocr`
- `glm45`
- `glm46-glm47`
- `glm5-glm51`
- `hunyuan3-preview`
- `kimi`
- `ltx23-hq`
- `minimax`
- `mixtral-quark-int4fp8-moe`
- `moss-vl`
- `qwen-image`
- `qwen-vlm-omni-asr`
- `qwen3-coder`
- `qwen3-core`
- `qwen3-next`
- `qwen35`
- `qwen36`
- `z-image-turbo`
- `ernie45`
- `gemma4`
- `gpt-oss`
- `intern-s1`
- `internvl35`
- `llama4`
- `mimo-v2-flash`
- `mistral-small-4`
- `nemotron-super`
- `step35`

Cross-family audits sit next to those directories:

- `model-skill-pr-dossier-quality-scan-2026-04-23.md`
- `model-skill-pr-dossier-quality-scan-2026-04-24.md`

vLLM bilingual model evolution notes live under
`model-pr-optimization-history/vllm/`:

- `deepseek-v3-r1`
- `deepseek-v31`
- `deepseek-v32`
- `deepseek-v4`
- `glm-vlm-ocr`
- `glm45`
- `glm46-glm47`
- `glm5-glm51`
- `hunyuan3-preview`
- `kimi`
- `ltx23-hq`
- `minimax`
- `mixtral-quark-int4fp8-moe`
- `moss-vl`
- `qwen-image`
- `qwen-vlm-omni-asr`
- `qwen3-coder`
- `qwen3-core`
- `qwen3-next`
- `qwen35`
- `qwen36`
- `z-image-turbo`
- `ernie45`
- `gemma4`
- `gpt-oss`
- `intern-s1`
- `internvl35`
- `llama4`
- `mimo-v2-flash`
- `mistral-small-4`
- `nemotron-super`
- `step35`

## Install

Copy the desired skill directory into your local skill path:

```bash
cp -r skills/sglang-prod-incident-triage <agent-skill-dir>/sglang-prod-incident-triage
cp -r skills/llm-torch-profiler-analysis <agent-skill-dir>/llm-torch-profiler-analysis
cp -r skills/model-architecture-diagram <agent-skill-dir>/model-architecture-diagram
cp -r skills/model-optimization/sglang/sglang-qwen3-core-optimization <agent-skill-dir>/sglang-qwen3-core-optimization
cp -r skills/model-optimization/vllm/vllm-qwen3-core-optimization <agent-skill-dir>/vllm-qwen3-core-optimization
```
