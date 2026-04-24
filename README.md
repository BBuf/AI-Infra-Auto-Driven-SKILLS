# AI-Infra-Auto-Driven-SKILLS

Agent skills for SGLang/vLLM/TensorRT-LLM development, profiling, and production incident triage.

## Structure

```
skills/
├── model-optimization/
│   ├── model-pr-diff-dossier/
│   ├── sglang/
│   └── vllm/
├── llm-serving-auto-benchmark/
├── llm-torch-profiler-analysis/
├── sglang-prod-incident-triage/
├── h100/
└── h100-sglang-diffusion/
```

Model histories are framework-scoped under `model-pr-optimization-history/sglang/` and `model-pr-optimization-history/vllm/`.

## SGLang Model Skills

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
## vLLM Model Skills

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
## Install

```bash
cp -r skills/model-optimization/sglang/sglang-qwen3-core-optimization <agent-skill-dir>/sglang-qwen3-core-optimization
cp -r skills/model-optimization/vllm/vllm-qwen3-core-optimization <agent-skill-dir>/vllm-qwen3-core-optimization
```
