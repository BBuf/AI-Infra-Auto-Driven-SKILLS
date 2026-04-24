# vLLM Step3.5 / Step3-VL Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Step3.5 / Step3-VL.

- Status: supported on current mainline

## Key Conclusions

- Step3.5 is split between text/MTP and VL processor work.
- NVFP4 and processor behavior are the main axes to track on the vLLM side.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/step3p5.py`
- `vllm/vllm/model_executor/models/step3p5_mtp.py`
- `vllm/vllm/model_executor/models/step3_vl.py`
- `vllm/vllm/model_executor/models/step3_text.py`

## Landed PRs

- [#33755](https://github.com/vllm-project/vllm/pull/33755) `Enable Step3p5ForCausalLM testing`: Stabilized the core Step3.5 text runtime.
- [#34478](https://github.com/vllm-project/vllm/pull/34478) `Add NVFP4 quantization support for Step3.5-Flash`: Opened the practical quantized deployment path.
- [#37579](https://github.com/vllm-project/vllm/pull/37579) `Refactor Step3-VL processor to HF style`: Modernized the Step3-VL processor contract.

## Matching Skill

- `skills/model-optimization/vllm/vllm-step35-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-step35-optimization/references/pr-history.md`
