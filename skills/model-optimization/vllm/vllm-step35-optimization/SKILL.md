---
name: vllm-step35-optimization
description: PR-backed optimization manual for Step3.5 / Step3-VL in vLLM. Use when Codex needs to audit, debug, extend, or document Step3.5-Flash and Step3-VL serving, NVFP4, tool/reasoning parser, and HF-style processor evolution.
---

# vLLM Step3.5 / Step3-VL Optimization

## Overview

This skill covers Step3.5-Flash and Step3-VL serving, NVFP4, tool/reasoning parser, and HF-style processor evolution.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/step35/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/step3p5.py`
- `vllm/vllm/model_executor/models/step3p5_mtp.py`
- `vllm/vllm/model_executor/models/step3_vl.py`
- `vllm/vllm/model_executor/models/step3_text.py`

## Current Main Summary

- Step3.5 is split between text/MTP and VL processor work.
- NVFP4 and processor behavior are the main axes to track on the vLLM side.

## Key Landed PRs

- [#33755](https://github.com/vllm-project/vllm/pull/33755) `Enable Step3p5ForCausalLM testing`: Stabilized the core Step3.5 text runtime.
- [#34478](https://github.com/vllm-project/vllm/pull/34478) `Add NVFP4 quantization support for Step3.5-Flash`: Opened the practical quantized deployment path.
- [#37579](https://github.com/vllm-project/vllm/pull/37579) `Refactor Step3-VL processor to HF style`: Modernized the Step3-VL processor contract.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
