---
name: vllm-intern-s1-optimization
description: PR-backed optimization manual for Intern-S1 in vLLM. Use when an engineer needs to audit, debug, extend, or document Intern-S1 language and video-aware serving, processor integration, and tool/reasoning parser behavior.
---

# vLLM Intern-S1 Optimization

## Overview

This skill covers Intern-S1 language and video-aware serving, processor integration, and tool/reasoning parser behavior.

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/vllm/intern-s1/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `vllm/vllm/model_executor/models/interns1.py`
- `vllm/vllm/model_executor/models/interns1_pro.py`

## Current Main Summary

- Intern-S1 leans heavily on shared InternVL processor code in vLLM.
- Most regressions come from processor compatibility and video-aware serving rather than the text stack alone.

## Key Landed PRs

- [#21628](https://github.com/vllm-project/vllm/pull/21628) `Support Intern-S1`: Initial Intern-S1 support in vLLM.
- [#21671](https://github.com/vllm-project/vllm/pull/21671) `Add video support for Intern-S1`: Extended the family beyond static images.
- [#22417](https://github.com/vllm-project/vllm/pull/22417) `Fix wrong method name in Intern-S1 image processor`: Patched a processor bug after bring-up.
- [#33636](https://github.com/vllm-project/vllm/pull/33636) `Intern-S1-Pro`: Added the Pro generation / alias path.

## Validation Lanes

- Startup on the current vLLM mainline checkpoint.
- Re-run the parser, quantization, MTP, or multimodal lane that matches this family.
- Re-check tests after touching loader or processor code.
