---
name: sglang-intern-s1-optimization
description: PR-backed optimization manual for Intern-S1 in SGLang. Use when Codex needs to audit, debug, extend, or document Intern-S1 language and video-aware serving, processor integration, and tool/reasoning parser behavior.
---

# SGLang Intern-S1 Optimization

## Overview

This skill covers Intern-S1 language and video-aware serving, processor integration, and tool/reasoning parser behavior.

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Canonical PR notes: `references/pr-history.md`
- History mirrors: `model-pr-optimization-history/sglang/intern-s1/README.zh.md` and `README.en.md`

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Runtime Surfaces

- `sglang/python/sglang/srt/models/interns1.py`
- `sglang/python/sglang/srt/models/internvl.py`

## Current Main Summary

- Intern-S1 leans heavily on shared InternVL processor code in SGLang.
- Most regressions come from processor compatibility, parser behavior, and video-aware serving rather than the text stack alone.

## Key Landed PRs

- [#9381](https://github.com/sgl-project/sglang/pull/9381) `InternS1 image token updates in InternVL processor`: Aligned the shared processor with Intern-S1 image semantics.
- [#12367](https://github.com/sgl-project/sglang/pull/12367) `Fix Intern-S1 accuracy and `/generate` input_ids support`: Closed early correctness gaps.
- [#14866](https://github.com/sgl-project/sglang/pull/14866) `Add tool calling and reasoning parser support for Intern-S1`: Added parser support that cookbook usage depends on.
- [#17040](https://github.com/sgl-project/sglang/pull/17040) `Support InternS1 text_config in InternVL processor`: Improved sub-config compatibility in shared processors.

## Validation Lanes

- Startup on the cookbook checkpoint or example route.
- Re-run the parser, quantization, or multimodal lane that matches this family.
- Re-check registered or manual tests after touching loader or processor code.
