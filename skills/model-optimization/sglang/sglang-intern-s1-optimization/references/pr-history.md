# SGLang Intern-S1 PR History

Evidence snapshot:

- SGLang mainline checked around `c122d343adb969cd9bbd1af2ca86727a11be3845`
- sgl-cookbook checked around `e88b0fd8ac5b1caa6eb42766035029220053369b`
- Scope: Intern-S1 language and video-aware serving, processor integration, and tool/reasoning parser behavior.

## Landed PRs

### PR #9381 - InternS1 image token updates in InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/9381
- Why it mattered: Aligned the shared processor with Intern-S1 image semantics.
- Runtime path: sglang/python/sglang/srt/models/interns1.py, sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #12367 - Fix Intern-S1 accuracy and `/generate` input_ids support

- Link: https://github.com/sgl-project/sglang/pull/12367
- Why it mattered: Closed early correctness gaps.
- Runtime path: sglang/python/sglang/srt/models/interns1.py, sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #14866 - Add tool calling and reasoning parser support for Intern-S1

- Link: https://github.com/sgl-project/sglang/pull/14866
- Why it mattered: Added parser support that cookbook usage depends on.
- Runtime path: sglang/python/sglang/srt/models/interns1.py, sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #17040 - Support InternS1 text_config in InternVL processor

- Link: https://github.com/sgl-project/sglang/pull/17040
- Why it mattered: Improved sub-config compatibility in shared processors.
- Runtime path: sglang/python/sglang/srt/models/interns1.py, sglang/python/sglang/srt/models/internvl.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.
