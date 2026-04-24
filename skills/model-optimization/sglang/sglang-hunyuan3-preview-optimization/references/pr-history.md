# Hunyuan 3 Preview PR History

Evidence sweep:

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
- Manual diff review date: `2026-04-24`
- Searched paths: Hunyuan3 Preview cookbook, command generator, docs navigation, SGLang Hunyuan parser/model surfaces.
- Searched PR terms: `Hunyuan 3`, `Hy3-preview`, `hunyuan3-preview`, `hunyuan`.

## Runtime and Docs Surfaces

- `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`
- `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`
- `docs_new/docs.json`
- existing parser/runtime surface: `python/sglang/srt/models/hunyuan.py` and Hunyuan reasoning/tool parsers

## Diff-Reviewed PR Cards

### PR #23532 - Add Hunyuan 3 Preview cookbook

- Link: https://github.com/sgl-project/sglang/pull/23532
- State: merged at `2026-04-23T09:44:47Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `1309` lines, `3` files; current-main docs rechecked at `bca3dd958`.
- Motivation: Hunyuan 3 Preview needed launch documentation that records BF16 memory sizing, parser choices, MTP speculative decoding, and Blackwell attention backend constraints.
- Key implementation: adds `Hunyuan3-Preview.mdx`, registers it in docs navigation, and adds `hunyuan3-preview-deployment.jsx`. The generator maps H200/B200 to TP=8, B300/GB300 to TP=4, includes reasoning and tool parser toggles, adds MTP/EAGLE flags when enabled, always includes `--trust-remote-code`, and appends `--attention-backend trtllm_mha` for Blackwell hardware.
- Key code excerpts:

```jsx
const modelConfigs = {
  h200: { tp: 8, mem: 0.9 },
  b200: { tp: 8, mem: 0.9 },
  b300: { tp: 4, mem: 0.9 },
  gb300: { tp: 4, mem: 0.9 },
};
```

```jsx
if (enableSpec) cmd += "SGLANG_ENABLE_SPEC_V2=1 ";
if (values.reasoning === "enabled") cmd += " \\\n  --reasoning-parser hunyuan";
if (values.toolcall === "enabled") cmd += " \\\n  --tool-call-parser hunyuan";
if (isBlackwell) cmd += " \\\n  --attention-backend trtllm_mha";
```

```bash
SGLANG_ENABLE_SPEC_V2=1 sglang serve \
  --model-path tencent/Hy3-preview \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser hunyuan \
  --tool-call-parser hunyuan \
  --trust-remote-code
```

- Reviewed files: `docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx`, `docs_new/src/snippets/autoregressive/hunyuan3-preview-deployment.jsx`, `docs_new/docs.json`.
- Validation implications: command generation is the regression surface. Check TP sizing, Blackwell backend, parser flags, MTP env/flags, and trust-remote-code. The PR is docs-only; no new Hunyuan model implementation file landed in this diff.

## Validation Notes

- The cookbook still carries a license TODO. Do not treat license text as verified runtime evidence.
- If runtime code support changes later, add separate cards for parser/model files instead of extending this docs-only PR.
