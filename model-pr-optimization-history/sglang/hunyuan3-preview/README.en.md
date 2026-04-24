# SGLang Hunyuan 3 Preview Support and Optimization Timeline

Scope: Tencent Hunyuan 3 Preview BF16 cookbook support, hardware TP sizing, reasoning/tool parsers, MTP/EAGLE flags, Blackwell attention backend, and trust-remote-code launch guidance.

Evidence snapshot: SGLang `origin/main` `bca3dd958` (`2026-04-24`). Full dossier: `skills/model-optimization/sglang/sglang-hunyuan3-preview-optimization/references/pr-history.md`.

## Diff-Reviewed PR

#23532 added the Hunyuan 3 Preview cookbook and command generator. Full diff reviewed: `1309` lines, `3` files. The generator maps H200/B200 to TP=8, B300/GB300 to TP=4, adds `hunyuan` parser toggles, MTP/EAGLE flags, `--trust-remote-code`, and Blackwell `--attention-backend trtllm_mha`.

Current caveat: this is docs/recipe support; no new Hunyuan3-specific SGLang model implementation file landed in the PR.
