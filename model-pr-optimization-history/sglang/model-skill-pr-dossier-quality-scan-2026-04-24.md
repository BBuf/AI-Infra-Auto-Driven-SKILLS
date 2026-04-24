# Model Skill PR Dossier Quality Scan - 2026-04-24

Scope: incremental audit after syncing SGLang to `origin/main` `bca3dd958` (`2026-04-24`) from the previous evidence snapshot `b3e6cf60a` (`2026-04-22`).

Method: inspected the SGLang `b3e6cf60a..bca3dd958` commit range, opened relevant public PR diffs with `gh pr diff --patch`, cross-checked current-main source in a detached worktree, and patched missing or stale model-support/model-optimization coverage.

## Added or Updated Coverage

- `sglang-glm5-glm51-optimization`: added #23060 dynamic chunking profiling crash fix and #23540 GLM-5.1 MI300X/MI325X generator split.
- `sglang-qwen-image-optimization`: updated #22953 from open to merged/current-main behavior.
- `sglang-deepseek-v3-r1-optimization` and `sglang-deepseek-v32-optimization`: updated #22774 MUSA backend from open radar to merged status.
- `sglang-deepseek-v4-optimization`: new docs/recipe skill and bilingual history for #23605, #23617, #23628, #23622, #23634.
- `sglang-moss-vl-optimization`: new runtime skill and bilingual history for #23454.
- `sglang-ltx23-hq-optimization`: new diffusion skill and bilingual history for #23366 and #23624.
- `sglang-hunyuan3-preview-optimization`: new docs/recipe skill and bilingual history for #23532.
- `sglang-z-image-turbo-optimization`: new diffusion skill and bilingual history for #17822, #19733, and #23455 after confirming the Z-Image family had no existing dossier.
- `sglang-mixtral-quark-int4fp8-moe-optimization`: new AMD quantization skill and bilingual history for #7392, #17116, and #23455 after confirming the Mixtral/quark INT4-FP8 MoE path had no existing dossier.

## Reviewed But Not Promoted To Model-Family Cards

These PRs are cross-model infrastructure or CI updates from the same SGLang range. They should be considered if a future model-family dossier touches their exact runtime surface:

- #22931: adds JIT `rmsnorm_hf` to match HuggingFace RMSNorm rounding semantics for transformers-backend accuracy.
- #23414 and #23542: detect actual FP8 expert weights from safetensors headers and support HF repo IDs.
- #23319: AMD AITER bpreshuffle FP8 blockscale GEMM path.
- #23545 and #23585: MoE no-combine/router-weight and dispatcher-owned `expert_mask_gpu` fixes.
- #23125: MXFP8 TrtllmGenMoe CI model/backend correction.
- #23455 also restored Mixtral and Z-Image direct-file tests; those model-specific pieces were promoted into the new dossiers above.

## Follow-Up Rule

When any legacy DeepSeek/Kimi/MiniMax dossier is rewritten into strict card format, include the cross-model infrastructure PRs above if the family actually depends on that path. Do not append title-only rows.
