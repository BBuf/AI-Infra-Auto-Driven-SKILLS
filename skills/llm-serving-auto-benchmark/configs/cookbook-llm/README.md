# Cookbook LLM Configs

These configs define a framework-neutral LLM serving cookbook model set and translate each model into a four-framework run plan for SGLang, vLLM, TensorRT-LLM, and TokenSpeed.

Scope:
- SGLang can preserve source-recipe `base_flags` and `search_space` where applicable; if a sequence limit is smaller than the default synthetic scenario, the config raises that limit so the shipped workload can run.
- vLLM uses framework-native `vllm serve` flags. The translation keeps the same model, tokenizer, dataset shape, GPU count, and high-impact batching/prefix-cache knobs; it does not copy SGLang-only parser or scheduler flags.
- TensorRT-LLM uses `trtllm-serve serve` with `backend: pytorch` fixed in `base_server_flags`. Backend choice is never searched.
- TokenSpeed uses `tokenspeed serve <model>` with framework-native TP, memory, max sequence, batching, chunked-prefill, and prefix-cache knobs. Treat these sections as first-pass candidates and validate them against the target `tokenspeed serve --help`.
- The two default random scenarios remain aligned pairs: `chat` uses `1000 -> 1000`, and `summarization` uses `8000 -> 1000`.
- A framework is enabled only when the recorded upstream head exposes the exact model/checkpoint and required launch flags. Otherwise its section is retained as `enabled: false` with `support_status: not_verified_at_recorded_head`; that status is not a claim that the framework can never support the model.

Current narrow additions:

- `MiniMaxAI/MiniMax-M3-MXFP8`: the enabled SGLang recipe follows the verified single-node 8×B200 launch (`tp=8`, FA4 sparse attention, DeepGEMM MoE, 0.65 static-memory fraction). The 2026-08-23 source head is recorded below; the last GPU smoke for this recipe remains the earlier B200 run. vLLM lists the exact checkpoint, so its section is an enabled generic translation; TensorRT-LLM and TokenSpeed stay disabled because their recorded recipes use a different hardware or checkpoint contract.
- `Qwen/Qwen3.6-35B-A3B-FP8`: the enabled SGLang recipe follows its current B200 single-GPU cookbook entry at the same recorded head. The other recorded heads do not expose this exact FP8 checkpoint contract, so those sections stay disabled.
- `Qwen/Qwen3.8-27B-FP8`: the enabled SGLang section copies the public H200 single-GPU cookbook cell (`flashinfer`, FP8 KV, 0.85 static-memory fraction, 32768 prefill chunk). vLLM lists `Qwen/Qwen3.8-27B` through the existing `qwen3_5` loader, TensorRT-LLM has same-day Qwen3.8 wave-2 merges, and TokenSpeed still has open Hopper cookbook `#1111`; those sections stay disabled because this refresh did not GPU-verify a matching four-framework contract.

Inkling, Unlimited OCR, Kimi K3, and DeepSeek V4 are intentionally excluded
from the cross-framework cookbook in this refresh because their current
endpoint, checkpoint, or four-framework comparison contracts are not uniform
enough for a defensible recipe. Kimi K3 is no longer an unmerged Day-0
branch: SGLang `v0.5.17` and vLLM `v0.27.0` both shipped it, but the 2.8T
hardware floor and TokenSpeed/TensorRT-LLM sidecar contracts still fail the
four-framework cookbook bar. Use `sglang-model-day0-support` and the official
SGLang/vLLM cookbooks instead of inventing a generic 8-GPU recipe.

Before a real run, capture the target framework `--help` output and validate the configs:

```bash
python skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py   skills/llm-serving-auto-benchmark/configs/cookbook-llm
```

With captured help files, add `--help-dir <artifact-help-dir>` to check the concrete flag names against that environment. This check only loads configs and renders candidate commands; it does not launch model servers.
