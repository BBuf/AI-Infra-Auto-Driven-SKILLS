# Qwen3 Core Playbook

## Fast Triage

| Symptom | First files to read | Likely PR history |
| --- | --- | --- |
| Wrong output after RoPE/QK-norm fusion | `qwen3.py`, `qwen3_moe.py`, fused QK norm kernels | `#13998`, `#15835`, `#19059`, `#21654` |
| FP8/NVFP4 checkpoint load failure | `qwen3_moe.py`, quantization loaders, ModelOpt logic | `#7912`, `#13715`, `#18189` |
| MoE throughput regression | fused MoE configs, `qwen3_moe.py`, DeepEP/AITER dispatch | `#9973`, `#13489`, `#14093`, `#18233` |
| LoRA mismatch | `qwen3.py`, `qwen3_moe.py`, LoRA test files | `#7312` and later Qwen LoRA tests |
| Parser loses tool calls | `qwen25_detector.py`, OpenAI protocol streaming logic | `#22837` |
| NPU or XPU regression | platform backend files plus registered NPU/XPU tests | `#10574`, `#12078`, `#15203`, `#20474` |

## Investigation Order

1. Identify whether the checkpoint maps to `Qwen3ForCausalLM` or `Qwen3MoeForCausalLM`.
2. Run BF16 correctness before enabling quantization, EAGLE3, or CP.
3. Compare single-GPU and target TP/EP output with deterministic prompts.
4. Inspect RoPE config fallback and tied embedding handling before changing tensor-parallel slicing.
5. If a fused kernel is involved, disable it and prove the fallback is correct.
6. If a parser issue is involved, run streaming tool calls with reasoning enabled.

## Validation Commands

Use the local repo's current test registration; these paths are the known Qwen3 anchors:

```bash
python -m pytest test/registered/models/test_qwen_models.py -k qwen
python -m pytest test/registered/4-gpu-models/test_qwen3_30b.py
python -m pytest test/srt/models/test_lora_qwen3.py
```

For performance work, capture both prefill and decode:

```bash
python -m sglang.bench_serving --backend sglang --model <checkpoint> --dataset-name random --num-prompts 512
```

## Change Rules

- Keep dense Qwen3, Qwen3 MoE, and Qwen3-Next fixes separate unless the shared base code is the actual source.
- Do not add a new backend default without documenting the hardware and KV/quantization assumptions.
- For quantized checkpoints, preserve BF16 fallback tests and weight-name mapping tests.
- For parser work, validate both accumulated final JSON and incremental streaming chunks.
