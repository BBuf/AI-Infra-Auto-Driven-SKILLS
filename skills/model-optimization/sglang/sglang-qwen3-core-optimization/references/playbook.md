# Qwen3 Core Playbook

Use this playbook after reading the canonical PR cards in [pr-history.md](pr-history.md). The cards are not optional background; they are the evidence base for changing Qwen3 Core docs or code.

## Diff-First Rule

For any new PR added to Qwen3 Core history:

1. Open the full PR diff or merge commit.
2. Read the changed source files, docs, and tests.
3. Write a card with motivation, implementation, key code excerpt, and validation/risk.
4. Mark open PRs as radar only.
5. Do not use scripts or generated summaries to fill card content.

## Fast Triage

| Symptom | First files to read | PR cards to read |
| --- | --- | --- |
| Qwen3 checkpoint will not load | `qwen3.py`, `qwen3_moe.py`, loader/quant files | `#4693`, `#6990`, `#17535`, `#17784`, `#20931`, `#22739` |
| Wrong output under PP or tied embeddings | `qwen3.py`, `qwen3_moe.py`, PP helpers | `#6250`, `#6546`, `#15223`, `#15890`, open `#20127` |
| MoE route/expert load issue | `qwen3_moe.py`, `layers/moe/`, `srt/eplb/` | `#5917`, `#6120`, `#6533`, `#6709`, `#6818`, `#6964`, `#8448`, `#13715` |
| DeepEP or A2A backend mismatch | `qwen3_moe.py`, MoE backend selector, server args | `#7222`, `#8421`, `#8658`, `#12078` |
| DP attention throughput regression | `LayerCommunicator`, `qwen3.py`, `qwen3_moe.py` | `#6121`, `#6598`, `#6652`, `#7681`, `#8280`, `#9101` |
| CP or long-context issue | attention CP utils, FlashAttention backend, `qwen3_moe.py` | `#18233`, `#21195`, `#22003` |
| FP8/NVFP4/FP4 launch failure | ModelOpt/FP4 loaders, packed mappings, MoE weight filters | `#7912`, `#8036`, `#8450`, `#13489`, `#13715`, `#18189`, open `#9147` |
| Wrong result after QK-norm/RoPE fusion | Qwen3 attention, JIT/AOT qknorm rope kernels | `#7740`, `#10749`, `#13998`, `#15835`, `#19059`, `#21654` |
| NPU launch or perf regression | NPU model paths, NPU kernels, registered NPU tests | `#10574`, `#12078`, `#15203`, `#15390`, `#16115`, `#19532`, open `#20520`, open `#23372` |
| LoRA adapter mismatch | LoRA utils, Qwen3 tests | `#7312`, `#8987` |
| EAGLE3 or DFLASH aux capture mismatch | layer capture hooks, logits processor, EAGLE worker | `#7745`, `#12002`, `#22358` |
| Parser loses tool calls | `qwen25_detector.py`, streaming parser tests | open `#22837` |
| Sliding-window Qwen3 model unsupported | Qwen3 config/layer init, `RadixAttention` | open `#22529` |

## Investigation Order

1. Confirm architecture, checkpoint, and exact server flags.
2. Read the relevant cards in `pr-history.md`.
3. Re-open the current PR diff or local SGLang merge commit if the card is older than the code you are editing.
4. Reproduce BF16 correctness before quantization or kernel fusion.
5. Disable fused paths to prove fallback behavior before debugging fused kernels.
6. Separate tensor shape, weight-name mapping, routing/top-k, and communication bugs.
7. Update the card if the investigation finds a new invariant.

## Validation Anchors

Known Qwen3 test entry points:

```bash
python -m pytest test/registered/models/test_qwen_models.py -k qwen
python -m pytest test/registered/4-gpu-models/test_qwen3_30b.py
python -m pytest test/srt/models/test_lora_qwen3.py
python -m pytest test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py
```

Useful manual/perf entry points:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --model <checkpoint> \
  --dataset-name random \
  --num-prompts 512
```

For Qwen3 MoE performance, always record:

- checkpoint and quantization,
- TP/DP/EP/PP/attention-CP/MoE-DP sizes,
- attention backend,
- MoE runner and A2A backend,
- whether DP attention, EAGLE3, TBO, EPLB, and fused kernels are enabled,
- input/output lengths and prompt count.

## Change Rules

- Keep dense Qwen3, Qwen3 MoE, and Qwen3-Next fixes separate unless a shared helper is the source.
- Do not add a backend default without naming hardware, quantization, and communication assumptions.
- For quantized checkpoints, preserve BF16 fallback tests and weight-name mapping tests.
- For parser work, validate accumulated final JSON and incremental streaming chunks.
- For docs-only PRs, quote the exact command/config diff and tie it to a validation lane.
- For open PRs, write "open optimization candidate" and avoid implying support has landed.
