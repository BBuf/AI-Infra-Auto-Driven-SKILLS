# Qwen3-Coder Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| Tool call JSON malformed | `qwen3_coder_detector.py` | `#16744`, `#13411` |
| Streaming arguments incomplete | `qwen3_coder_detector.py`, streaming protocol | `#21829` |
| Coder-Next load failure | `qwen3_next.py`, loader rules | `#18224`, `#18355`, `#18700`, `#19736` |
| AMD accuracy/perf regression | AMD registered tests and AITER backend | `#18355`, `#18608`, `#19736` |
| MoE perf regression | fused MoE configs and Qwen3-Coder-Next dispatch | `#17965`, `#18195` |

## Investigation Order

1. Reproduce parser-only behavior with synthetic chunks.
2. Reproduce model output without parser.
3. Enable parser and streaming.
4. Enable quantized checkpoint.
5. Enable Coder-Next hybrid/MTP features if relevant.

## Validation Checklist

- Complete tool-call JSON.
- Incremental streaming tool-call arguments.
- Empty tool list and multi-tool schemas.
- Quantized Coder-Next loading.
- AMD and NPU smoke tests if those backends are touched.

## Change Rules

- Parser changes must not assume one schema shape.
- Model runtime fixes for Coder-Next should reference Qwen3-Next history.
- Cookbook updates should include the parser flag explicitly.
