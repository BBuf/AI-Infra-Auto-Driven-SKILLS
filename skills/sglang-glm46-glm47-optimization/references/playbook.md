# GLM-4.6/4.7 Playbook

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| Shared expert fusion wrong | `glm4_moe.py`, MoE fusion code | `#13873`, `#21403` |
| MoE overlap regression | dual-stream MoE dispatch | `#13786`, `#22801` |
| GLM-4.7 tool parser wrong | `glm47_moe_detector.py` | `#15333`, `#15753`, `#15754`, `#15520` |
| Flash/lite load failure | `glm4_moe_lite.py`, config import | `#17247`, `#21851`, `#22509`, `#22720`, `#22823` |
| MTP/NVFP4 issue | `glm4_moe_nextn.py`, draft quant config | `#17166`, `#22315` |

## Investigation Order

1. BF16 base model.
2. Parser/chat template.
3. Lite/Flash import format.
4. Quantized checkpoint.
5. MTP/NextN.
6. MoE fusion/overlap.
7. AMD/NPU backend.

## Production Optimization Checklist

- Shared-expert fusion and QKNorm fusion should each have a logits check and a profile before being combined.
- Suffix decoding is a separate serving feature; validate cache correctness and throughput independently from MoE fusion.
- EAGLE/speculative decoding should be enabled only after base GLM-4.7/Flash loading and parser behavior are stable.
- Parser fixes must include `glm47` streaming output arguments and `glm45` compatibility where shared detector code is touched.

## Validation Checklist

- GLM-4.6 registered 8-GPU test.
- GLM-4.7 tool parser streaming.
- GLM-4.7-Flash startup and quantized loading.
- MTP target/draft path.
- AMD/NPU perf and accuracy.

## Change Rules

- Keep parser fixes covered by streaming and complex schema cases.
- Flash/lite import fixes need one compressed/quantized checkpoint.
- Shared-expert fusion changes require logits comparison and perf numbers.
