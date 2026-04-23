# SGLang Qwen3 Core 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）、sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 Qwen3 dense、Qwen3 MoE、Qwen3-30B、Qwen3-235B-A22B 以及共享的 Qwen3 parser/quant/LoRA/平台优化路径。

结论：Qwen3 core 是后续 Qwen3.5、Qwen3-Next、Qwen3.6 的基础层。当前主线已经具备 dense/MoE、RoPE config fallback、fused QK-norm/RoPE、FP8/NVFP4 ModelOpt、EPLB/FP4、FlashInfer TRTLLM-GEN-MoE、EAGLE3 DP attention、LoRA、PP/NPU/AMD/Blackwell 等能力。后续重点是滑动窗口注意力、XPU/NPU 量化、W4AFP8、tool-call/reasoning parser 边界和 Qwen MoE tied embeddings。

## 代码面

- `python/sglang/srt/models/qwen3.py`
- `python/sglang/srt/models/qwen3_moe.py`
- `python/sglang/srt/function_call/qwen25_detector.py`
- `test/registered/models/test_qwen_models.py`
- `test/registered/4-gpu-models/test_qwen3_30b.py`
- `test/registered/stress/test_stress_qwen3_235b.py`
- `test/srt/models/test_lora_qwen3.py`
- `docs/basic_usage/qwen3.md`
- `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`

## 已合入主线 PR

- `#7312`：Qwen3 LoRA hidden dim。
- `#7912`：Qwen FP8/NVFP4 ModelOpt 量化。
- `#9973`：Qwen3-MoE FlashInfer fused all-reduce。
- `#10574`、`#12078`、`#15203`、`#15390`、`#16115`、`#19532`：Ascend/NPU Qwen3 优化、GPTQ、PP、DP LM-head、spec 条件。
- `#10749`：Qwen3-MoE fused write KV buffer into RoPE。
- `#12002`：Qwen3 MoE EAGLE3 DP attention。
- `#13489`：FlashInfer TRTLLM-GEN-MoE 支持 Qwen3。
- `#13715`：EPLB + FP4。
- `#13998`、`#15835`、`#19059`、`#21654`：fused QK-norm/RoPE 及 JIT/sincosf 去重。
- `#14093`：TRTLLM MHA FP8 KV cache write fusion。
- `#15223`：Qwen3 PP 加载。
- `#15890`：PP tied embeddings。
- `#17535`：Qwen3 embedding weight rename。
- `#17784`：transformers 5.3.0 兼容。
- `#18189`：Qwen3-235B NVFP4 ModelOpt launch。
- `#18233`：Qwen3 MoE context parallel。
- `#20931`：Qwen3 RoPE 参数兼容。
- `#21195`：Qwen3 测试启用。
- `#22003`：MoE/attention-CP 拓扑放宽。
- `#22358`：DFLASH 支持。
- `#22739`：恢复 Qwen3 RoPE config fallback。

## Open PR 雷达

- `#9147`：Qwen3-MoE W4AFP8。
- `#20127`：Qwen MoE/Qwen3-Next tied embeddings。
- `#20474`：Intel XPU Qwen3 layernorm/MRoPE。
- `#22529`：Qwen3 sliding window attention。
- `#22674`：NPU Qwen3.5-MoE/Qwen3-Next 量化，涉及共享 Qwen 量化逻辑。
- `#22837`：Qwen3 reasoning detector 吞掉 tool_call。

## sgl-cookbook / 文档

- `sgl-cookbook#74`：Qwen3 AMD 支持并修 tool calling 文档。
- `sgl-cookbook#245`：Qwen 文档刷新。

## 下一步优化建议

1. 优先补齐 Qwen3 parser 的 streaming reasoning + tool-call 联合测试。
2. 把 Qwen3 MoE W4AFP8、NPU/XPU 量化、sliding window attention 分成独立验证矩阵。
3. 任何 Qwen3 core 改动都要回归 Qwen3.5/Qwen3-Next，因为二者复用大量 Qwen3 基础逻辑。

## 2026-04-23 三轮复查补充

详细账本见 `model-pr-optimization-history/qwen-glm-three-pass-audit-2026-04-23.md`。本轮补入早期 Qwen3/Qwen3-MoE 支持与 MoE runtime 演进 PR：`#4693`、`#5917`、`#6120`、`#6121`、`#6250`、`#6546`、`#6533`、`#6598`、`#6652`、`#6709`、`#6818`、`#6820`、`#6964`、`#6990`、`#7222`、`#7634`、`#7681`、`#7723`、`#7740`、`#7745`、`#7966`、`#8036`、`#8280`、`#8421`、`#8448`、`#8450`、`#8658`、`#8751`、`#8987`、`#9101`、`#9338`、`#10975`。

新增 open radar：`#20520`、`#21412`、`#21770`、`#22429`、`#22446`、`#22450`、`#22687`、`#23372`、`#23397`、`#23434`。验证入口补充 `test/manual/test_qwen3_235b.py`、`test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py` 和 NPU Qwen3 tests。
