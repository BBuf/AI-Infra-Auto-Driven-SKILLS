# TensorRT-LLM Qwen3.5 模型 PR 优化历史

## 2026-08-23 源码 head 刷新

已复核 TensorRT-LLM 上游 main：
`NVIDIA/TensorRT-LLM@da38c1d2e0dffd073b7dfb6d69e15ee7b45d84a9`。
已读完上一记录 head `1b4ffc0291d75a21ad20118e8f44de6e3831f786`
之后的 7-commit 完整增量；其中没有新的 Qwen3.5 专属实现提交，最新 PR #16677
仅涉及 VisualGen/Wan 的 Attention2D + TP，以下 4 个此前提升的 runtime PR
继续作为当前模型证据。

结果：Qwen3.5 的 MoE 与 Dense VLM 路径先后落地，随后加入注意力预处理融合以及 AllReduce + Gemma RMSNorm 融合。只改 waiver 的测试 PR 不混入 runtime 证据集。

| 合并日期 | PR | Runtime 信号 |
| --- | --- | --- |
| 2026-07-04 | [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599) | Qwen3.5 MoE VLM + MTP |
| 2026-07-07 | [#15249](https://github.com/NVIDIA/TensorRT-LLM/pull/15249) | Qwen3.5 Dense VLM |
| 2026-07-21 | [#16469](https://github.com/NVIDIA/TensorRT-LLM/pull/16469) | 融合 QK norm + RoPE + gate |
| 2026-07-24 | [#15194](https://github.com/NVIDIA/TensorRT-LLM/pull/15194) | 融合 AllReduce + Gemma RMSNorm |

## 2026-06-27 PR 补漏复核

本文的逐 PR diff 审计卡基于 TensorRT-LLM 上游
`HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980` 生成；根目录 TensorRT-LLM
history index 已跟踪 2026-06-27 runtime refresh
`aaffa2f9fef3025e0f698d978385a73460344e0b`。本文覆盖 Qwen3.5 相关 merged
PR，并采用 SGLang 风格的模型实现覆盖、PR 时间线和逐 PR diff 审计卡。

本轮筛选规则：标题/文件命中 `Qwen3.5`、`Qwen3_5`、`qwen3_5`、`AutoDeploy`、`NVFP4`、`FP8`、`DFlash`、`reasoning_parser`、`EPLB`、`MoE backend`、`model_registry` 的 merged PR；过滤纯重排和不触碰模型/loader/test lane 的基础设施 PR。

## 模型实现文件覆盖

| 文件 | 关联 PR |
| --- | --- |
| `tensorrt_llm/_torch/models/modeling_qwen3_5.py` | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) |
| `tensorrt_llm/_torch/models/checkpoints/hf/qwen3_5_weight_mapper.py` | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302), [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090), [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716), [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) |
| `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114), [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667), [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) |
| `examples/auto_deploy/model_registry/configs/qwen3.5_moe_*.yaml` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114), [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667), [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) |
| `examples/auto_deploy/model_registry/models.yaml` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114), [#15001](https://github.com/NVIDIA/TensorRT-LLM/pull/15001) |
| `tensorrt_llm/_torch/auto_deploy/transform/library/mrope_delta_cache.py` | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) |
| `tensorrt_llm/_torch/models/modeling_speculative.py` / `speculative/dflash.py` | [#13782](https://github.com/NVIDIA/TensorRT-LLM/pull/13782), [#13996](https://github.com/NVIDIA/TensorRT-LLM/pull/13996) |
| `tensorrt_llm/llmapi/reasoning_parser.py` | [#14659](https://github.com/NVIDIA/TensorRT-LLM/pull/14659) |
| `tensorrt_llm/_torch/modules/fused_moe/moe_load_balancer.py` | [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) |
| `tests/integration/defs/accuracy/test_llm_api_pytorch.py` | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302), [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090), [#15081](https://github.com/NVIDIA/TensorRT-LLM/pull/15081), [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) |

## PR 覆盖总览

- 本轮审计 PR 数: 14
- 文件反查命令: `git log --name-only -- <model-files>`
- diff 来源: `gh pr diff` / GitHub Pull Request files API patch，本地缓存 `/tmp/model_pr_diffs/tensorrt_llm/pr*.diff`
- 已读 patch 行数: 12,514
- TensorRT-LLM Qwen3.5 关键形态: AutoDeploy cookbook/registry、mRoPE/3D position、NVFP4/FP8 weight mapper、dense/MoE wrappers、DFlash speculative decoding、reasoning parser、CUTLASS/DeepGEMM backend选择、EPLB。

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2026-02-26 | [#11728](https://github.com/NVIDIA/TensorRT-LLM/pull/11728) | merged | Added Qwen3.5 Cookbook | AutoDeploy cookbook notebook |
| 2026-03-24 | [#12302](https://github.com/NVIDIA/TensorRT-LLM/pull/12302) | merged | Add Qwen 3.5 supporting (NVFP4) | model wrapper, weight mapper, tests |
| 2026-03-25 | [#12114](https://github.com/NVIDIA/TensorRT-LLM/pull/12114) | merged | Qwen 3.5 fix 3d position ID handling | AutoDeploy Qwen3.5 MoE, mRoPE cache, registry configs |
| 2026-04-30 | [#13090](https://github.com/NVIDIA/TensorRT-LLM/pull/13090) | merged | Qwen3.5 dense weight loading | weight mapper, dense tests |
| 2026-05-04 | [#13716](https://github.com/NVIDIA/TensorRT-LLM/pull/13716) | merged | Fix Qwen3.5 NVFP4 weight loading by preserving weight_scales | HF mapper |
| 2026-05-12 | [#13782](https://github.com/NVIDIA/TensorRT-LLM/pull/13782) | merged | Qwen3.5 DFlash | speculative/DFlash runtime |
| 2026-05-16 | [#13996](https://github.com/NVIDIA/TensorRT-LLM/pull/13996) | merged | Perf optimizations for DFlash | DFlash model engine and speculative code |
| 2026-05-29 | [#14659](https://github.com/NVIDIA/TensorRT-LLM/pull/14659) | merged | Add a reasoning parser for qwen3_5 | `reasoning_parser.py` |
| 2026-06-02 | [#14667](https://github.com/NVIDIA/TensorRT-LLM/pull/14667) | merged | AutoDeploy: Qwen3.5 400B NVFP4 accuracy regression fix | shared expert sharding, SwiGLU fusion |
| 2026-06-05 | [#15001](https://github.com/NVIDIA/TensorRT-LLM/pull/15001) | merged | Uncomment Qwen3.5 from model registry | `models.yaml` |
| 2026-06-09 | [#15081](https://github.com/NVIDIA/TensorRT-LLM/pull/15081) | merged | Select CUTLASS MoE backend on non-Blackwell SMs | accuracy test backend selection |
| 2026-06-11 | [#15067](https://github.com/NVIDIA/TensorRT-LLM/pull/15067) | merged | Generalize FP8 checkpoint loading for Qwen3.5 | weight mapper, modeling |
| 2026-06-13 | [#15185](https://github.com/NVIDIA/TensorRT-LLM/pull/15185) | merged | Qwen3.5 whitelist sharding and lm_head sharding | AutoDeploy sharding IR/tests |
| 2026-06-26 | [#15543](https://github.com/NVIDIA/TensorRT-LLM/pull/15543) | merged | Add EPLB support for Qwen3.5 | MoE load balancer and B200/GB200 tests |

## 逐 PR diff 审计卡

### PR #14599 - 支持 Qwen3.5 VL MoE 并修复 MTP

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14599
- 状态/时间: merged / 2026-07-04
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 1,734 行 diff，16 个文件，+1140/-256。
- 动机: TensorRT-LLM 已有可复用的 Qwen3Next 文本 runtime 和 Qwen3-VL vision tower，但缺少 Qwen3.5-35B-A3B 的复合多模态架构、原生配置归一化、权重映射以及 speculative decoding 的 token 传递。
- 实现要点: 注册 `Qwen3_5MoeForConditionalGeneration`，保留 HF text/vision 子配置并归一化 runtime alias，把 `Qwen3VisionModel` 与 MoE decoder 组合起来，映射语言模型权重；VLM wrapper 构造 embedding 后，MTP/Eagle 从 `orig_input_ids` 取回 prompt token。
- 代码 diff 细节: 新 VLM 类负责多模态 placeholder 元数据与 device path，同时复用 Qwen3Next LM；测试覆盖配置路由、权重加载、模态 parity、MTP 和 MMMU accuracy。
- 关键代码摘录:

```diff
+@register_auto_model("Qwen3_5MoeForConditionalGeneration")
+class Qwen3_5MoeVLModel(Qwen3VLModelBase):
+    """VLM wrapper composing Qwen3 vision encoder with Qwen3.5 MoE text decoder."""
+    kwargs["vision_model_class"] = Qwen3VisionModel
```

- 已读文件: runtime：`modeling_qwen3_5.py`、`qwen3_5_weight_mapper.py`、`modeling_speculative.py`、model loader/config utilities；测试/文档：`test_modeling_qwen3_5_vl_moe.py`、MMMU references、supported-model matrix。
- 验证与风险: benchmark 必须区分 text-only Qwen3.5 与 MoE VLM；需验证 mRoPE、图片/视频 placeholder、FP8 exclude-module 归一化和 MTP prompt-token 恢复。

### PR #15249 - 支持 Qwen3.5 VL Dense

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15249
- 状态/时间: merged / 2026-07-07
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 776 行 diff，11 个文件，+594/-26。
- 动机: MoE VLM wrapper 无法覆盖 Dense Qwen3.5-27B；后者使用 Dense `GatedMLP`，并有不同的 architecture 注册。
- 实现要点: 泛化共享 Qwen3.5 VLM 基类，注册 `Qwen3_5ForConditionalGeneration`，在保留 Qwen3 vision/mRoPE 处理的同时选择 Dense decoder，并把同一 HF mapper 扩展到 Dense 架构。
- 代码 diff 细节: 生产配置归一化保留 Dense `intermediate_size` 与空 deepstack index；新 parity suite 覆盖 image、multi-image、video、chunked-prefill position slicing 和模型构造。
- 关键代码摘录:

```diff
+@register_auto_model("Qwen3_5ForConditionalGeneration")
+class Qwen3_5VLModel(_Qwen3_5VLModel):
+    """VLM wrapper composing Qwen3 vision encoder with dense Qwen3.5 text decoder."""
```

- 已读文件: runtime：`modeling_qwen3_5.py`、`qwen3_5_weight_mapper.py`、model/config registries；测试/文档：`test_modeling_qwen3_5_vl.py`、MMMU references、supported-model matrix。
- 验证与风险: Dense 与 MoE checkpoint 需要独立的 accuracy/performance 行；需验证 composite config alias、mRoPE chunk slicing、SSM-cache dtype 和多模态 forward parity。

### PR #16469 - 融合 Qwen3.5/3.6 注意力预处理

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/16469
- 状态/时间: merged / 2026-07-21
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 913 行 diff，6 个文件，+775/-19。
- 动机: full-attention layer 过去分别执行 Q/gate 解交错、Q/K 归一化、RoPE、V 拷贝，并在输出端另启 sigmoid/multiply gate。
- 实现要点: 新增 Triton fast path，一次读取 interleaved projection，完成 Gemma RMSNorm、普通/交错 mRoPE 并输出 packed QKV 与 gate；输出 gate 用 in-place fused sigmoid-multiply，不支持的布局回退到通用路径。
- 代码 diff 细节: `QKNormRoPEAttention.preprocess_qkv` 按 dtype/layout/RoPE contract 控制融合，使权重加载、LoRA、编译、HIP 和不支持的 scaling 与优化解耦。
- 关键代码摘录:

```diff
+qkv, gate = fused_qkv_gemma_rmsnorm_rope_gate(
+    qkv, self.q_norm.weight, self.k_norm.weight,
+    self.rotary_emb.rotary_cos_sin, positions.contiguous(), ...)
+return qkv, None, None, gate
```

- 已读文件: runtime：`attention.py`、`qk_norm_attention.py`、`modeling_qwen3_next.py`、`fused_qk_norm_rope_gate.py`；测试：`test_fused_qk_norm_rope_gate.py`、B200 test database。
- 验证与风险: 需验证 BF16/FP16、partial/full rotary dim、mRoPE section、zero-token 与非连续输入，并同时对照 Python reference 和生产 THOP 路径。

### PR #15194 - 为 Qwen3-Next/Qwen3.5 融合 Gemma RMSNorm 与 AllReduce

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15194
- 状态/时间: merged / 2026-07-24
- 反查来源: `git log --name-only -- <model-files>`，并结合最终上游提交和 PR 正文。
- 代码 diff 已读范围: 完整 418 行 diff，3 个文件，+278/-29。
- 动机: TP Qwen3.5 decoder layer 需要分别做 collective 和 Gemma RMSNorm，而已有融合 collective 读取普通 norm weight，不包含 Gemma 的 `(1 + weight)` 语义。
- 实现要点: 对非 attention-DP 的 TP 默认开启 eager fusion，把 attention/MoE collective 延迟到 `RESIDUAL_RMS_NORM`，权重加载后一次性预计算 Gemma-adjusted norm weight；MTP 没有下一层 norm 时禁用 post-MoE fusion。
- 代码 diff 细节: 同一 PR 还为 FlashInfer GDN decode 增加仅在标量 slice 未满足 CuTe-DSL 32-byte 对齐时 clone 的保护，避免拖慢已对齐 shape。
- 关键代码摘录:

```diff
+norm._fused_norm_weight = (w.float() + 1.0).to(w.dtype)
+fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
+norm_weight=_fused_norm_weight(self.post_attention_layernorm),
```

- 已读文件: runtime：`modeling_qwen3_next.py`、`fused_sigmoid_gating_recurrent.py`；测试：`test_qwen3_next_eager_fusion.py`。
- 验证与风险: 需确认 collective 只有一个 owner、attention-DP 保持非融合、加载后 derived weight 会刷新、Gemma 数值使用 `(1 + weight)`，未对齐 Qwen3.6 head slice 走受控 copy。

### PR #11728 - Added Qwen3.5 Cookbook

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/11728
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+385/-0，本地 patch 402 行。
- 动机: 给 `Qwen/Qwen3.5-397B-A17B` 和 `nvidia/Qwen3.5-397B-A17B-NVFP4` 提供 AutoDeploy cookbook，明确 TensorRT-LLM 的服务命令和 B200 资源假设。
- 实现要点: notebook 写入 `trtllm-serve` 命令、AutoDeploy registry 配置、NVFP4 4xB200 说明和示例 OpenAI 请求。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+trtllm-serve "nvidia/Qwen3.5-397B-A17B-NVFP4" \
+MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
```

- 已读文件: `examples/auto_deploy/cookbooks/qwen_3.5_trtllm_cookbook.ipynb`
- 验证与风险: 这是公平 benchmark 的 TensorRT-LLM deployment 证据；需要和 PyTorch backend / AutoDeploy registry 配置一起记录。

### PR #12302 - Add Qwen 3.5 supporting (NVFP4)

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12302
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 9 个文件，+225/-31，本地 patch 436 行。
- 动机: TensorRT-LLM PyTorch backend 需要支持 Qwen3.5 dense/MoE 和官方 NVFP4 checkpoint。
- 实现要点: 注册 `Qwen3_5ForCausalLM` 和 `Qwen3_5MoeForCausalLM`，扩展 HF mapper，normalize exclude modules，并新增 397B A17B NVFP4 accuracy tests。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+@register_auto_model("Qwen3_5ForCausalLM")
+class Qwen3_5ForCausalLM(Qwen3NextForCausalLM):
```

- 已读文件: `modeling_qwen3_5.py`, `qwen3_5_weight_mapper.py`, `config_utils.py`, accuracy refs/tests
- 验证与风险: 与 SGLang 对比时要区分 dense 和 MoE wrapper，以及 NVFP4 exclude module 规则。

### PR #12114 - Qwen 3.5 fix 3D position ID handling

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/12114
- 状态/时间: merged / 2026-03-25
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 15 个文件，+3448/-275，本地 patch 7,822 行。
- 动机: Qwen3.5 VLM/mRoPE 需要 3D position IDs、chunked multimodal positions、video grid normalization 和 mRoPE delta cache，原 AutoDeploy path 不完整。
- 实现要点: 重写/扩展 `modeling_qwen3_5_moe.py` 的 multimodal input processor、mRoPE delta cache transform、registry configs 和单元测试。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+@TransformRegistry.register("initialize_mrope_delta_cache")
+mm_token_positions: torch.Tensor
```

- 已读文件: `modeling_qwen3_5_moe.py`, `mrope_delta_cache.py`, registry YAML, `test_qwen3_5_moe.py`, serving utils tests
- 验证与风险: 多模态 Qwen3.5 不能只比较 decode kernel；position construction 和 cache resource 命名也会影响 correctness。

### PR #13090 - Qwen3.5 dense weight loading

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13090
- 状态/时间: merged / 2026-04-30
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+85/-1，本地 patch 225 行。
- 动机: Qwen3.5 dense 4B/FP8 weight loading 需要独立覆盖，不能只靠 MoE mapper 过关。
- 实现要点: 扩展 `qwen3_5_weight_mapper.py`，新增 `Qwen/Qwen3.5-4B` accuracy refs 和 `TestQwen3_5_4B`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+class TestQwen3_5_4B(LlmapiAccuracyTestHarness):
+MODEL_NAME = "Qwen/Qwen3.5-4B"
```

- 已读文件: HF mapper, accuracy refs, `test_llm_api_pytorch.py`, test lists
- 验证与风险: SOTA loop 如果选 dense Qwen3.5，不能直接套 397B MoE 的 mapper 风险结论。

### PR #13716 - Preserve Qwen3.5 NVFP4 weight_scales

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13716
- 状态/时间: merged / 2026-05-04
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+9/-3，本地 patch 45 行。
- 动机: mapper 把 `weight_scales` 归一成 `weight_scale_inv` 的 FP8 逻辑会破坏 NVFP4 loader。
- 实现要点: 在 HF mapper 中识别 NVFP4 prefix，保留 `weight_scales` 给 `NVFP4LinearMethod.load_weight_scales`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+        nvfp4_prefixes = {
+            key[: -len(".weight_scale_2")] for key in weights if key.endswith(".weight_scale_2")
+        }
+                if prefix not in nvfp4_prefixes:
```

- 已读文件: `qwen3_5_weight_mapper.py`
- 验证与风险: Qwen3.5 NVFP4 对 scale 名称敏感；SGLang 对比 weight loader 时要检查 scale key remap。

### PR #13782 - Qwen3.5 DFlash

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13782
- 状态/时间: merged / 2026-05-12
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+144/-55，本地 patch 413 行。
- 动机: Qwen3.5 speculative/DFlash 需要把 GDN/Mamba cache 和 pyexecutor/model engine 接到 DFlash runtime。
- 实现要点: 修改 `gdn_mixer.py`、`mamba_cache_manager.py`、`model_engine.py` 和 `speculative/dflash.py`，让 hybrid linear-attention 模型能参与 DFlash。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+from tensorrt_llm._torch.speculative import dflash
+mamba_cache_manager
```

- 已读文件: `gdn_mixer.py`, `pyexecutor/_util.py`, `mamba_cache_manager.py`, `model_engine.py`, `speculative/dflash.py`
- 验证与风险: 对 SGLang MTP/speculative 对比时，必须记录 DFlash 是否启用；它会改变 decode state update 形态。

### PR #13996 - Perf optimizations for DFlash

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/13996
- 状态/时间: merged / 2026-05-16
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+455/-285，本地 patch 1,606 行。
- 动机: DFlash 初始支持后需要减少状态搬运和模型 engine overhead。
- 实现要点: 调整 speculative modeling、GDN mixer、model engine 和 `llm_args.py`，把 DFlash path 做成更稳定的 perf path。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+    def _build_fused_kv_buffers(self) -> None:
+        """Stack per-layer KV projection + k_norm weights for a single fused GEMM.
+        return self.max_draft_len + 1
```

- 已读文件: `modeling_speculative.py`, `gdn_mixer.py`, `model_engine.py`, `speculative/dflash.py`, `llm_args.py`
- 验证与风险: 如果 TensorRT-LLM 领先来自 DFlash，需要和 SGLang MTP/SpecV2 分开归因。

### PR #14659 - Add a reasoning parser for qwen3_5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14659
- 状态/时间: merged / 2026-05-29
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+9/-0，本地 patch 30 行。
- 动机: Qwen3.5 forced-thinking 模板输出一开始就在 reasoning block 内，旧 `qwen3` parser 的 `reasoning_at_start=False` 不匹配。
- 实现要点: 注册 `qwen3_5` parser，并设置 `reasoning_at_start=True`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+@register_reasoning_parser("qwen3_5", reasoning_at_start=True)
```

- 已读文件: `tensorrt_llm/llmapi/reasoning_parser.py`
- 验证与风险: benchmark 输出清洗/CoT 解析不一致会影响评测分数，不能只看吞吐。

### PR #14667 - AutoDeploy Qwen3.5 400B NVFP4 accuracy regression fix

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/14667
- 状态/时间: merged / 2026-06-02
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+72/-35，本地 patch 464 行。
- 动机: AutoDeploy Qwen3.5 400B NVFP4 出现 accuracy regression，shared expert sharding 与 SwiGLU fusion 需要修正。
- 实现要点: 将 shared expert 复制而不是 TP sharding，加入 whitelist sharding hints，并扩展 SwiGLU fusion pattern。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+# The shared expert is replicated
+apply_sharding_hints:
```

- 已读文件: `qwen3.5_moe_400b.yaml`, `modeling_qwen3_5_moe.py`, `custom_ops/linear/swiglu.py`, `fuse_swiglu.py`, waives
- 验证与风险: SGLang 若遇到 Qwen3.5 MoE 精度差异，应检查 shared expert 并行策略和 SwiGLU fusion，不要只比较 MoE matmul。

### PR #15001 - Uncomment Qwen3.5 from model registry

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15001
- 状态/时间: merged / 2026-06-05
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 1 个文件，+9/-12，本地 patch 50 行。
- 动机: Qwen3.5 AutoDeploy registry entry 从注释状态变成默认可发现。
- 实现要点: 在 `models.yaml` 启用 `Qwen/Qwen3.5-35B-A3B` 和 `Qwen/Qwen3.5-397B-A17B`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+- name: Qwen/Qwen3.5-397B-A17B
+  config_id: qwen3_5_moe_400b
```

- 已读文件: `examples/auto_deploy/model_registry/models.yaml`
- 验证与风险: SOTA loop 可以把 registry entry 视作 TensorRT-LLM 官方 AutoDeploy lane，而不是临时命令。

### PR #15081 - Select CUTLASS MoE backend on non-Blackwell SMs

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15081
- 状态/时间: merged / 2026-06-09
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+8/-2，本地 patch 52 行。
- 动机: Qwen3.5 FP8 test 在非 Blackwell SM 上不应默认使用 DeepGEMM，需要 fallback 到 CUTLASS MoE backend。
- 实现要点: accuracy test 按 SM 版本选择 `DEEPGEMM` 或 `CUTLASS`。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+moe_backend = "DEEPGEMM" if get_sm_version() in (100, 103) else "CUTLASS"
```

- 已读文件: `test_llm_api_pytorch.py`, `waives.txt`
- 验证与风险: 跨 GPU 比较必须记录 MoE backend；否则 H100/B200 结果不可直接混用。

### PR #15067 - Generalize FP8 checkpoint loading for Qwen3.5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15067
- 状态/时间: merged / 2026-06-11
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 2 个文件，+68/-48，本地 patch 220 行。
- 动机: Qwen3.5 FP8 checkpoint 的 scale/exclude module 命名需要更通用的 mapper 处理。
- 实现要点: 重构 `qwen3_5_weight_mapper.py` 与 `modeling_qwen3_5.py` 中的 FP8/NVFP4 normalization。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+    # gdn_mixer uses Linear module for weight management of depthwise conv1d
+    # but conv1d is not a proper linear module and should be excluded from quant
+    normalized.add("*linear_attn.conv1d")
```

- 已读文件: `qwen3_5_weight_mapper.py`, `modeling_qwen3_5.py`
- 验证与风险: 对 FP8 checkpoint 的 load failure 或精度差异，先查 mapper normalization，而不是 kernel。

### PR #15185 - Qwen3.5 whitelist sharding and lm_head sharding

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15185
- 状态/时间: merged / 2026-06-13
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 5 个文件，+193/-118，本地 patch 735 行。
- 动机: AutoDeploy Qwen3.5 需要白名单式 sharding，并让 `lm_head` 参与 sharding，以减少单 rank 负载和保持图变换可控。
- 实现要点: 更新 registry YAML、`modeling_qwen3_5_moe.py` sharding hints、`fuse_swiglu.py` 和 `sharding_ir.py` 测试。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+lm_head:
+apply_sharding_hints
```

- 已读文件: `qwen3.5_moe_400b.yaml`, `modeling_qwen3_5_moe.py`, `fuse_swiglu.py`, `sharding_ir.py`, tests
- 验证与风险: SGLang 对标时应单独观察 lm_head 和 shared expert 的 sharding/communication，而不是只看 MoE expert GEMM。

### PR #15543 - Add EPLB support for Qwen3.5

- 链接: https://github.com/NVIDIA/TensorRT-LLM/pull/15543
- 状态/时间: merged / 2026-06-26
- 反查来源: `git log --name-only -- <model-files>` 与 GitHub Pull Request files API。
- 代码 diff 已读范围: 3 个文件，+73/-0，本地 patch 130 行。
- 动机: Qwen3.5 MoE 需要 EPLB，尤其是 B200/GB200 perf sanity 和 PyTorch backend test coverage。
- 实现要点: 在 `moe_load_balancer.py` 增加 Qwen3.5 支持，并把 B200/GB200 test-db lane 接入。
- 代码 diff 细节: 见上方已读范围和下方摘录，保留本卡审计到的文件级变化。
- 关键代码摘录:

```diff
+    'Qwen2MoeForCausalLM',
+    'Qwen3MoeForCausalLM',
+    'Qwen3_5MoeForCausalLM',
```

- 已读文件: `moe_load_balancer.py`, `test_llm_api_pytorch.py`, B200/GB200 test-db YAML
- 验证与风险: 与 SGLang 的 EPLB/DeepEP/EP 对比时，必须记录 load-balancer 是否启用和测试集群拓扑。
