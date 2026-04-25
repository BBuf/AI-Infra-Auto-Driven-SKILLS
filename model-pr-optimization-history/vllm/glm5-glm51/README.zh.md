# vLLM GLM-5 / 5.1 支持与 PR 历史

本文记录 vLLM 在提交 `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
附近对 GLM-5 / 5.1 的已落地支持。

- 状态: 当前 mainline 仅部分支持

## 核心结论

- GLM-5 现在不是独立的 `glm5.py` 实现，而是通过 DeepSeek-V2/V3 的
  MLA/MoE 运行时做适配。
- 当前最关键的两步是:
  `#34124` 负责架构和配置适配，
  `#34385` 负责修正 MTP 草稿模型的 logits 正确性。

## 主要代码面

- `vllm/vllm/model_executor/models/deepseek_v2.py`
- `vllm/vllm/model_executor/models/registry.py`
- `vllm/vllm/config/speculative.py`
- `vllm/vllm/transformers_utils/model_arch_config_convertor.py`
- `vllm/vllm/v1/spec_decode/eagle.py`

## 已合入 PR

- [#34124](https://github.com/vllm-project/vllm/pull/34124)
  `GLM adaptation`
  已审 diff: `7` 个文件，`13` 行新增，`3` 行删除。
  它把 `GlmMoeDsaForCausalLM` 接到 DeepSeek-V2 运行时上，并补了
  speculative config 与 `indexer_rope_interleave` 适配。
- [#34385](https://github.com/vllm-project/vllm/pull/34385)
  `Fix MTP accuracy for GLM-5`
  已审 diff: `1` 个文件，`18` 行新增。
  它把目标模型 `lm_head` 显式共享给每个 MTP layer 的
  `shared_head.head`，否则 GLM-5 draft logits 会出现 NaN 或未初始化输出。

## 当前结论

遇到 GLM-5 问题时，优先检查 DeepSeek 复用路径和 speculative decode 基础
设施，而不是先去看旧的 `glm4*` 文件。
