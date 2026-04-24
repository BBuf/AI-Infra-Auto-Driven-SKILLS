# vLLM Qwen2.5-VL / Qwen3-VL / Qwen3-Omni / Qwen3-ASR 支持与 PR 历史

本文记录 vLLM 在提交 `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
附近的多模态 Qwen 家族支持情况。

- 状态: 当前 mainline 已支持

## 核心结论

- 这条家族线已经覆盖 Qwen2.5-VL、Qwen3-VL、Qwen3-VL-MoE、
  Qwen3-Omni thinker、Qwen3-ASR，以及 realtime Qwen3-ASR。
- 真正高风险的地方主要在:
  placeholder 展开、视频时间戳、interleaved MRoPE、
  `use_audio_in_video`、音频特征长度换算，以及 realtime prompt 扩展。

## 主要代码面

- `vllm/vllm/model_executor/models/qwen2_5_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl.py`
- `vllm/vllm/model_executor/models/qwen3_vl_moe.py`
- `vllm/vllm/model_executor/models/qwen3_omni_moe_thinker.py`
- `vllm/vllm/model_executor/models/qwen3_asr.py`
- `vllm/vllm/model_executor/models/qwen3_asr_realtime.py`
- `vllm/vllm/model_executor/layers/rotary_embedding/mrope.py`

## 已合入 PR

- [#13155](https://github.com/vllm-project/vllm/pull/13155)
  `Qwen2.5-VL Optimization`
  已审 diff: `2` 个文件，`47` 行新增，`51` 行删除。
  主要优化 Qwen2.5-VL 的视觉注意力 fallback 路径，并切到共享 `RMSNorm`。
- [#24727](https://github.com/vllm-project/vllm/pull/24727)
  `Support Qwen3-VL Model Series`
  已审 diff: `13` 个文件，`2084` 行新增，`17` 行删除。
  这是 Qwen3-VL / Qwen3-VL-MoE 的主落地 PR，并补了视频 placeholder 和处理链。
- [#25055](https://github.com/vllm-project/vllm/pull/25055)
  `Add Triton kernel for Qwen3-VL interleaved MRoPE`
  已审 diff: `2` 个文件，`88` 行新增，`46` 行删除。
  它让 Qwen3-VL 的 interleaved MRoPE 真正成为受测的主路径。
- [#25550](https://github.com/vllm-project/vllm/pull/25550)
  `Add Qwen3-Omni moe thinker`
  已审 diff: `6` 个文件，`1795` 行新增，`36` 行删除。
  它加入 thinker 运行时，并单独处理 `use_audio_in_video` 的 placeholder 记账。
- [#33312](https://github.com/vllm-project/vllm/pull/33312)
  `Qwen3-ASR`
  已审 diff: `9` 个文件，`1269` 行新增。
  它补了 Qwen3-ASR 的 config、processor、模型和 transcription 路径。
- [#34613](https://github.com/vllm-project/vllm/pull/34613)
  `Add Qwen3-ASR realtime streaming support`
  已审 diff: `5` 个文件，`256` 行新增，`1` 行删除。
  它加入 realtime 子类、音频 buffer 和 prompt 扩展逻辑。

## 当前结论

这条家族线出问题时，不要笼统地说“多模态坏了”。要先定位是
Qwen2.5-VL attention fallback、Qwen3-VL 视频 prompt replacement、
Qwen3-Omni 的 audio-in-video 记账，还是 Qwen3-ASR 的
prompt / audio length 逻辑出错。
