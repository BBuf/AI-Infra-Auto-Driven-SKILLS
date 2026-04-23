# SGLang Qwen VLM / Omni / ASR 支持与优化历史

本文覆盖 Qwen2.5-VL、Qwen3-VL、Qwen3-VL-MoE、Qwen3-Omni、Qwen3-ASR，以及共享 Qwen VLM processor / encoder-disaggregation 路径的 Qwen3.5 多模态改动。

证据口径：

- 每个 SGLang PR 都已阅读 `gh pr view` 与 `gh pr diff --patch`，必要时再读 merge commit diff。
- 逐 PR 的完整源码卡片见 `skills/model-optimization/sglang/sglang-qwen-vlm-omni-asr-optimization/references/pr-history.md`。
- 本 README 保留每个 PR 的动机、关键实现和最重要源码锚点，便于按时间线阅读。
- open / closed-unmerged PR 只作为雷达，不写成 main 分支事实。
- 已补充阅读 SGLang Qwen3-VL 官方文档、LMSYS AMD Qwen3/Qwen3-VL 延迟优化博客、SGLang `#18466` AMD 跟踪 issue，以及 sgl-cookbook Qwen VLM AMD 部署 PR。

## 结论

Qwen 多模态优化的核心不在纯语言模型 forward，而在这些链路：

- processor：图像、视频、音频、ASR 输入展开和 placeholder 对齐。
- ViT：attention backend、CUDA graph、cos/sin cache、cu_seqlens、DeepStack。
- mRoPE / 3D mRoPE：图像、视频、音频和 EVS token pruning 都会影响位置。
- 分布式：encoder DP、PP、EPD、MoE expert 权重加载、LoRA adapter routing。
- 缓存与传输：长视频/多图 chunk、per-image cache、lazy device transfer。
- 硬件路径：AMD AITER/ROCm、NPU processor/audio loader、CPU SDPA/AMX。
- ASR：Qwen3-ASR model bring-up、HTTP streaming、WebSocket realtime input。

## 主线 PR 时间线

### #10911 Qwen3-Omni thinker-only bring-up

- 链接：https://github.com/sgl-project/sglang/pull/10911，已合入。
- 动机：SGLang 需要识别 Qwen3-Omni 的 thinker/talker/code2wav 嵌套配置，并把 audio 输入接入 Qwen VL processor 与 mRoPE。
- 实现：注册 `Qwen3OmniMoeForConditionalGeneration`，新增 `Qwen3OmniMoeConfig`，`base_processor.py` 处理 audio，`qwen_vl.py` 计算 `audio_feature_lengths` 并传给 `MRotaryEmbedding.get_rope_index`。
- 关键代码：

```python
audio_feature_lengths = torch.sum(audio_item.feature_attention_mask, dim=1)
MRotaryEmbedding.get_rope_index(..., audio_seqlens=audio_feature_lengths)
```

### #10985 Qwen3-VL MRotaryEmbedding 参数修复

- 链接：https://github.com/sgl-project/sglang/pull/10985，已合入。
- 动机：fused KV buffer 改动把 `fused_set_kv_buffer_arg` 传进 rotary embedding，但 Qwen3-VL 的 `MRotaryEmbedding` 不支持这个保存 KV 的路径。
- 实现：`MRotaryEmbedding.forward` 接受可选参数但 assert 不使用；Qwen3 attention 检测到 mRoPE 后关闭 fused KV buffer 兼容标志。
- 关键代码：

```python
self.compatible_with_fused_kv_buffer = (
    False if isinstance(self.rotary_emb, MRotaryEmbedding) else True
)
```

### #12333 Qwen3-VL pipeline parallelism

- 链接：https://github.com/sgl-project/sglang/pull/12333，已合入。
- 动机：Qwen3-VL 在 `--tp 2 --pp-size 2` 下缺少 PP-aware 的媒体 embedding、logits、权重加载和 rank-local layer 处理。
- 实现：加入 `get_pp_group()`；非最后 PP rank 用 `PPMissingLayer`；非最后 rank 返回 hidden states；最后 rank 才计算 logits；权重加载时只在最后 rank 处理 tied `lm_head`。
- 关键代码：

```python
if self.pp_group.is_last_rank:
    self.lm_head = ParallelLMHead(...)
else:
    self.lm_head = PPMissingLayer()
```

### #13724 Qwen3-VL vision encoder DP

- 链接：https://github.com/sgl-project/sglang/pull/13724，已合入。
- 动机：高并发图像/视频时 ViT 是 TTFT 瓶颈，需要把 vision encoder 做 DP sharding。
- 实现：vision MLP/block/patch merger 接收 `use_data_parallel`；DP 下 vision TP size/rank 固定为 1/0；image/video feature 走 `run_dp_sharded_mrope_vision_model(..., rope_type="rope_3d")`。
- 关键代码：

```python
if self.use_data_parallel:
    return run_dp_sharded_mrope_vision_model(
        self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
    )
```

### #13736 Qwen-VL cu_seqlens NumPy 优化

- 链接：https://github.com/sgl-project/sglang/pull/13736，已合入。
- 动机：ViT `cu_seqlens` 构造里的 CPU `torch.repeat_interleave` 出现在 TTFT profile 中。
- 实现：新增 `compute_cu_seqlens_from_grid_numpy`，用 `np.repeat` 和 `cumsum(np.int32)` 替换 PyTorch CPU op，Qwen2/Qwen3 VLM 复用。
- 关键代码：

```python
cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(
    axis=0, dtype=np.int32
)
```

### #14292 Qwen-VL rotary position-id cache

- 链接：https://github.com/sgl-project/sglang/pull/14292，已合入。
- 动机：2D rotary position id 每次按图像尺寸重建，浪费 CPU 时间。
- 实现：新增 `RotaryPosMixin.rot_pos_ids(h,w,spatial_merge_size)`，使用 `lru_cache(maxsize=1024)`，Qwen2.5-VL 与 Qwen3-VL 复用。
- 关键代码：

```python
@lru_cache(maxsize=1024)
def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
```

### #14907 chunked ViT attention

- 链接：https://github.com/sgl-project/sglang/pull/14907，已合入。
- 动机：Qwen3-VL-235B FP8 在单请求几百张图/帧时 ViT 一次性处理会 OOM。
- 实现：新增 `SGLANG_VLM_MAX_PATCHES_PER_VIT` / `SGLANG_VLM_MAX_IMAGES_PER_VIT`，按图像边界切 `pixel_values` 和 `grid_thw`，逐 chunk 跑 ViT 后 concat。
- 关键代码：

```python
chunk_embeds = self.visual(pixel_chunk, grid_thw=grid_chunk)
all_chunk_embeds.append(chunk_embeds)
return torch.cat(all_chunk_embeds, dim=0)
```

### #15205 Qwen3-VL / GLM-4.1V vision RoPE cos/sin cache

- 链接：https://github.com/sgl-project/sglang/pull/15205，已合入。
- 动机：vision RoPE 频繁重新计算 cos/sin；PR body 中单段路径约 490us 降到 186us，并报告 Qwen3-VL TTFT 约 2% 收益。
- 实现：`RotaryEmbedding` 暴露 `get_cos_sin`；`VisionAttention` 接收 `rotary_pos_emb_cos/sin`；Qwen3-VL 用 SGLang `get_rope` 替换 HF vision rotary embedding，并从 cache index 出 2D cos/sin。
- 关键代码：

```python
cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
cos_combined = cos[pos_ids].flatten(1)
sin_combined = sin[pos_ids].flatten(1)
```

### #15320 Qwen3-VL ViT piecewise CUDA graph

- 链接：https://github.com/sgl-project/sglang/pull/15320，已合入。
- 动机：把 Qwen3-VL ViT 计算纳入 PCG，支持 TP>1 与 DeepStack；PR 报告 8xH20 Qwen3-VL-8B TP4 TTFT 1384.53ms -> 1120.68ms。
- 实现：VisionAttention 解除 TP==1 graph 限制；Qwen3-VL 新增 `forward_with_cuda_graph`；`ViTCudaGraphRunner` 支持 Qwen3 DeepStack visual indexes 和 merger list。
- 关键代码：

```python
if get_bool_env_var("SGLANG_VIT_ENABLE_CUDA_GRAPH"):
    return self.forward_with_cuda_graph(x, grid_thw)
```

### #16366 Qwen3-VL video memory

- 链接：https://github.com/sgl-project/sglang/pull/16366，已合入。
- 动机：高并发视频时 `item.feature` 留在设备上，concat 后仍占显存，导致 OOM。
- 实现：concat 前移动到 visual device，得到 `pixel_values` 后把每个 `item.feature` 放回 CPU。
- 关键代码：

```python
pixel_values = torch.cat([item.feature for item in items], dim=0).type(self.visual.dtype)
for item in items:
    item.feature = item.feature.to("cpu")
```

### #17624 Qwen3-VL DP size > 1

- 链接：https://github.com/sgl-project/sglang/pull/17624，已合入。
- 动机：`--mm-enable-dp-encoder` 与 `--enable-dp-attention` 在 TP/DP 不一致时有 launch/精度问题，`mrope_positions` padding 维度也不对。
- 实现：`mrope_positions` 按 token 维 padding；DP sharded vision 使用 attention TP group；Qwen3-VL lm_head 支持 `enable_dp_lm_head`；RowParallelLinear 增加 attention TP all-reduce。
- 关键代码：

```python
self.mrope_positions = torch.cat([...], dim=1)
gathered_embeds = get_attention_tp_group().all_gather(image_embeds_local_padded, dim=0)
```

### #18024 Qwen3-VL untied lm_head 权重加载

- 链接：https://github.com/sgl-project/sglang/pull/18024，已合入。
- 动机：Qwen3-VL-8B 输出异常，因为 `embed_tokens.weight` 被无条件复制给 `lm_head.weight`，但模型可能 `tie_word_embeddings=False`。
- 实现：只在最后 PP rank 且 `self.config.tie_word_embeddings` 为真时复制。
- 关键代码：

```python
and self.config.tie_word_embeddings
```

### #18185 Qwen3-Omni audio encoder 优化

- 链接：https://github.com/sgl-project/sglang/pull/18185，已合入。
- 动机：Qwen3-Omni thinker audio/ASR 路径慢；PR body 报告 ASR throughput 约 0.28 -> 3.12 req/s。
- 实现：audio encoder FFN 改为 `ColumnParallelLinear` / `RowParallelLinear`；mask 构造向量化；小 batch conv 走 fast path；`feature_attention_mask` 移到 audio tower device。
- 关键代码：

```python
self.fc1 = ColumnParallelLinear(...)
self.fc2 = RowParallelLinear(...)
idx = torch.arange(max_len_after_cnn, device=padded_feature.device)
```

### #19003 FlashInfer CUDNN prefill ViT backend

- 链接：https://github.com/sgl-project/sglang/pull/19003，已合入。
- 动机：为 Qwen3-VL ViT 增加 `flashinfer_cudnn` attention backend；PR 报告 TTFT 1054ms -> 931ms。
- 实现：新增 `VisionFlashInferAttention`，调用 `cudnn_batch_prefill_with_kv_cache`；server args 增加 backend；Qwen3-VL 计算 packed q/k/v/o indptr、bucket batch/max-seqlen。
- 关键代码：

```python
output, _ = cudnn_batch_prefill_with_kv_cache(
    q, k, v, scale, self.workspace_buffer,
    batch_offsets_q=indptr_qk,
    batch_offsets_v=indptr_v,
    batch_offsets_o=indptr_o,
)
```

### #19291 Qwen3-VL missing quant_config

- 链接：https://github.com/sgl-project/sglang/pull/19291，已合入。
- 动机：Qwen3.5 NVFP4 版本走 Qwen3-VL 路径时 KV cache 退回 bf16，因为模型未保存 `quant_config`。
- 实现：初始化时保存 `self.quant_config = quant_config`。
- 关键代码：

```python
self.quant_config = quant_config
```

### #19333 Qwen3-VL visual module loading

- 链接：https://github.com/sgl-project/sglang/pull/19333，已合入。
- 动机：visual merger/visual prefix 映射丢失导致视觉权重没正确加载，图像回答变差。
- 实现：visual 权重加载分支补回 `model.visual.` -> `visual.` 映射。
- 关键代码：

```python
name = name.replace(r"model.visual.", r"visual.")
```

### #20759 / #20788 Qwen3-VL DP encoder position embedding hang

- 链接：
  - https://github.com/sgl-project/sglang/pull/20759，已合入。
  - https://github.com/sgl-project/sglang/pull/20788，已合入。
- 动机：DP encoder 时有 rank 没有 image item，TP position embedding/all-reduce 可能等待导致 hang。
- 实现：DP encoder 下 `pos_embed` 关闭 TP；`#20759` 是当前更完整规则，确保 `use_attn_tp_group` 也不在 DP encoder 关闭 TP 时误用。
- 关键代码：

```python
enable_tp=not use_data_parallel,
use_attn_tp_group=is_dp_attention_enabled() and not use_data_parallel,
```

### #21458 AMD Qwen3-VL decode fusion

- 链接：https://github.com/sgl-project/sglang/pull/21458，已合入。
- 动机：ROCm decode path 中 QKV split、QK RMSNorm、3D mRoPE、KV cache 写入是分散 kernel。
- 实现：检测 HIP + AITER + `MRotaryEmbedding` + `mrope_section`，调用 `fused_qk_norm_mrope_3d_cache_pts_quant_shuffle`，并在 attention 中 `save_kv_cache=False`。
- 关键代码：

```python
self.use_fused_qk_norm_mrope = (
    _has_fused_qk_norm_mrope and isinstance(self.rotary_emb, MRotaryEmbedding)
)
```

### #21469 Qwen3-VL-30B-A3B-Instruct LoRA

- 链接：https://github.com/sgl-project/sglang/pull/21469，已合入。
- 动机：Qwen3-VL-MoE 需要支持 30B-A3B-Instruct 的 LoRA adapter，尤其 MoE expert、lm_head、embed_tokens 目标。
- 实现：扩展 `Qwen3VLMoeForConditionalGeneration.should_apply_lora` 正则；新增 H200 registered logprob-diff 测试。
- 关键代码：

```python
r"^(?:model\.layers\.(\d+)\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.experts)|lm_head|model\.embed_tokens)$"
```

### #21849 Qwen3.5 VLM encoder disaggregation

- 链接：https://github.com/sgl-project/sglang/pull/21849，已合入。
- 动机：Qwen3.5 多模态 runtime 已支持，但 EPD 启动 allowlist 不包含 `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`。
- 实现：加入架构 allowlist，并把 Qwen3.5 dense/MoE 加入 video timestamp metadata 处理。
- 关键代码：

```python
self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
```

### #22038 chunk-aware ViT encoding cache / lazy transfer

- 链接：https://github.com/sgl-project/sglang/pull/22038，已合入。
- 动机：早期 request-level chunked ViT 会编码过多媒体并过早搬到 GPU；长视频/多图 chunked prefill 需要 per-image cache。
- 实现：在 `mm_utils.py` 中按 item 与当前 token chunk 的 overlap 判断是否编码；命中 `embedding_cache.get_single(item.hash)`；miss 的 item 才 `_move_items_to_device`；Qwen3-VL 移除内部 env chunking。
- 关键代码：

```python
cached = embedding_cache.get_single(item.hash)
_move_items_to_device(miss_item_list, device)
all_miss_embedding = data_embedding_func(miss_item_list)
```

### #22073 Qwen3-ASR support

- 链接：https://github.com/sgl-project/sglang/pull/22073，已合入。
- 动机：实现 Qwen3-ASR 0.6B/1.7B，通过 `/v1/audio/transcriptions` 服务 ASR。
- 实现：新增 `Qwen3ASRProcessor`，把单个 `<|audio_pad|>` 展开为真实 audio token 数；新增 `Qwen3ASRForConditionalGeneration`，复用 `Qwen3OmniMoeAudioEncoder` 与 `Qwen3ForCausalLM`；权重加载重映射 thinker audio/model/lm_head；OpenAI transcription route 加 Qwen3-ASR adapter。
- 关键代码：

```python
audio_token_counts = self._get_feat_extract_output_lengths(feat_lengths)
new_ids.extend([audio_pad_id] * n)
```

### #22089 Qwen3-ASR chunk-based streaming

- 链接：https://github.com/sgl-project/sglang/pull/22089，已合入。
- 动机：`#22073` 只能完整上传音频后返回最终结果，需要边转写边输出。
- 实现：新增 `StreamingASRState` 和 `split_audio_chunks`；Qwen3-ASR adapter 配置 2 秒 chunk、unfixed chunk/token；`serving_transcription.py` 输出 SSE word delta，并处理 chunk 间空格。
- 关键代码：

```python
if self._adapter.supports_chunked_streaming:
    return StreamingResponse(..., media_type="text/event-stream")
```

### #22230 Qwen3-VL EAGLE3

- 链接：https://github.com/sgl-project/sglang/pull/22230，已合入。
- 动机：Qwen3-VL 需要 EAGLE3 speculative decoding，且要捕获 auxiliary hidden states。
- 实现：加入 `capture_aux_hidden_states`；forward 中将 aux hidden states 传给 `logits_processor`；新增 `get_embed_and_head` 与 `set_eagle3_layers_to_capture`。
- 关键代码：

```python
return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states)
```

### #22266 NPU Qwen3.5 video processor

- 链接：https://github.com/sgl-project/sglang/pull/22266，已合入。
- 动机：Qwen3.5 video processor 中超过 8 维的 `permute` 不被 NPU 支持。
- 实现：patch Transformers `Qwen3VLVideoProcessor._preprocess`，用 NPU 兼容 reshape/permute 展平 patch。
- 关键代码：

```python
apply_module_patch(
    "transformers.models.qwen3_vl.video_processing_qwen3_vl.Qwen3VLVideoProcessor",
    "_preprocess",
    [npu_wrapper_video_preprocess],
)
```

### #22431 Qwen3.5 processor_output video

- 链接：https://github.com/sgl-project/sglang/pull/22431，已合入。
- 动机：用户传入 processor output 格式的视频时，`preprocess_video` 返回单值，后续代码期望 `(video, metadata)`。
- 实现：非 `VideoDecoderWrapper` 时返回 `(vr, None)`。
- 关键代码：

```python
if not is_video_obj:
    return vr, None
```

## 文档与部署证据

### #12554 Qwen3-VL 官方文档

- 链接：https://github.com/sgl-project/sglang/pull/12554，已合入。
- 内容：新增 Qwen3-VL 使用文档，覆盖 FP8/BF16 launch、image/video request、`--mm-attention-backend`、`--mm-max-concurrent-calls`、`--keep-mm-feature-on-device` 与 CUDA IPC。
- 关键命令：

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tp 8 \
  --ep 8 \
  --keep-mm-feature-on-device
```

### #12703 Qwen3-Omni 官方文档

- 链接：https://github.com/sgl-project/sglang/pull/12703，open。
- 内容：补 Qwen3-Omni launch 和 image/audio/video OpenAI 请求样例。
- 注意：未合入，不能当作 main 文档事实。

### sgl-cookbook 与公开博客

- sgl-cookbook `#76/#102/#124`：Qwen3-VL 在 AMD MI300X/MI355X/MI325X 的部署配置。
- sgl-cookbook `#84/#110`：Qwen2.5-VL 在 AMD MI300X/MI355X/MI325X 的部署配置。
- LMSYS AMD latency blog：Qwen3-VL-235B 基于 SGLang 在 MI300X 上做延迟优化，报告 TTFT 1.62x、TPOT 1.90x。
- SGLang `#18466`：把 AMD Qwen3-VL 优化拆成 preprocessing、multimodal transfer、ViT DP、ViT kernel fusion 等类目。

## Open / Closed 雷达

- `#12662` CPU Qwen3-VL/Qwen3-Omni：CPU 下切到 SDPA，processor device 走 CPU，并处理 unaligned CPU TP padding。
- `#12261` Qwen2.5-VL cu_seqlens：修 multi-frame/multi-patch 的 `cu_seqlens`。
- `#13918` / `#17276` Qwen3-VL EAGLE3 早期方案：重点是 mRoPE interleaving 与 DeepStack 后捕获层。
- `#14886` Qwen3-Omni DP encoder：把 Qwen3-VL vision DP 思路扩展到 Omni audio/vision tower。
- `#16491` Qwen3-VL-MoE PP expert weight skip：PP rank 上不存在的 expert 权重要跳过。
- `#16571` ROCm Qwen3-VL ViT add+LayerNorm fusion：AITER fused layernorm 路径。
- `#16785` Qwen3-VL DeepStack recompile：预分配 `input_deepstack_embeds`，让 text-only / multimodal 混流不反复触发 TorchDynamo recompile。
- `#16996` Qwen3-Omni `use_audio_in_video`：视频源内音频要作为 audio item 加入。
- `#17202` Qwen3-VL remove CPU/device ops：移除 vision attention `.contiguous()`，用 `masked_scatter_` 写媒体 embedding。
- `#18721` Qwen3-VL DP encoder hang follow-up：与 `#20759` 重叠。
- `#18771` Qwen3-Omni fused MoE tuner：把 Omni 架构加入 Qwen MoE benchmark/tuner 列表。
- `#19242` 早期 Qwen3-ASR 支持：被 `#22073` 的完整实现取代。
- `#19693` NPU Qwen3-VL-8B accuracy：NPU RoPE、embedding compile、QKV RMSNorm/RoPE split。
- `#20857` Qwen3-VL EVS：视频 token pruning 后 mRoPE 只推进实际 token 数。
- `#22052` Qwen3-VL precise embedding interpolation：默认 precise，避免 HF 对齐误差。
- `#22839` Qwen3-VL config `from_dict`：Transformers 5.5.0+ 嵌套 config dict 兼容。
- `#22848` Qwen3-ASR WebSocket realtime input：`/v1/audio/transcriptions/stream` 接收 PCM16 frame 并输出 delta。
- `#23115` / `#23220` Qwen3-VL-MoE encoder-only guard：同一行 `hasattr(self, "model")` 修复。
- `#23304` Qwen3-VL RoPE config compatibility：closed unmerged，只记录兼容性风险。
- `#23469` NPU Qwen3-ASR audio loading：NPU 下用 `soundfile` + `resample_poly` 替代 torchaudio CUDA 依赖。

## 验证建议

1. 单图、多图、视频、processor_output 视频分别测。
2. Qwen3-VL 的 encoder DP、PP、EPD 要单独测，不要只测普通 launch。
3. 长视频必须测 chunked prefill、cache hit/miss、feature transfer。
4. Qwen3-Omni 音频相关改动必须测 audio-only、video+audio、feature_attention_mask。
5. Qwen3-ASR 必须测最终转写、HTTP streaming chunk 边界、WebSocket realtime input（如果启用）。
6. AMD/NPU/CPU 不能互相推断；硬件相关 PR 必须在对应硬件 lane 验证。
