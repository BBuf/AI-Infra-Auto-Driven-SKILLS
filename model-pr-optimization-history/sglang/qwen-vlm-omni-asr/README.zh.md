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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Qwen VLM / Omni / ASR`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-25 | [#10911](https://github.com/sgl-project/sglang/pull/10911) | merged | model: qwen3-omni (thinker-only) | model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-09-27 | [#10985](https://github.com/sgl-project/sglang/pull/10985) | merged | Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-10-28 | [#12261](https://github.com/sgl-project/sglang/pull/12261) | open | [BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl | model wrapper | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-10-29 | [#12333](https://github.com/sgl-project/sglang/pull/12333) | merged | [PP] Add pp support for Qwen3-VL | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_omni_moe.py` |
| 2025-11-03 | [#12554](https://github.com/sgl-project/sglang/pull/12554) | merged | [Docs] Add docs for Qwen3-VL image and video support | docs/config | `docs/basic_usage/qwen3_vl.md`, `docs/index.rst` |
| 2025-11-05 | [#12662](https://github.com/sgl-project/sglang/pull/12662) | open | [CPU] Add support for Qwen3-vl and Qwen3-omni | model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, docs/config | `sgl-kernel/csrc/cpu/gemm.cpp`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py` |
| 2025-11-05 | [#12703](https://github.com/sgl-project/sglang/pull/12703) | open | add qwen3-omni docs | docs/config | `docs/basic_usage/qwen3_omni.md`, `docs/index.rst` |
| 2025-11-21 | [#13724](https://github.com/sgl-project/sglang/pull/13724) | merged | support qwen3_vl vision model dp | model wrapper, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py` |
| 2025-11-21 | [#13736](https://github.com/sgl-project/sglang/pull/13736) | merged | [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series | model wrapper, tests/benchmarks | `test/srt/ops/test_repeat_interleave.py`, `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-11-25 | [#13918](https://github.com/sgl-project/sglang/pull/13918) | open | [VLM] support qwen3-vl eagle infer | model wrapper | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py` |
| 2025-12-02 | [#14292](https://github.com/sgl-project/sglang/pull/14292) | merged | [VLM] Introduce Cache for positional embedding ids for Qwen-VL family | model wrapper | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-11 | [#14886](https://github.com/sgl-project/sglang/pull/14886) | open | Support qwen3-omni with DP Encoder | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py` |
| 2025-12-11 | [#14907](https://github.com/sgl-project/sglang/pull/14907) | merged | [VLM] Support chunked vit attention | model wrapper | `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-12-15 | [#15205](https://github.com/sgl-project/sglang/pull/15205) | merged | [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V | model wrapper, attention/backend, multimodal/processor | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2025-12-17 | [#15320](https://github.com/sgl-project/sglang/pull/15320) | merged | [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL | model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks | `python/sglang/srt/multimodal/vit_cuda_graph_runner.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2026-01-04 | [#16366](https://github.com/sgl-project/sglang/pull/16366) | merged | Optimize Qwen3-VL video memory usage | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-01-05 | [#16491](https://github.com/sgl-project/sglang/pull/16491) | open | [Qwen3-VL][PP] Skip loading expert weights not on this rank | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-01-06 | [#16571](https://github.com/sgl-project/sglang/pull/16571) | open | [Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT | model wrapper | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py` |
| 2026-01-09 | [#16785](https://github.com/sgl-project/sglang/pull/16785) | open | [Bugfix] fix recompile in qwen3 vl | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-01-13 | [#16996](https://github.com/sgl-project/sglang/pull/16996) | open | feat: Support 'use_audio_in_video' option for qwen3omnimoe model | multimodal/processor | `python/sglang/srt/utils/common.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-01-16 | [#17202](https://github.com/sgl-project/sglang/pull/17202) | open | [Feat] Accelerate qwen3vl by remove cpu op | attention/backend, multimodal/processor | `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2026-01-18 | [#17276](https://github.com/sgl-project/sglang/pull/17276) | open | Add Qwen3VL Eagle3 Inference Support | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-01-23 | [#17624](https://github.com/sgl-project/sglang/pull/17624) | merged | [BUGFIX] Fix dp size > 1 for qwen3 vl model | model wrapper, attention/backend, multimodal/processor, scheduler/runtime | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py` |
| 2026-01-31 | [#18024](https://github.com/sgl-project/sglang/pull/18024) | merged | fix: correct weight loading prefix mapping for Qwen3-VL | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-03 | [#18185](https://github.com/sgl-project/sglang/pull/18185) | merged | [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_omni_moe.py` |
| 2026-02-12 | [#18721](https://github.com/sgl-project/sglang/pull/18721) | open | [BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models | model wrapper | `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-13 | [#18771](https://github.com/sgl-project/sglang/pull/18771) | open | Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton | MoE/router, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-02-19 | [#19003](https://github.com/sgl-project/sglang/pull/19003) | merged | [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend | model wrapper, attention/backend, multimodal/processor, tests/benchmarks | `python/sglang/srt/models/qwen3_vl.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2026-02-24 | [#19242](https://github.com/sgl-project/sglang/pull/19242) | open | [feat] feat: add Qwen3-ASR support like whisper | multimodal/processor, docs/config | `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py` |
| 2026-02-25 | [#19291](https://github.com/sgl-project/sglang/pull/19291) | merged | [Qwen3.5] Fix missing `quant_config` in `Qwen3VL` | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-02-25 | [#19333](https://github.com/sgl-project/sglang/pull/19333) | merged | fix qwen3_vl visual module loading | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-02 | [#19693](https://github.com/sgl-project/sglang/pull/19693) | open | [NPU] Fix Qwen3-VL-8B Accuracy for NPU | model wrapper, attention/backend, MoE/router, scheduler/runtime | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/models/llama.py` |
| 2026-03-17 | [#20759](https://github.com/sgl-project/sglang/pull/20759) | merged | [Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-17 | [#20788](https://github.com/sgl-project/sglang/pull/20788) | merged | [DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-18 | [#20857](https://github.com/sgl-project/sglang/pull/20857) | open | add EVS support for Qwen3-VL | model wrapper, multimodal/processor, docs/config | `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-03-26 | [#21458](https://github.com/sgl-project/sglang/pull/21458) | merged | [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-03-26 | [#21469](https://github.com/sgl-project/sglang/pull/21469) | merged | [3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct | model wrapper, MoE/router, tests/benchmarks | `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`, `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-01 | [#21849](https://github.com/sgl-project/sglang/pull/21849) | merged | [VLM]: allow Qwen3.5 models for encoder disaggregation | multimodal/processor, tests/benchmarks | `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-03 | [#22038](https://github.com/sgl-project/sglang/pull/22038) | merged | [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer | model wrapper, multimodal/processor, scheduler/runtime | `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-04-03 | [#22052](https://github.com/sgl-project/sglang/pull/22052) | open | [Fix] Enable precise embedding interpolation by default for Qwen3-VL | model wrapper, docs/config | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` |
| 2026-04-03 | [#22073](https://github.com/sgl-project/sglang/pull/22073) | merged | [Feature] Adding Qwen3-asr Model Support | model wrapper, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` |
| 2026-04-04 | [#22089](https://github.com/sgl-project/sglang/pull/22089) | merged | [Feature] Add chunk-based streaming ASR for Qwen3-ASR | multimodal/processor | `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py` |
| 2026-04-07 | [#22230](https://github.com/sgl-project/sglang/pull/22230) | merged | [Feature] Support eagle3 for qwen3-vl | model wrapper | `python/sglang/srt/models/qwen3_vl.py` |
| 2026-04-07 | [#22266](https://github.com/sgl-project/sglang/pull/22266) | merged | [NPU] fix qwen3.5 video processor | multimodal/processor | `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` |
| 2026-04-09 | [#22431](https://github.com/sgl-project/sglang/pull/22431) | merged | Fix Qwen3.5 video processing when passing video_data in "processor_output" format | multimodal/processor | `python/sglang/srt/multimodal/processors/qwen_vl.py` |
| 2026-04-15 | [#22839](https://github.com/sgl-project/sglang/pull/22839) | open | fix(config): Add from_dict() for Qwen3VL config classes | tests/benchmarks, docs/config | `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py` |
| 2026-04-15 | [#22848](https://github.com/sgl-project/sglang/pull/22848) | open | [Feature] WebSocket streaming audio input for ASR | model wrapper, tests/benchmarks | `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py` |
| 2026-04-18 | [#23115](https://github.com/sgl-project/sglang/pull/23115) | open | fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-20 | [#23220](https://github.com/sgl-project/sglang/pull/23220) | open | Bugfix: Qwen3-VL-MoE adapt encoder_only | model wrapper, MoE/router | `python/sglang/srt/models/qwen3_vl_moe.py` |
| 2026-04-21 | [#23304](https://github.com/sgl-project/sglang/pull/23304) | closed | [Bugfix] Fix Qwen3-VL rope config compatibility | model wrapper | `python/sglang/srt/models/qwen3.py` |
| 2026-04-22 | [#23469](https://github.com/sgl-project/sglang/pull/23469) | open | [NPU] adapt the Qwen3-ASR model for deployment on NPU | misc | `python/sglang/srt/utils/common.py` |

### 逐 PR 代码 diff 阅读记录

### PR #10911 - model: qwen3-omni (thinker-only)

- 链接：https://github.com/sgl-project/sglang/pull/10911
- 状态/时间：`merged`，created 2025-09-25, merged 2025-10-16；作者 `mickqian`。
- 代码 diff 已读范围：`16` 个文件，`+1947/-328`；代码面：model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config；关键词：vision, attention, moe, config, cache, quant, expert, processor, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_omni_moe.py` added +661/-0 (661 lines); hunk: +# Copyright 2025 Qwen Team; 符号: Qwen3OmniMoeAudioEncoderLayer, __init__, forward, SinusoidsPositionEmbedding
  - `python/sglang/srt/configs/qwen3_omni.py` added +613/-0 (613 lines); hunk: +from transformers import PretrainedConfig; 符号: Qwen3OmniMoeAudioEncoderConfig, __init__, Qwen3OmniMoeVisionEncoderConfig, __init__
  - `python/sglang/srt/layers/rotary_embedding.py` modified +357/-2 (359 lines); hunk: def get_rope_index(; def get_rope_index(; 符号: get_rope_index, get_rope_index, get_rope_index, get_rope_index_qwen3_omni
  - `test/srt/test_vision_openai_server_common.py` modified +132/-96 (228 lines); hunk: import base64; AUDIO_BIRD_SONG_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3"; 符号: TestOpenAIOmniServerBase, TestOpenAIMLLMServerBase, setUpClass, get_or_download_file
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +53/-168 (221 lines); hunk: # ==============================================================================; class Qwen3MoeLLMModel(Qwen3MoeModel):; 符号: Qwen3MoeLLMModel, __init__, get_input_embeddings, get_image_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 vision, attention, moe, config, cache, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_omni_moe.py`, `python/sglang/srt/configs/qwen3_omni.py`, `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10985 - Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg

- 链接：https://github.com/sgl-project/sglang/pull/10985
- 状态/时间：`merged`，created 2025-09-27, merged 2025-10-01；作者 `yhyang201`。
- 代码 diff 已读范围：`2` 个文件，`+14/-2`；代码面：model wrapper, MoE/router；关键词：cache, kv, attention, config, moe, quant, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunk: from sglang.srt.layers.moe.topk import TopK; def __init__(; 符号: __init__, forward_prepare, forward_core
  - `python/sglang/srt/layers/rotary_embedding.py` modified +4/-0 (4 lines); hunk: def forward(; def forward(; 符号: forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 cache, kv, attention, config, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12261 - [BugFix][Qwen2.5-VL]: fix cu_seqlens in qwen2.5-vl

- 链接：https://github.com/sgl-project/sglang/pull/12261
- 状态/时间：`open`，created 2025-10-28；作者 `gjghfd`。
- 代码 diff 已读范围：`1` 个文件，`+5/-5`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-5 (10 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_5_vl.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_5_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12333 - [PP] Add pp support for Qwen3-VL

- 链接：https://github.com/sgl-project/sglang/pull/12333
- 状态/时间：`merged`，created 2025-10-29, merged 2025-12-17；作者 `XucSh`。
- 代码 diff 已读范围：`5` 个文件，`+119/-20`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：mla, test, attention, config, expert, fp4, moe, processor, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +53/-19 (72 lines); hunk: get_tensor_model_parallel_rank,; def __init__(; 符号: __init__, __init__, forward, forward
  - `test/srt/test_pp_single_node.py` modified +57/-0 (57 lines); hunk: python3 -m unittest test_pp_single_node.TestPPAccuracy.test_gsm8k; DEFAULT_MLA_MODEL_NAME_FOR_TEST,; 符号: test_mgsm_en, TestQwenVLPPAccuracy, setUpClass, test_gsm8k
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +4/-1 (5 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
  - `python/sglang/srt/managers/schedule_policy.py` modified +4/-0 (4 lines); hunk: def _update_prefill_budget(; 符号: _update_prefill_budget, add_chunked_req
  - `python/sglang/test/test_utils.py` modified +1/-0 (1 lines); hunk: DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN = "lmsys/sglang-ci-dsv3-test-NextN"
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_omni_moe.py`；patch 关键词为 mla, test, attention, config, expert, fp4。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/qwen3_omni_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12554 - [Docs] Add docs for Qwen3-VL image and video support

- 链接：https://github.com/sgl-project/sglang/pull/12554
- 状态/时间：`merged`，created 2025-11-03, merged 2025-11-10；作者 `adarshxs`。
- 代码 diff 已读范围：`2` 个文件，`+131/-0`；代码面：docs/config；关键词：doc, attention, cache, cuda, flash, fp8, quant, spec, test, vision。
- 代码 diff 细节：
  - `docs/basic_usage/qwen3_vl.md` added +130/-0 (130 lines); hunk: +# Qwen3-VL Usage
  - `docs/index.rst` modified +1/-0 (1 lines); hunk: Its core features include:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/qwen3_vl.md`, `docs/index.rst`；patch 关键词为 doc, attention, cache, cuda, flash, fp8。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/qwen3_vl.md`, `docs/index.rst` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12662 - [CPU] Add support for Qwen3-vl and Qwen3-omni

- 链接：https://github.com/sgl-project/sglang/pull/12662
- 状态/时间：`open`，created 2025-11-05；作者 `blzheng`。
- 代码 diff 已读范围：`12` 个文件，`+496/-55`；代码面：model wrapper, attention/backend, MoE/router, kernel, multimodal/processor, docs/config；关键词：attention, config, vision, kv, quant, cuda, expert, flash, moe, processor。
- 代码 diff 细节：
  - `sgl-kernel/csrc/cpu/gemm.cpp` modified +142/-0 (142 lines); hunk: void weight_packed_linear_kernel_impl(; at::Tensor fused_linear_sigmoid_mul(; 符号: int64_t, int64_t, int64_t
  - `python/sglang/srt/layers/attention/vision.py` modified +80/-12 (92 lines); hunk: from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size; with_multi_stream,; 符号: forward, VisionAMXAttention, __init__, forward
  - `python/sglang/srt/configs/update_config.py` modified +54/-20 (74 lines); hunk: def adjust_config_with_unaligned_cpu_tp(; 符号: adjust_config_with_unaligned_cpu_tp
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +60/-0 (60 lines); hunk: from sglang.srt.multimodal.processors.base_processor import (; FPS_MAX_FRAMES = 768; 符号: hacked_preprocess, smart_resize
  - `python/sglang/srt/models/qwen3_vl.py` modified +44/-6 (50 lines); hunk: from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; logger = logging.getLogger(__name__); 符号: Qwen3_VisionMLP, __init__, __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/cpu/gemm.cpp`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py`；patch 关键词为 attention, config, vision, kv, quant, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/cpu/gemm.cpp`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/configs/update_config.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12703 - add qwen3-omni docs

- 链接：https://github.com/sgl-project/sglang/pull/12703
- 状态/时间：`open`，created 2025-11-05；作者 `jiapingW`。
- 代码 diff 已读范围：`2` 个文件，`+150/-0`；代码面：docs/config；关键词：doc, spec, test。
- 代码 diff 细节：
  - `docs/basic_usage/qwen3_omni.md` added +149/-0 (149 lines); hunk: +# Qwen3-Omni Usage
  - `docs/index.rst` modified +1/-0 (1 lines); hunk: Its core features include:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/qwen3_omni.md`, `docs/index.rst`；patch 关键词为 doc, spec, test。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/qwen3_omni.md`, `docs/index.rst` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13724 - support qwen3_vl vision model dp

- 链接：https://github.com/sgl-project/sglang/pull/13724
- 状态/时间：`merged`，created 2025-11-21, merged 2025-11-28；作者 `Lzhang-hub`。
- 代码 diff 已读范围：`2` 个文件，`+50/-2`；代码面：model wrapper, tests/benchmarks；关键词：test, attention, awq, config, moe, processor, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +49/-2 (51 lines); hunk: ); from sglang.srt.model_loader.weight_utils import default_weight_loader; 符号: __init__, __init__, __init__, __init__
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunk: MODELS = [
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py`；patch 关键词为 test, attention, awq, config, moe, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `test/nightly/test_encoder_dp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13736 - [VLM] Replace torch.repeat_interleave with faster np.repeat for Qwen-VL series

- 链接：https://github.com/sgl-project/sglang/pull/13736
- 状态/时间：`merged`，created 2025-11-21, merged 2025-11-22；作者 `yuan-luo`。
- 代码 diff 已读范围：`5` 个文件，`+169/-13`；代码面：model wrapper, tests/benchmarks；关键词：processor, test, attention, benchmark, config, fp8, quant。
- 代码 diff 细节：
  - `test/srt/ops/test_repeat_interleave.py` added +141/-0 (141 lines); hunk: +import time; 符号: torch_ref_impl, benchmark_once, _generate_random_grid, TestRepeatInterleave:
  - `python/sglang/srt/models/utils.py` modified +23/-0 (23 lines); hunk: # limitations under the License.; def permute_inv(perm: torch.Tensor) -> torch.Tensor:; 符号: permute_inv, compute_cu_seqlens_from_grid_numpy
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-9 (11 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors; def forward(; 符号: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +2/-4 (6 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def forward(; 符号: forward
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunk: TestFile("openai_server/validation/test_matched_stop.py", 60),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/ops/test_repeat_interleave.py`, `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 processor, test, attention, benchmark, config, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/ops/test_repeat_interleave.py`, `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13918 - [VLM] support qwen3-vl eagle infer

- 链接：https://github.com/sgl-project/sglang/pull/13918
- 状态/时间：`open`，created 2025-11-25；作者 `Lzhang-hub`。
- 代码 diff 已读范围：`2` 个文件，`+30/-3`；代码面：model wrapper；关键词：config, eagle, processor, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +23/-1 (24 lines); hunk: def __init__(; def forward(; 符号: __init__, separate_deepstack_embeds, forward, load_weights
  - `python/sglang/srt/models/llama_eagle3.py` modified +7/-2 (9 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py`；patch 关键词为 config, eagle, processor, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/llama_eagle3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14292 - [VLM] Introduce Cache for positional embedding ids for Qwen-VL family

- 链接：https://github.com/sgl-project/sglang/pull/14292
- 状态/时间：`merged`，created 2025-12-02, merged 2025-12-04；作者 `yuan-luo`。
- 代码 diff 已读范围：`3` 个文件，`+48/-47`；代码面：model wrapper；关键词：vision, cache, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/utils.py` modified +38/-0 (38 lines); hunk: # limitations under the License.; def compute_cu_seqlens_from_grid_numpy(grid_thw: torch.Tensor) -> torch.Tensor:; 符号: compute_cu_seqlens_from_grid_numpy, RotaryPosMixin:, rot_pos_ids
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +5/-25 (30 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors; def forward(self, x: torch.Tensor) -> torch.Tensor:; 符号: forward, Qwen2_5_VisionTransformer, Qwen2_5_VisionTransformer, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +5/-22 (27 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors; def forward(self, x: torch.Tensor) -> torch.Tensor:; 符号: forward, Qwen3VLMoeVisionModel, Qwen3VLMoeVisionModel, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 vision, cache, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14886 - Support qwen3-omni with DP Encoder

- 链接：https://github.com/sgl-project/sglang/pull/14886
- 状态/时间：`open`，created 2025-12-11；作者 `apinge`。
- 代码 diff 已读范围：`2` 个文件，`+32/-3`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：attention, config, cuda, expert, moe, quant, test, triton, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +31/-3 (34 lines); hunk: Qwen3OmniMoeVisionEncoderConfig,; Qwen3VLMoeForConditionalGeneration,; 符号: __init__, __init__, _get_feat_extract_output_lengths, Qwen3OmniMoeAudioEncoder
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunk: register_cuda_ci(est_time=500, suite="nightly-4-gpu", nightly=True)
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py`；patch 关键词为 attention, config, cuda, expert, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_omni_moe.py`, `test/nightly/test_encoder_dp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14907 - [VLM] Support chunked vit attention

- 链接：https://github.com/sgl-project/sglang/pull/14907
- 状态/时间：`merged`，created 2025-12-11, merged 2025-12-15；作者 `yuan-luo`。
- 代码 diff 已读范围：`2` 个文件，`+363/-8`；代码面：model wrapper；关键词：cache, eagle, processor, vision。
- 代码 diff 细节：
  - `python/sglang/srt/managers/mm_utils.py` modified +266/-0 (266 lines); hunk: _GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None; def _get_precomputed_embedding(; 符号: init_feature_buffer, _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _get_chunked_prefill_embedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +97/-8 (105 lines); hunk: # ==============================================================================; from sglang.srt.models.utils import RotaryPosMixin, compute_cu_seqlens_from_gr; 符号: get_image_feature, get_video_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 cache, eagle, processor, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- 链接：https://github.com/sgl-project/sglang/pull/15205
- 状态/时间：`merged`，created 2025-12-15, merged 2025-12-18；作者 `yuan-luo`。
- 代码 diff 已读范围：`4` 个文件，`+100/-80`；代码面：model wrapper, attention/backend, multimodal/processor；关键词：cache, vision, config, processor, quant, attention。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunk: from sglang.srt.layers.logits_processor import LogitsProcessor; def forward(; 符号: forward, forward, forward, Glm4vVisionRotaryEmbedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunk: import torch.nn as nn; from sglang.srt.layers.logits_processor import LogitsProcessor; 符号: forward, __init__, dtype, device
  - `python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunk: def forward(; def forward(; 符号: forward, forward
  - `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunk: def get_cos_sin_with_position(self, positions):; 符号: get_cos_sin_with_position, get_cos_sin, forward_native
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`；patch 关键词为 cache, vision, config, processor, quant, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15320 - [VLM] Support ViT Piecewise CUDA Graph for Qwen3-VL

- 链接：https://github.com/sgl-project/sglang/pull/15320
- 状态/时间：`merged`，created 2025-12-17, merged 2025-12-20；作者 `yuan-luo`。
- 代码 diff 已读范围：`6` 个文件，`+233/-64`；代码面：model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks；关键词：cuda, vision, attention, cache, flash, processor, spec, test, triton。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/vit_cuda_graph_runner.py` modified +175/-57 (232 lines); hunk: from __future__ import annotations; class ViTCudaGraphRunner:; 符号: ViTCudaGraphRunner:, __init__, __init__, _get_graph_key
  - `python/sglang/srt/models/qwen3_vl.py` modified +52/-1 (53 lines); hunk: compute_cu_seqlens_from_grid_numpy,; def forward(; 符号: forward, forward, __init__, dtype
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +2/-3 (5 lines); hunk: def forward(; def forward(; 符号: forward, forward, forward
  - `python/sglang/srt/layers/attention/vision.py` modified +2/-2 (4 lines); hunk: def forward(; def forward(; 符号: forward, forward
  - `python/sglang/srt/distributed/parallel_state.py` modified +1/-1 (2 lines); hunk: def _all_reduce_out_place(; 符号: _all_reduce_out_place
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/vit_cuda_graph_runner.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py`；patch 关键词为 cuda, vision, attention, cache, flash, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/vit_cuda_graph_runner.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen2_5_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16366 - Optimize Qwen3-VL video memory usage

- 链接：https://github.com/sgl-project/sglang/pull/16366
- 状态/时间：`merged`，created 2026-01-04, merged 2026-01-22；作者 `cen121212`。
- 代码 diff 已读范围：`1` 个文件，`+8/-0`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +8/-0 (8 lines); hunk: def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:; 符号: get_image_feature, get_video_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16491 - [Qwen3-VL][PP] Skip loading expert weights not on this rank

- 链接：https://github.com/sgl-project/sglang/pull/16491
- 状态/时间：`open`，created 2026-01-05；作者 `MtFitzRoy`。
- 代码 diff 已读范围：`1` 个文件，`+3/-0`；代码面：model wrapper, MoE/router；关键词：expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +3/-0 (3 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl_moe.py`；patch 关键词为 expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16571 - [Feature] [ROCM] Support Add & LayerNorm fused for Qwen3-VL VIT

- 链接：https://github.com/sgl-project/sglang/pull/16571
- 状态/时间：`open`，created 2026-01-06；作者 `qichu-yun`。
- 代码 diff 已读范围：`2` 个文件，`+87/-15`；代码面：model wrapper；关键词：cuda, attention, moe, processor, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +60/-15 (75 lines); hunk: get_attention_tp_size,; from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; 符号: Qwen3_VisionMLP, __init__, forward, forward
  - `python/sglang/srt/layers/layernorm.py` modified +27/-0 (27 lines); hunk: ); def __init__(; 符号: __init__, forward_cuda, forward_cpu, forward_aiter
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py`；patch 关键词为 cuda, attention, moe, processor, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/layernorm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16785 - [Bugfix] fix recompile in qwen3 vl

- 链接：https://github.com/sgl-project/sglang/pull/16785
- 状态/时间：`open`，created 2026-01-09；作者 `narutolhy`。
- 代码 diff 已读范围：`5` 个文件，`+113/-36`；代码面：model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks；关键词：config, cache, cuda, attention, deepep, mla, moe, test, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-18 (59 lines); hunk: def __init__(; def forward(; 符号: __init__, get_deepstack_embeds, forward, forward
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +29/-8 (37 lines); hunk: def __init__(; def forward(; 符号: __init__, get_input_embeddings, get_deepstack_embeds, forward
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +25/-1 (26 lines); hunk: def __init__(self, model_runner: ModelRunner):; def warmup_torch_compile(self, num_tokens: int):; 符号: __init__, warmup_torch_compile, warmup_torch_compile, _cache_loc_dtype
  - `python/sglang/srt/managers/mm_utils.py` modified +13/-6 (19 lines); hunk: def embed_mm_inputs(; def embed_mm_inputs(; 符号: embed_mm_inputs, embed_mm_inputs, general_mm_embed_routine, general_mm_embed_routine
  - `test/manual/nightly/test_vlms_piecewise_cuda_graph.py` modified +5/-3 (8 lines); hunk: ); def run_mmmu_eval(; 符号: run_mmmu_eval, _run_vlm_mmmu_test
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；patch 关键词为 config, cache, cuda, attention, deepep, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/qwen3_vl_moe.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16996 - feat: Support 'use_audio_in_video' option for qwen3omnimoe model

- 链接：https://github.com/sgl-project/sglang/pull/16996
- 状态/时间：`open`，created 2026-01-13；作者 `srLi24`。
- 代码 diff 已读范围：`6` 个文件，`+129/-12`；代码面：multimodal/processor；关键词：moe, processor, config, spec, test, vision。
- 代码 diff 细节：
  - `python/sglang/srt/utils/common.py` modified +63/-7 (70 lines); hunk: from unittest import SkipTest; def load_audio(; 符号: load_audio, extract_audio_via_av, ImageData:, get_image_bytes
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +34/-3 (37 lines); hunk: def process_mm_data(; def _load_single_item(; 符号: process_mm_data, _load_single_item, _load_single_item, submit_data_loading_tasks
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +25/-2 (27 lines); hunk: async def preprocess_video(; async def preprocess_video(; 符号: preprocess_video, preprocess_video, __init__, process_mm_data_async
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +3/-0 (3 lines); hunk: class ChatCompletionRequest(BaseModel):; def get_param(param_name: str):; 符号: ChatCompletionRequest, get_param
  - `python/sglang/srt/managers/io_struct.py` modified +3/-0 (3 lines); hunk: class GenerateReqInput(BaseReq, APIServingTimingMixin):; 符号: GenerateReqInput, contains_mm_input
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/utils/common.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`；patch 关键词为 moe, processor, config, spec, test, vision。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/utils/common.py`, `python/sglang/srt/multimodal/processors/base_processor.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17202 - [Feat] Accelerate qwen3vl by remove cpu op

- 链接：https://github.com/sgl-project/sglang/pull/17202
- 状态/时间：`open`，created 2026-01-16；作者 `ZLkanyo009`。
- 代码 diff 已读范围：`2` 个文件，`+27/-9`；代码面：attention/backend, multimodal/processor；关键词：attention, kv, vision。
- 代码 diff 细节：
  - `python/sglang/srt/managers/mm_utils.py` modified +24/-6 (30 lines); hunk: def embed_mm_inputs(; 符号: embed_mm_inputs
  - `python/sglang/srt/layers/attention/vision.py` modified +3/-3 (6 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/layers/attention/vision.py`；patch 关键词为 attention, kv, vision。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/layers/attention/vision.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17276 - Add Qwen3VL Eagle3 Inference Support

- 链接：https://github.com/sgl-project/sglang/pull/17276
- 状态/时间：`open`，created 2026-01-18；作者 `ardenma`。
- 代码 diff 已读范围：`1` 个文件，`+35/-0`；代码面：model wrapper；关键词：config, eagle, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +35/-0 (35 lines); hunk: def __init__(; def forward(; 符号: __init__, separate_deepstack_embeds, forward, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 config, eagle, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17624 - [BUGFIX] Fix dp size > 1 for qwen3 vl model

- 链接：https://github.com/sgl-project/sglang/pull/17624
- 状态/时间：`merged`，created 2026-01-23, merged 2026-01-30；作者 `zju-stu-lizheng`。
- 代码 diff 已读范围：`5` 个文件，`+48/-19`；代码面：model wrapper, attention/backend, multimodal/processor, scheduler/runtime；关键词：attention, vision, config, quant, cuda, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +14/-13 (27 lines); hunk: from transformers.activations import ACT2FN; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/multimodal/mm_utils.py` modified +13/-3 (16 lines); hunk: def run_dp_sharded_mrope_vision_model(; def run_dp_sharded_mrope_vision_model(; 符号: run_dp_sharded_mrope_vision_model, run_dp_sharded_mrope_vision_model
  - `python/sglang/srt/layers/linear.py` modified +10/-2 (12 lines); hunk: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; def __init__(; 符号: __init__, __init__, forward
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +9/-1 (10 lines); hunk: def _pad_inputs_to_size(self, model_runner: ModelRunner, num_tokens, bs):; 符号: _pad_inputs_to_size
  - `python/sglang/srt/layers/attention/vision.py` modified +2/-0 (2 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py`；patch 关键词为 attention, vision, config, quant, cuda, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/multimodal/mm_utils.py`, `python/sglang/srt/layers/linear.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18024 - fix: correct weight loading prefix mapping for Qwen3-VL

- 链接：https://github.com/sgl-project/sglang/pull/18024
- 状态/时间：`merged`，created 2026-01-31, merged 2026-02-02；作者 `Lollipop`。
- 代码 diff 已读范围：`1` 个文件，`+7/-1`；代码面：model wrapper；关键词：config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +7/-1 (8 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18185 - [Omni] Optimize AudioEncoder for Qwen3_Omni_Thinker

- 链接：https://github.com/sgl-project/sglang/pull/18185
- 状态/时间：`merged`，created 2026-02-03, merged 2026-03-14；作者 `yuan-luo`。
- 代码 diff 已读范围：`1` 个文件，`+52/-28`；代码面：model wrapper, MoE/router；关键词：attention, config, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_omni_moe.py` modified +52/-28 (80 lines); hunk: def __init__(; def forward(; 符号: __init__, forward, forward, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_omni_moe.py`；patch 关键词为 attention, config, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_omni_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18721 - [BUG] fix mm_enable_dp_encoder hang for Qwen3-VL models

- 链接：https://github.com/sgl-project/sglang/pull/18721
- 状态/时间：`open`，created 2026-02-12；作者 `kousakawang`。
- 代码 diff 已读范围：`2` 个文件，`+3/-1`；代码面：model wrapper；关键词：attention, config, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +2/-1 (3 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 attention, config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18771 - Add Qwen3-Omni to Qwen MoE architecture handling in fused_moe_triton

- 链接：https://github.com/sgl-project/sglang/pull/18771
- 状态/时间：`open`，created 2026-02-13；作者 `AwesomeKeyboard`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：MoE/router, kernel, tests/benchmarks；关键词：benchmark, config, expert, moe, topk, triton。
- 代码 diff 细节：
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +1/-0 (1 lines); hunk: def get_model_config(; 符号: get_model_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/fused_moe_triton/common_utils.py`；patch 关键词为 benchmark, config, expert, moe, topk, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/fused_moe_triton/common_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19003 - [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend

- 链接：https://github.com/sgl-project/sglang/pull/19003
- 状态/时间：`merged`，created 2026-02-19, merged 2026-02-24；作者 `yuan-luo`。
- 代码 diff 已读范围：`4` 个文件，`+678/-14`；代码面：model wrapper, attention/backend, multimodal/processor, tests/benchmarks；关键词：attention, flash, cuda, cache, test, vision, benchmark, config, kv, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +259/-13 (272 lines); hunk: from sglang.srt.distributed import get_tensor_model_parallel_world_size; from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; 符号: Qwen3_VisionMLP, __init__, __init__, __init__
  - `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py` added +258/-0 (258 lines); hunk: +import argparse; 符号: TestVLMViTFlashinferCudnn, setUpClass, run_mmmu_eval, _run_vlm_mmmu_test
  - `python/sglang/srt/layers/attention/vision.py` modified +152/-0 (152 lines); hunk: _is_hip = is_hip(); "normal": apply_rotary_pos_emb,; 符号: SingletonCache:, forward, VisionFlashInferAttention, __init__
  - `python/sglang/srt/server_args.py` modified +9/-1 (10 lines); hunk: def add_cli_args(parser: argparse.ArgumentParser):; 符号: add_cli_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`, `python/sglang/srt/layers/attention/vision.py`；patch 关键词为 attention, flash, cuda, cache, test, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py`, `python/sglang/srt/layers/attention/vision.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19242 - [feat] feat: add Qwen3-ASR support like whisper

- 链接：https://github.com/sgl-project/sglang/pull/19242
- 状态/时间：`open`，created 2026-02-24；作者 `LuYanFCP`。
- 代码 diff 已读范围：`5` 个文件，`+475/-0`；代码面：multimodal/processor, docs/config；关键词：config, attention, moe, processor, cache, spec, vision。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +252/-0 (252 lines); hunk: +import logging; 符号: Qwen3ASRMultimodalProcessor, __init__, _get_feature_extractor, _compute_audio_output_length
  - `python/sglang/srt/configs/qwen3_asr.py` added +217/-0 (217 lines); hunk: +"""; 符号: Qwen3ASRHFProcessor, __init__, from_pretrained, Qwen3ASRAudioEncoderConfig
  - `python/sglang/srt/configs/__init__.py` modified +2/-0 (2 lines); hunk: from sglang.srt.configs.nemotron_h import NemotronHConfig; "Olmo3Config",
  - `python/sglang/srt/configs/model_config.py` modified +2/-0 (2 lines); hunk: def is_generation_model(model_architectures: List[str], is_embedding: bool = Fal; def is_image_gen_model(model_architectures: List[str]):; 符号: is_generation_model, is_image_gen_model, is_audio_model
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +2/-0 (2 lines); hunk: NemotronH_Nano_VL_V2_Config,; JetNemotronConfig,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py`；patch 关键词为 config, attention, moe, processor, cache, spec。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/configs/__init__.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19291 - [Qwen3.5] Fix missing `quant_config` in `Qwen3VL`

- 链接：https://github.com/sgl-project/sglang/pull/19291
- 状态/时间：`merged`，created 2026-02-25, merged 2026-03-02；作者 `mmangkad`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper；关键词：config, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19333 - fix qwen3_vl visual module loading

- 链接：https://github.com/sgl-project/sglang/pull/19333
- 状态/时间：`merged`，created 2026-02-25, merged 2026-02-27；作者 `WingEdge777`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper；关键词：attention, kv, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 attention, kv, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19693 - [NPU] Fix Qwen3-VL-8B Accuracy for NPU

- 链接：https://github.com/sgl-project/sglang/pull/19693
- 状态/时间：`open`，created 2026-03-02；作者 `Todobe`。
- 代码 diff 已读范围：`14` 个文件，`+199/-108`；代码面：model wrapper, attention/backend, MoE/router, scheduler/runtime；关键词：kv, attention, cache, config, cuda, mla, spec, eagle, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +79/-20 (99 lines); hunk: from functools import lru_cache, partial; def rot_pos_emb(; 符号: rot_pos_emb, fast_pos_embed_interpolate, forward, forward
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +37/-38 (75 lines); hunk: from sglang.srt.layers.radix_attention import AttentionType; def forward_decode_graph(; 符号: forward_decode_graph, forward_decode_graph
  - `python/sglang/srt/models/llama.py` modified +37/-4 (41 lines); hunk: maybe_remap_kv_scale_name,; def __init__(; 符号: LlamaMLP, __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py` modified +12/-23 (35 lines); hunk: from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa; def _capture_graph(self, graph, pool, stream, run_once_fn):; 符号: _capture_graph, _get_update_attr_name, _get_update_attr_type, _get_update_attr_name
  - `python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py` modified +11/-10 (21 lines); hunk: from sglang.srt.configs.model_config import AttentionArch, is_deepseek_nsa; def _capture_graph(self, graph, pool, stream, run_once_fn):; 符号: _capture_graph, _get_update_attr_name, _get_update_attr_name, _get_update_attr_type
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/models/llama.py`；patch 关键词为 kv, attention, cache, config, cuda, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/models/llama.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20759 - [Bugfix] fix qwen3vl hang when --mm-enable-dp-encoder is enable

- 链接：https://github.com/sgl-project/sglang/pull/20759
- 状态/时间：`merged`，created 2026-03-17, merged 2026-03-19；作者 `ZLkanyo009`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：model wrapper；关键词：attention, config, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +2/-2 (4 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 attention, config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20788 - [DP encoder] Fix `pos_emb `layer TP issue when DP encoder enabled for Qwen3 VL

- 链接：https://github.com/sgl-project/sglang/pull/20788
- 状态/时间：`merged`，created 2026-03-17, merged 2026-03-18；作者 `jianan-gu`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper；关键词：attention, config, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 attention, config, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20857 - add EVS support for Qwen3-VL

- 链接：https://github.com/sgl-project/sglang/pull/20857
- 状态/时间：`open`，created 2026-03-18；作者 `artetaout`。
- 代码 diff 已读范围：`5` 个文件，`+151/-4`；代码面：model wrapper, multimodal/processor, docs/config；关键词：config, vision, processor, cuda, moe, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +109/-0 (109 lines); hunk: from PIL import Image; from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration; 符号: __init__, __init__, _maybe_apply_qwen3_evs, get_mm_data
  - `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py` modified +20/-2 (22 lines); hunk: def get_rope_index(; def get_rope_index(; 符号: get_rope_index, get_rope_index, get_rope_index
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-2 (12 lines); hunk: WeightsMapper,; def forward(; 符号: forward, Qwen3VLForConditionalGeneration, Qwen3VLForConditionalGeneration, __init__
  - `python/sglang/srt/multimodal/evs/evs_processor.py` modified +10/-0 (10 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/configs/qwen3_vl.py` modified +2/-0 (2 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 config, vision, processor, cuda, moe, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/qwen_vl.py`, `python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py`, `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- 链接：https://github.com/sgl-project/sglang/pull/21458
- 状态/时间：`merged`，created 2026-03-26, merged 2026-04-01；作者 `yctseng0211`。
- 代码 diff 已读范围：`1` 个文件，`+101/-3`；代码面：model wrapper；关键词：attention, cache, config, cuda, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunk: from sglang.srt.layers.quantization.base_config import QuantizationConfig; from sglang.srt.models.qwen2 import Qwen2Model; 符号: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`；patch 关键词为 attention, cache, config, cuda, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21469 - [3/n] lora moe - Support Qwen3-VL-30B-A3B-Instruct

- 链接：https://github.com/sgl-project/sglang/pull/21469
- 状态/时间：`merged`，created 2026-03-26, merged 2026-04-01；作者 `yushengsu-thu`。
- 代码 diff 已读范围：`3` 个文件，`+152/-235`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：attention, lora, moe, cache, expert, kv, spec, test, config, cuda。
- 代码 diff 细节：
  - `test/manual/lora/test_lora_qwen3_vl.py` removed +0/-233 (233 lines); hunk: -import random; 符号: TestLoRAQwen3VLGating, _assert_pattern, test_qwen3_vl_should_apply_lora_regex, test_qwen3_vl_moe_should_apply_lora_regex
  - `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py` added +151/-0 (151 lines); hunk: +# Copyright 2023-2025 SGLang Team; 符号: kl_v2, get_prompt_logprobs, TestLoRAQwen3VL_30B_A3B_Instruct_LogprobDiff, test_lora_qwen3_vl_30b_a3b_instruct_logprob_accuracy
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-2 (3 lines); hunk: def __init__(; 符号: __init__, should_apply_lora
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`, `python/sglang/srt/models/qwen3_vl_moe.py`；patch 关键词为 attention, lora, moe, cache, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/manual/lora/test_lora_qwen3_vl.py`, `test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`, `python/sglang/srt/models/qwen3_vl_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21849 - [VLM]: allow Qwen3.5 models for encoder disaggregation

- 链接：https://github.com/sgl-project/sglang/pull/21849
- 状态/时间：`merged`，created 2026-04-01, merged 2026-04-06；作者 `Ratish1`。
- 代码 diff 已读范围：`4` 个文件，`+190/-3`；代码面：multimodal/processor, tests/benchmarks；关键词：moe, processor, config, cuda, scheduler, test。
- 代码 diff 细节：
  - `test/registered/distributed/test_epd_disaggregation.py` modified +184/-0 (184 lines); hunk: # Omni model for local testing; override via env var EPD_OMNI_MODEL; def test_mmmu(self):; 符号: test_mmmu, TestEPDDisaggregationQwen35, setUpClass, start_encode
  - `python/sglang/srt/disaggregation/encode_server.py` modified +3/-2 (5 lines); hunk: async def _process_mm_items(self, mm_items, modality):; 符号: _process_mm_items
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunk: def get_mm_data(self, prompt, embeddings, **kwargs):; 符号: get_mm_data
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunk: def _handle_encoder_disaggregation(self):; 符号: _handle_encoder_disaggregation
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py`；patch 关键词为 moe, processor, config, cuda, scheduler, test。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/distributed/test_epd_disaggregation.py`, `python/sglang/srt/disaggregation/encode_server.py`, `python/sglang/srt/multimodal/processors/qwen_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22038 - [VLM] Chunk-aware ViT encoding with per-image cache and lazy device transfer

- 链接：https://github.com/sgl-project/sglang/pull/22038
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-04；作者 `yhyang201`。
- 代码 diff 已读范围：`7` 个文件，`+167/-410`；代码面：model wrapper, multimodal/processor, scheduler/runtime；关键词：cache, vision, processor, attention, cuda, eagle, spec。
- 代码 diff 细节：
  - `python/sglang/srt/managers/mm_utils.py` modified +147/-286 (433 lines); hunk: _GPU_FEATURE_BUFFER: Optional[torch.Tensor] = None; def _get_precomputed_embedding(; 符号: _get_precomputed_embedding, get_embedding_items_per_chunk_with_extra_padding, _move_items_to_device, _get_chunked_prefill_embedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +10/-104 (114 lines); hunk: """Inference-only Qwen3-VL model compatible with HuggingFace weights."""; from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; 符号: get_image_feature, get_video_feature
  - `python/sglang/srt/managers/schedule_batch.py` modified +0/-15 (15 lines); hunk: def prepare_for_extend(self):; 符号: prepare_for_extend
  - `python/sglang/srt/mem_cache/multimodal_cache.py` modified +7/-0 (7 lines); hunk: def set(; 符号: set, get_single, has
  - `python/sglang/srt/models/deepseek_vl2.py` modified +1/-3 (4 lines); hunk: def get_image_feature(self, items: List[MultimodalDataItem]):; 符号: get_image_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/schedule_batch.py`；patch 关键词为 cache, vision, processor, attention, cuda, eagle。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/managers/mm_utils.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/managers/schedule_batch.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22052 - [Fix] Enable precise embedding interpolation by default for Qwen3-VL

- 链接：https://github.com/sgl-project/sglang/pull/22052
- 状态/时间：`open`，created 2026-04-03；作者 `chengmengli06`。
- 代码 diff 已读范围：`3` 个文件，`+10/-11`；代码面：model wrapper, docs/config；关键词：vision, moe, cache, config, doc, fp8, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +3/-7 (10 lines); hunk: def __init__(; def fast_pos_embed_interpolate_from_list(self, grid_thw):; 符号: __init__, fast_pos_embed_interpolate_from_list
  - `python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunk: Please consult the documentation below and [server_args.py](https://github.com/s
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`；patch 关键词为 vision, moe, cache, config, doc, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22073 - [Feature] Adding Qwen3-asr Model Support

- 链接：https://github.com/sgl-project/sglang/pull/22073
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-07；作者 `adityavaid`。
- 代码 diff 已读范围：`10` 个文件，`+571/-11`；代码面：model wrapper, multimodal/processor, tests/benchmarks, docs/config；关键词：config, moe, attention, processor, vision, spec, benchmark, cache, doc, kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunk: +"""Qwen3-ASR model compatible with HuggingFace weights"""; 符号: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunk: +import torch; 符号: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, __init__
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunk: +import re; 符号: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7 (64 lines); hunk: TIMESTAMP_BASE_TOKEN_ID = 50365 # <\|0.00\|>; def _convert_to_internal_request(; 符号: _detect_model_family, OpenAIServingTranscription, __init__, _request_id_prefix
  - `docs/supported_models/text_generation/multimodal_language_models.md` modified +29/-0 (29 lines); hunk: in the GitHub search bar.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`；patch 关键词为 config, moe, attention, processor, vision, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22089 - [Feature] Add chunk-based streaming ASR for Qwen3-ASR

- 链接：https://github.com/sgl-project/sglang/pull/22089
- 状态/时间：`merged`，created 2026-04-04, merged 2026-04-09；作者 `SammLSH`。
- 代码 diff 已读范围：`5` 个文件，`+263/-2`；代码面：multimodal/processor；关键词：config, processor, spec, cache, kv。
- 代码 diff 细节：
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +125/-0 (125 lines); hunk: from __future__ import annotations; TranscriptionVerboseResponse,; 符号: _handle_streaming_request, _generate_transcription_stream, _generate_chunked_asr_stream
  - `python/sglang/srt/entrypoints/openai/streaming_asr.py` added +93/-0 (93 lines); hunk: +import io; 符号: StreamingASRState:, get_prefix_text, update, finalize
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py` modified +23/-0 (23 lines); hunk: class TranscriptionAdapter(ABC):; 符号: TranscriptionAdapter, build_sampling_params, supports_chunked_streaming, prompt_template
  - `python/sglang/srt/entrypoints/openai/transcription_adapters/qwen3_asr.py` modified +20/-0 (20 lines); hunk: TranscriptionAdapter,; 符号: Qwen3ASRAdapter, supports_chunked_streaming, chunked_streaming_config, prompt_template
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` modified +2/-2 (4 lines); hunk: AUDIO_PLACEHOLDER = "<\|audio_start\|><\|audio_pad\|><\|audio_end\|>"; def _build_transcription_prompt(self, input_text: Union[str, list]) -> str:; 符号: _build_transcription_prompt, compute_mrope_positions
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py`；patch 关键词为 config, processor, spec, cache, kv。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/entrypoints/openai/serving_transcription.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`, `python/sglang/srt/entrypoints/openai/transcription_adapters/base.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22230 - [Feature] Support eagle3 for qwen3-vl

- 链接：https://github.com/sgl-project/sglang/pull/22230
- 状态/时间：`merged`，created 2026-04-07, merged 2026-04-09；作者 `litmei`。
- 代码 diff 已读范围：`1` 个文件，`+24/-0`；代码面：model wrapper；关键词：config, eagle, processor, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +24/-0 (24 lines); hunk: def __init__(; def forward(; 符号: __init__, separate_deepstack_embeds, forward, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 config, eagle, processor, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22266 - [NPU] fix qwen3.5 video processor

- 链接：https://github.com/sgl-project/sglang/pull/22266
- 状态/时间：`merged`，created 2026-04-07, merged 2026-04-08；作者 `zhaozx-cn`。
- 代码 diff 已读范围：`1` 个文件，`+177/-21`；代码面：multimodal/processor；关键词：processor, test。
- 代码 diff 细节：
  - `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` modified +177/-21 (198 lines); hunk: group_images_by_shape,; def _preprocess(; 符号: transform_patches_to_flatten, npu_wrapper_preprocess, _preprocess, _preprocess
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py`；patch 关键词为 processor, test。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/hardware_backend/npu/modules/qwen_vl_processor.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22431 - Fix Qwen3.5 video processing when passing video_data in "processor_output" format

- 链接：https://github.com/sgl-project/sglang/pull/22431
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-18；作者 `lkhl`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：multimodal/processor；关键词：processor。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +1/-1 (2 lines); hunk: async def preprocess_video(; 符号: preprocess_video
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/qwen_vl.py`；patch 关键词为 processor。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/qwen_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22839 - fix(config): Add from_dict() for Qwen3VL config classes

- 链接：https://github.com/sgl-project/sglang/pull/22839
- 状态/时间：`open`，created 2026-04-15；作者 `libermeng`。
- 代码 diff 已读范围：`5` 个文件，`+306/-0`；代码面：tests/benchmarks, docs/config；关键词：config, moe, vision, attention, expert, topk, kv, router, test。
- 代码 diff 细节：
  - `test/registered/unit/configs/test_qwen3_vl_config.py` added +198/-0 (198 lines); hunk: +"""Unit tests for qwen3_vl and qwen3_5 config from_dict() handling.; 符号: TestQwen3VLConfigFromDict, test_qwen3vl_config_dict_conversion, test_qwen3vl_config_with_object, test_qwen3vl_moe_config_dict_conversion
  - `python/sglang/srt/configs/qwen3_5.py` modified +71/-0 (71 lines); hunk: class Qwen3_5Config(PretrainedConfig):; class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):; 符号: Qwen3_5Config, from_dict, __init__, Qwen3_5MoeVisionConfig
  - `python/sglang/srt/configs/qwen3_vl.py` modified +30/-0 (30 lines); hunk: class Qwen3VLConfig(PretrainedConfig):; def __init__(; 符号: Qwen3VLConfig, from_dict, __init__, __init__
  - `python/sglang/srt/utils/hf_transformers_utils.py` modified +4/-0 (4 lines); hunk: Qwen3_5Config,; DeepseekVLV2Config,
  - `python/sglang/srt/configs/__init__.py` modified +3/-0 (3 lines); hunk: from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig; "JetVLMConfig",
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py`；patch 关键词为 config, moe, vision, attention, expert, topk。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/configs/test_qwen3_vl_config.py`, `python/sglang/srt/configs/qwen3_5.py`, `python/sglang/srt/configs/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22848 - [Feature] WebSocket streaming audio input for ASR

- 链接：https://github.com/sgl-project/sglang/pull/22848
- 状态/时间：`open`，created 2026-04-15；作者 `SammLSH`。
- 代码 diff 已读范围：`6` 个文件，`+937/-43`；代码面：model wrapper, tests/benchmarks；关键词：config, cache, kv, processor, test。
- 代码 diff 细节：
  - `test/manual/models/test_qwen3_asr.py` modified +451/-3 (454 lines); hunk: """; TEST_AUDIO_ZH_URL = (; 符号: _normalize_for_wer, _wer, download_audio, download_audio
  - `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py` added +376/-0 (376 lines); hunk: +"""WebSocket transport for OpenAI Realtime API-style transcription.; 符号: names, _safe_close_websocket, _pcm_to_wav, RealtimeMessageType
  - `python/sglang/srt/entrypoints/openai/streaming_asr.py` modified +78/-7 (85 lines); hunk: +import asyncio; class StreamingASRState:; 符号: StreamingASRState:, StreamingASRState:, get_prefix_text, _record_emit
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +15/-33 (48 lines); hunk: import uuid; TranscriptionVerboseResponse,; 符号: _generate_chunked_asr_stream, _generate_chunked_asr_stream, _generate_chunked_asr_stream, handle_websocket
  - `python/sglang/srt/server_args.py` modified +10/-0 (10 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py`；patch 关键词为 config, cache, kv, processor, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/manual/models/test_qwen3_asr.py`, `python/sglang/srt/entrypoints/openai/serving_transcription_websocket.py`, `python/sglang/srt/entrypoints/openai/streaming_asr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23115 - fix: guard self.model access in Qwen3VLMoeForConditionalGeneration.load_weights

- 链接：https://github.com/sgl-project/sglang/pull/23115
- 状态/时间：`open`，created 2026-04-18；作者 `octo-patch`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper, MoE/router；关键词：moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl_moe.py`；patch 关键词为 moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23220 - Bugfix: Qwen3-VL-MoE adapt encoder_only

- 链接：https://github.com/sgl-project/sglang/pull/23220
- 状态/时间：`open`，created 2026-04-20；作者 `Hide-on-bushsh`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper, MoE/router；关键词：moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl_moe.py` modified +1/-0 (1 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl_moe.py`；patch 关键词为 moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23304 - [Bugfix] Fix Qwen3-VL rope config compatibility

- 链接：https://github.com/sgl-project/sglang/pull/23304
- 状态/时间：`closed`，created 2026-04-21, closed 2026-04-21；作者 `Chokoyo`。
- 代码 diff 已读范围：`1` 个文件，`+1/-10`；代码面：model wrapper；关键词：attention, config。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3.py` modified +1/-10 (11 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3.py`；patch 关键词为 attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23469 - [NPU] adapt the Qwen3-ASR model for deployment on NPU

- 链接：https://github.com/sgl-project/sglang/pull/23469
- 状态/时间：`open`，created 2026-04-22；作者 `xdtbynd`。
- 代码 diff 已读范围：`1` 个文件，`+18/-0`；代码面：misc；关键词：cuda。
- 代码 diff 细节：
  - `python/sglang/srt/utils/common.py` modified +18/-0 (18 lines); hunk: def load_audio(; 符号: load_audio
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/utils/common.py`；patch 关键词为 cuda。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/utils/common.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：51；open PR 数：22。
- 仍需跟进的 open PR：[#12261](https://github.com/sgl-project/sglang/pull/12261), [#12662](https://github.com/sgl-project/sglang/pull/12662), [#12703](https://github.com/sgl-project/sglang/pull/12703), [#13918](https://github.com/sgl-project/sglang/pull/13918), [#14886](https://github.com/sgl-project/sglang/pull/14886), [#16491](https://github.com/sgl-project/sglang/pull/16491), [#16571](https://github.com/sgl-project/sglang/pull/16571), [#16785](https://github.com/sgl-project/sglang/pull/16785), [#16996](https://github.com/sgl-project/sglang/pull/16996), [#17202](https://github.com/sgl-project/sglang/pull/17202), [#17276](https://github.com/sgl-project/sglang/pull/17276), [#18721](https://github.com/sgl-project/sglang/pull/18721), [#18771](https://github.com/sgl-project/sglang/pull/18771), [#19242](https://github.com/sgl-project/sglang/pull/19242), [#19693](https://github.com/sgl-project/sglang/pull/19693), [#20857](https://github.com/sgl-project/sglang/pull/20857), [#22052](https://github.com/sgl-project/sglang/pull/22052), [#22839](https://github.com/sgl-project/sglang/pull/22839), [#22848](https://github.com/sgl-project/sglang/pull/22848), [#23115](https://github.com/sgl-project/sglang/pull/23115) ...
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
