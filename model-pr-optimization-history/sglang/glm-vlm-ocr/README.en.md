# SGLang GLM VLM/OCR Support and Optimization Timeline

This document is based on SGLang `origin/main` snapshot `b3e6cf60a` (`2026-04-22`) and sgl-cookbook `origin/main` snapshot `816bad5` (`2026-04-21`). It covers GLM-4V, GLM-4.1V, GLM-4.5V, GLM-4.6V, GLM-Glyph, and GLM-OCR.

For this rewrite, each PR was inspected through source diff. Every card records motivation/root cause, implementation idea, key code fragment, validation signal, and current risk.

## Summary

GLM VLM/OCR risk is concentrated at the multimodal boundary rather than in ordinary text-only GLM MoE:

- processor registration, especially optional GLM-OCR dependencies
- vision encoder TP/DP/PP with 12 vision heads, dummy heads, and pipeline stages
- separate loaders in GLM4V-MoE and GLM-OCR
- Transformers 5.x field drift into `text_config`
- Conv3D/Linear patch embedding optimization and its loader regression history
- OCR-specific validation for OCR/MTP/NextN rather than image-caption smoke only

## Code Surfaces

- `python/sglang/srt/models/glm4v.py`
- `python/sglang/srt/models/glm4v_moe.py`
- `python/sglang/srt/models/glm_ocr.py`
- `python/sglang/srt/models/glm_ocr_nextn.py`
- `python/sglang/srt/multimodal/processors/glm4v.py`
- `python/sglang/srt/layers/attention/vision.py`
- `python/sglang/srt/layers/rotary_embedding.py`
- `docs_new/docs/basic_usage/glmv.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-Glyph.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx`

## Merged PRs

### `#8798`: GLM-4.1V / GLM-4.5V support

- Motivation: add native SGLang support for GLM-4.1V Thinking and GLM-4.5V instead of relying on an incomplete Qwen2.5-VL-like path. The PR adds model registration, GLM4V/GLM4V-MoE model files, processor, chat template, MRoPE handling, and tests.
- Implementation: registers `Glm4vForConditionalGeneration`, adds the `glm4v` conversation template, supports multiple vision-start token ids, and routes GLM4V through the multimodal RoPE index logic.
- Key code fragment:

```python
"Glm4vForConditionalGeneration",
```

```python
register_conv_template(
    Conversation(
        name="glm4v",
        image_token="<|begin_of_image|><|image|><|end_of_image|>",
        video_token="<|begin_of_video|><|video|><|end_of_video|>",
    )
)
```

```python
vision_start_token_id = (
    [vision_start_token_id]
    if isinstance(vision_start_token_id, int)
    else vision_start_token_id
)
```

- Validation: the PR reports single-image, multi-image, video tests, and GLM4.1V MMMU around `0.701`.
- Risk: GLM4V resembles Qwen VLM, but token wrappers, MRoPE, and processor behavior are not identical.

### `#9059`: dummy-head TP for GLM4.1V/4.5V

- Motivation: text attention can shard over `num_key_value_heads=8`, while vision attention has 12 heads. TP=8 needs dummy heads to make the vision side divisible.
- Implementation: compute `num_dummy_heads`, store it on `vision_config`, and pad q/k/v, projection, and norm weights during GLM4V and GLM4V-MoE loading.
- Key code fragment:

```python
num_dummy_heads = ((num_heads + tp_size) // tp_size) * tp_size - num_heads
setattr(self.config.vision_config, "num_dummy_heads", num_dummy_heads)
```

```python
if "attn.qkv_proj" in name:
    wq, wk, wv = loaded_weight.chunk(3, dim=0)
    loaded_weight = torch.cat([wq, wk, wv], dim=0)
```

- Validation: removes the practical TP ceiling caused by vision-head divisibility.
- Risk: dummy-head config and loader padding must stay synchronized.

### `#9245`: default GLM-4.5V to FA3

- Motivation: GLM-4.5V recommended FA3, but SGLang default selection did not reflect that.
- Implementation: adds GLM4V-MoE architecture to the default-FA3 list.
- Key code fragment:

```python
"Glm4vMoeForConditionalGeneration",
```

- Validation: default backend change only.
- Risk: unsupported platforms need explicit `--mm-attention-backend`.

### `#9554`: fix GLM45V torch.compile launch

- Motivation: cuda graph plus `torch.compile` failed because a fake tensor requiring grad interacted with an `out=` operation on the shared VLM forward path.
- Implementation: wrap the VLM forward in inference-only no-grad mode.
- Key code fragment:

```python
@torch.no_grad()
def forward(...):
```

- Validation: PR reports successful GLM45V cuda graph and torch compile launch with unchanged MMMU.
- Risk: GLM4V compile failures can live in shared Qwen VLM files.

### `#9884`: fix GLM4V vision-block norm

- Motivation: after a shared vision-block refactor, `norm2(..., residual=attn2d)` was called, but GLM4V had overridden norms with a forward signature that did not accept `residual`.
- Implementation: remove the incompatible GLM-specific norm override and pass GLM RMSNorm epsilon into the compatible parent path.
- Key code fragment:

```python
rms_norm_eps=config.rms_norm_eps,
```

- Validation: fixes a runtime signature mismatch.
- Risk: inheritance from Qwen VLM blocks requires signature checks, not only shape checks.

### `#10147` / `#10228`: EAGLE3 field coverage

- Motivation: EAGLE3/speculative infrastructure expected `capture_aux_hidden_states` to exist on model objects.
- Implementation: initialize the field on GLM4V and GLM4V-MoE.
- Key code fragment:

```python
self.capture_aux_hidden_states = False
```

- Validation: prevents missing-attribute failures.
- Risk: dense, MoE, and OCR variants should keep the same speculative-decoding object contract.

### `#11166`: utility package move

- Motivation: move `python/sglang/srt/utils.py` into a `utils/` package.
- Implementation: add package re-export and update GLM VLM imports.
- Key code fragment:

```python
from .common import *
```

```python
from sglang.srt.utils.hf_transformers_utils import get_processor
```

- Validation: structural import change only.
- Risk: import-only changes can still break processor registration.

### `#11388`: replace `F.pad` with `torch.cat`

- Motivation: prefixing zero to `cu_seqlens` is a hot VLM path, and `torch.cat` is lighter than `F.pad`.
- Implementation: use `torch.cat` in GLM4V and sibling VLM files.
- Key code fragment:

```python
cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
```

- Validation: micro-optimization with unchanged semantics.
- Risk: avoid reintroducing heavier padding in vision sequence construction.

### `#11922`: improve ruff checks

- Motivation: malformed pre-commit arguments made F401/F821 checks unreliable.
- Implementation: split ruff args and auto-fix import issues; GLM4V import hygiene was touched.
- Key code fragment:

```yaml
args:
  - --select=F401,F821
  - --fix
```

- Validation: import hygiene.
- Risk: lint fixes can affect dynamic processor discovery.

### `#12117`: GLM-4-0414 / GLM-4.1V refactor

- Motivation: migrate GLM-4 and GLM-4.1V to newer SGLang model interfaces.
- Implementation: rewrite `Glm4vVisionBlock` as a standalone module, use `VisionAttention`, route multimodal embeddings through `general_mm_embed_routine`, use `MultiModalityDataPaddingPatternMultimodalTokens`, and align PP missing-layer handling.
- Key code fragment:

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    projection_size=dim,
    use_qkv_parallel=True,
    proj_bias=True,
)
```

```python
x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
x = x_after_add + mlp_out
```

```python
pattern = MultiModalityDataPaddingPatternMultimodalTokens()
return pattern.pad_input_tokens(input_ids, mm_inputs)
```

- Validation: later DP/PP and processor work builds on this interface.
- Risk: bypassing `general_mm_embed_routine` can break PP proxy tensors or image token padding.

### `#13228`: clean up vision attention code

- Motivation: remove per-model hard-coded/dead vision attention parameters and centralize backend selection.
- Implementation: let `VisionAttention` resolve the backend rather than model-specific code.
- Key code fragment:

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    projection_size=dim,
    flatten_batch=True,
)
```

- Validation: reduces GLM/Qwen vision path divergence.
- Risk: backend tuning should live in common attention or launch args.

### `#14097`: GLM-V vision encoder DP

- Motivation: TP=8 is awkward for 12 vision heads and VLM TTFT is sensitive to the vision encoder; GLM-V needed data-parallel encoder support.
- Implementation: read `mm_enable_dp_encoder`; set vision merger TP size/rank to `1/0` under DP; use `run_dp_sharded_mrope_vision_model(..., rope_type="rope_3d")`.
- Key code fragment:

```python
self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
```

```python
return run_dp_sharded_mrope_vision_model(
    self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
)
```

```python
self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
```

- Validation: official GLM VLM docs also recommend `--mm-enable-dp-encoder` for TP=8.
- Risk: compare DP and no-DP outputs after changes.

### `#14720`: GLM-4.6V / GLM-4.1V PP

- Motivation: non-last PP ranks do not own `lm_head.weight`, but GLM4V loader and multimodal embedding assumed full ownership.
- Implementation: pass `PPProxyTensors` into `general_mm_embed_routine`, skip `lm_head.*` on non-last ranks, and skip parameters absent from the current PP stage.
- Key code fragment:

```python
def forward(..., pp_proxy_tensors: Optional[PPProxyTensors] = None):
    hidden_states = general_mm_embed_routine(..., pp_proxy_tensors=pp_proxy_tensors)
```

```python
if name.startswith("lm_head.") and not self.pp_group.is_last_rank:
    continue
if name not in params_dict:
    continue
```

- Validation: adds GLM4.1V PP accuracy coverage.
- Risk: loader changes must be stage-aware.

### `#14927`: nightly CI for `glm4v_moe`

- Motivation: dense GLM4V had coverage, but GLM4V-MoE could regress silently.
- Implementation: add `zai-org/GLM-4.5V-FP8` to nightly VLM MMMU.
- Key code fragment:

```python
ModelLaunchSettings(
    "zai-org/GLM-4.5V-FP8", extra_args=["--tp=2"]
): ModelEvalMetrics(0.26, 32.0)
```

- Validation: continuous regression guard for GLM4V-MoE.
- Risk: loader/quantization/processor changes should run MoE VLM, not only dense GLM4V.

### `#14998`: Transformers version validation for GLM-4.6V MoE

- Motivation: GLM-4.6V MoE needs Transformers 5.x, while unrelated models should not be forced onto TF5.
- Implementation: detect by model path or `vision_config.model_type`, raise on too-old Transformers for needed models, warn for models that do not require TF5.
- Key code fragment:

```python
is_glm_46vmoe = "glm-4.6v" in self.model_path.lower() or (
    vision_config is not None
    and getattr(vision_config, "model_type", None) == "glm4v_moe_vision"
)
```

- Validation: version mismatch fails early.
- Risk: keep the check scoped to the exact architecture requiring TF5.

### `#15205`: cos/sin cache for Qwen3-VL and GLM-4.1V

- Motivation: 2D vision RoPE repeatedly recomputed frequencies and `cos()/sin()`. The PR reports a local reduction from about `490us` to `186us`.
- Implementation: add `RotaryEmbedding.get_cos_sin`, switch GLM4V visual RoPE to `get_rope`, return cached cos/sin from `rot_pos_emb`, and let `VisionAttention` accept explicit cos/sin tensors.
- Key code fragment:

```python
def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = self.cos_sin_cache[:seqlen]
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin
```

```python
rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(grid_thw)
```

- Validation: PR reports no Qwen3-VL MMMU drop and better VLM cache TTFT.
- Risk: GLM visual RoPE half-dimension duplication and grid indexing must remain exact.

### `#15434`: move NPU `cu_seqlens` to CPU once

- Motivation: Ascend/NPU vision attention moved `cu_seqlens` to CPU inside every layer, creating dispatch bubbles.
- Implementation: resolve sequence lengths on CPU in `VisionAttention` and move GLM4V `cu_seqlens` once on NPU.
- Key code fragment:

```python
cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")
```

```python
if is_npu():
    cu_seqlens = cu_seqlens.to("cpu")
```

- Validation: platform hot-path optimization with unchanged CUDA behavior.
- Risk: keep NPU transfers out of per-layer loops.

### `#17122`: GLM-4V dummy-head and NPU processor bugfix

- Motivation: `VisionAttention` needs `num_dummy_heads` during construction/calculation, and NPU processor handling needed a GLM4V exception.
- Implementation: call `vision_utils.update_vit_attn_dummy_heads_config` before building visual modules, pass `num_dummy_heads` into blocks, and exclude `Glm4vProcessor` from the generic NPU processor patch branch.
- Key code fragment:

```python
vision_utils.update_vit_attn_dummy_heads_config(self.config)
```

```python
num_dummy_heads=vision_config.num_dummy_heads,
```

```python
elif processor.__class__.__name__ not in {"Glm4vProcessor"}:
```

- Validation: adds Ascend GLM-4.5V coverage.
- Risk: dummy-head config must be updated before module construction.

### `#17420`: optimize GLM4V `get_rope_index`

- Motivation: GLM4V `get_rope_index` was expensive for long multimodal inputs.
- Implementation: preallocate token-type metadata, reduce `.item()`/CPU round-trips, group consecutive modalities, use device-local ranges, and compute MRoPE deltas with tensor reductions.
- Key code fragment:

```python
input_token_type = [""] * len(input_tokens)
```

```python
t_index = (
    torch.arange(llm_grid_t, device=position_ids.device)
    .view(-1, 1)
    .expand(llm_grid_t, llm_grid_h * llm_grid_w)
    .reshape(-1)
)
```

```python
max_position_ids = position_ids.amax(dim=0, keepdim=False)
mrope_position_deltas = max_position_ids.amax(-1, keepdim=True) + 1 - attention_mask.shape[-1]
```

- Validation: PR reports no lmms-eval drop.
- Risk: test image, multi-image, and video token layouts.

### `#17582`: GLM-OCR support

- Motivation: add GLM-OCR architecture, processor, Transformers 5.x requirement handling, and NextN/MTP speculative support.
- Implementation: register `GlmOcrForConditionalGeneration`, add `glm_ocr.py` and `glm_ocr_nextn.py`, use `VisionAttention(qk_normalization_by_head_size=True)`, map draft architecture to `GlmOcrForConditionalGenerationNextN`, and register OCR in the GLM4V processor.
- Key code fragment:

```python
"GlmOcrForConditionalGeneration",
```

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    qk_normalization_by_head_size=True,
    flatten_batch=True,
)
```

```python
if is_draft_model and self.hf_config.architectures[0] in [
    "GlmOcrForConditionalGeneration",
]:
    self.hf_config.architectures[0] = "GlmOcrForConditionalGenerationNextN"
```

- Validation: official docs later cover OCRBench, OmniDocBench, and EAGLE/MTP launch.
- Risk: validate OCR pages and MTP acceptance, not just image captioning.

### `#18885`: processor registration when GLM-OCR is unavailable

- Motivation: without `transformers.models.glm_ocr`, the GLM4V processor module failed to import, dropping GLM-4.1V and GLM-4.5V registrations too.
- Implementation: make only the OCR import conditional and filter `None` from the model list.
- Key code fragment:

```python
try:
    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None
```

```python
models = [
    m
    for m in [
        Glm4vForConditionalGeneration,
        Glm4vMoeForConditionalGeneration,
        GlmOcrForConditionalGeneration,
    ]
    if m is not None
]
```

- Validation: PR body links nightly failures with missing processor registration.
- Risk: optional OCR dependency must not break non-OCR GLM VLM serving.

### `#20033`: replace GLM4V Conv3D projection with Linear

- Motivation: GLM4V patch embedding Conv3D is equivalent to Linear for already-flattened patch tensors and can be faster.
- Implementation: add `linear`, copy reshaped Conv3D weights into it after loading, delete `proj`, update dtype/device accessors, and add correctness/perf tests.
- Key code fragment:

```python
k = self.in_channels * self.temporal_patch_size * self.patch_size**2
self.linear = nn.Linear(in_features=k, out_features=self.hidden_size, bias=True)
```

```python
def copy_conv3d_weight_to_linear(self):
    with torch.no_grad():
        self.linear.weight.copy_(self.proj.weight.view(self.hidden_size, -1))
        self.linear.bias.copy_(self.proj.bias)
    del self.proj
```

```python
self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- Validation: adds Conv3D/Linear close test and CUDA benchmark; PR reports no lmms-eval drop.
- Risk: GLM4V-MoE and OCR have independent loaders.

### `#20282`: unified `Conv2dLayer` / `Conv3dLayer`

- Motivation: PyTorch 2.9.1 plus old CuDNN had a Conv3D bug, and patch-embedding convs with kernel=stride can be accelerated with unfold+linear.
- Implementation: add `sglang/srt/layers/conv.py`, detect when a conv can be linearized, migrate GLM4V/Qwen VLM and many vision models, and remove the global server compatibility check.
- Key code fragment:

```python
def _check_enable_linear(kernel_size, stride, padding, dilation, groups) -> bool:
    return (
        kernel_size == stride
        and all(p == 0 for p in padding)
        and all(d == 1 for d in dilation)
        and groups == 1
    )
```

```python
x = x.unfold(2, K1, K1).unfold(3, K2, K2)
x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)
```

- Validation: adds `test/unit/test_conv_layer.py`.
- Risk: this is the more durable surface after `#20033`; read it together with `#20463`/`#20740`.

### `#20463` and `#20740`: MoE/OCR loader regression and revert

- Motivation: `#20033` copied Conv3D weights only in the dense `glm4v.py` loader. `glm4v_moe.py` and `glm_ocr.py` have their own loaders, so Linear weights could remain random and produce image-unrelated outputs.
- `#20463` implementation: call `copy_conv3d_weight_to_linear()` at the end of MoE/OCR loaders, guarded by `is_nextn`.
- `#20740` implementation: revert that direct call; current main no longer contains it.
- Key code fragment:

```python
if not is_nextn:
    self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

```python
# #20740 removed the direct MoE/OCR loader copy call.
```

- Validation: `#20463` reports B200 TP=4 GLM-4.6V-FP8 image correctness after the fix; `#20740` defines current-main state.
- Risk: document both PRs. The long-term fix surface is shared Conv layer plus loader-aware validation.

### `#21134`: GLM-V/OCR Transformers 5.x and MTP omission fix

- Motivation: Transformers 5.x can place GLM fields under `text_config`; MTP safetensors loading missed that field; GLM-OCR merger dimension should come from `text_config.intermediate_size`.
- Implementation: read `num_nextn_predict_layers` through `getattr(hf_config, "text_config", hf_config)`, normalize `language_model.` and `model.visual.` names before MTP/decoder mapping, and pass OCR text config into the vision model.
- Key code fragment:

```python
num_nextn_layers = getattr(
    getattr(hf_config, "text_config", hf_config),
    "num_nextn_predict_layers",
    getattr(hf_config, "num_nextn_predict_layers", 0),
)
```

```python
if "language_model." in name:
    name = name.replace("language_model.", "")
if "model.visual." in name:
    name = name.replace("model.visual.", "visual.")
```

```python
context_dim=text_config.intermediate_size,
```

- Validation: fixes MTP acceptance and GLM-OCR config-field drift.
- Risk: future GLM-OCR changes should prefer `text_config` when available.

## Open PR Radar

### `#9349`: GLM-4.5V FP8 fused-MoE tuning

- Motivation: add MoE kernel-generation support for GLM-4.5V FP8.
- Implementation: let the tuning script recognize `Glm4vMoeForConditionalGeneration`, read expert/top-k/intermediate-size fields from `config.text_config`, and add an L40S FP8 config.
- Key code fragment:

```python
cfg_source = config.text_config if is_glm4v_moe else config
E = cfg_source.n_routed_experts
topk = cfg_source.num_experts_per_tok
```

- Status/risk: open and no benchmark in the PR body; radar only.

### `#14662`: GLM4.6V ktransformers

- Motivation: expose expert-location metadata for GLM4.6V.
- Implementation: add `get_model_config_for_expert_location` to `Glm4vMoeForConditionalGeneration`.
- Key code fragment:

```python
return ModelConfigForExpertLocation(
    num_layers=config.text_config.num_hidden_layers,
    num_logical_experts=config.text_config.n_routed_experts,
    num_groups=None,
)
```

- Status/risk: open; expert placement metadata rather than visual correctness fix.

### `#19728`: ROCm GLM-4.5V-FP8 startup

- Motivation: MI300X GLM-4.5V-FP8 startup failed with AITER disabled due to MoE padding and HIP FP8 fallback padding interactions.
- Implementation: disable padding adjustment when runtime hidden size already matches weights; add HIP copy helper for padded buffers.
- Key code fragment:

```python
elif hidden_states.shape[1] == w1.shape[2]:
    padded_size = 0
```

```python
dst[: src.shape[0]].copy_(src)
if dst.shape[0] > src.shape[0]:
    dst[src.shape[0] :].fill_(pad_value)
```

- Validation: PR reports targeted MI300X tests and end-to-end startup.
- Status/risk: open; sync AMD docs and FP8 fallback tests if merged.

### `#22961`: NPU GLM-4.5V

- Motivation: NPU `split_qkv_rmsnorm_rope` supports a no-norm mode, and GLM-4.5V should pass arguments according to `use_qk_norm`.
- Implementation: pass norm weights/epsilon only when QK norm exists; otherwise pass `None`.
- Key code fragment:

```python
if self.use_qk_norm:
    eps = self.q_norm.variance_epsilon
    q_weight = self.q_norm.weight
    k_weight = self.k_norm.weight
else:
    eps = None
    q_weight = None
    k_weight = None
```

- Validation: PR reports MMMU accuracy `0.2802`, invalid `0.000`, latency `89.380s`, output throughput `33.565 token/s`.
- Status/risk: open; touches text attention code for GLM-4.5V platform support.

## Cookbook and Public-Doc Evidence

- SGLang `docs/basic_usage/glmv.mdx`: FP8/BF16 launch, `--keep-mm-feature-on-device`, `--mm-attention-backend`, `--mm-max-concurrent-calls`, `--mm-enable-dp-encoder`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, `SGLANG_VLM_CACHE_SIZE_MB=0`, and GLM thinking-budget/custom-logit-processor coverage.
- GLM-4.5V cookbook: NVIDIA B200/H100/H200 and AMD MI300X/MI325X/MI355X support; TP=8 guidance through `--mm-enable-dp-encoder`.
- GLM-4.6V cookbook: 128K context, native multimodal function calling, document understanding, frontend replication, and video input.
- GLM-OCR cookbook: EAGLE/MTP launch, OCRBench, and OmniDocBench validation cues.
- LMSYS blog `GLM-4.5 Meets SGLang`: GLM parser, tool-call parser, MTP/EAGLE, FP8 variants, and MoE architecture context.
- sgl-cookbook `#95`: GLM-4.5V AMD MI300X/MI325X/MI355X.
- sgl-cookbook `#131`: GLM-4.5V/4.6V MI325X.
- sgl-cookbook `#136`: GLM-OCR cookbook.

## Next Work

1. Add OCRBench/OmniDocBench small examples plus MTP acceptance checks for GLM-OCR.
2. Cover dense, MoE, OCR, and NextN loaders for any GLM VLM loader change.
3. Keep no-DP baseline and PP-stage loader checks for vision encoder DP/PP work.
4. For AMD/NPU PRs, preserve startup, graph capture, API request, and accuracy evidence.
5. Any cited PR should have motivation, implementation idea, key code, validation, and current status/risk.
