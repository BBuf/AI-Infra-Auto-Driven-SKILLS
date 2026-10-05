#!/usr/bin/env python3
"""Model profile definitions for LLM pipeline analysis.

Each ModelProfile captures model-family-specific knowledge needed by the
analysis scripts:

  - Which GPU kernel marks the boundary between consecutive transformer layers
    (anchor_kernel)?
  - How many anchor-kernel blocks does one transformer layer produce
    (blocks_per_layer)?
  - What labels describe each sub-block within a layer (half_labels)?
  - How to classify kernels into categories (category_rules)?
  - How to simplify verbose kernel names (simplify_rules)?

Profiles can be auto-inferred from a model's config.json via
``infer_profile(config)``, or explicitly selected via ``--profile``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """Model-family profile for trace analysis.

    Attributes:
        name: Human-readable profile identifier (e.g. "dsv4_csa_hca").
        anchor_kernel: Substring to identify layer-boundary kernels in the
            trace.  ``None`` means the user must supply ``--anchor-kernel``.
        blocks_per_layer: Number of anchor-kernel invocations per transformer
            layer.  For example, DeepSeek-V4 produces 2 mhc_post blocks per
            layer (one for the attention half, one for the MoE half).
        half_labels: Labels for each sub-block within a layer.  Length must
            equal ``blocks_per_layer``.  E.g. ["attn", "ffn"] or ["full"].
        category_rules: Ordered list of ``(display_label, machine_key, rule)``
            tuples for kernel classification.  Rules are evaluated in order;
            the first match wins.  Unmatched kernels fall into "other".
        simplify_rules: Ordered list of ``(pattern, replacement)`` string
            pairs applied (in order) to simplify verbose kernel names.
        default_num_layers: Fallback layer count when auto-detection fails.
    """

    name: str
    anchor_kernel: Optional[str]
    blocks_per_layer: int
    half_labels: List[str]
    category_rules: list  # List[Tuple[str, str, Callable[[str], bool]]]
    simplify_rules: List[Tuple[str, str]]
    default_num_layers: int = 1


# ---------------------------------------------------------------------------
# Helper: build classification functions
# ---------------------------------------------------------------------------


def _any_sub(*substrings: str) -> Callable[[str], bool]:
    """Return a rule that matches if *any* substring is found in the name."""
    return lambda n: any(s.lower() in n.lower() for s in substrings)


def _sub(s: str) -> Callable[[str], bool]:
    """Return a rule that matches a single substring."""
    return lambda n: s.lower() in n.lower()


# ---------------------------------------------------------------------------
# Universal (framework-level) rules
# ---------------------------------------------------------------------------

_UNIVERSAL_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    (
        "● Communication",
        "allreduce",
        _any_sub(
            "allreduce",
            "all_reduce",
            "cross_device_reduce",
            "reduce_scatter",
            "allgather",
            "all_gather",
            "lamport",
            "multimem",
            "nccl",
        ),
    ),
    (
        "  RMSNorm",
        "rmsnorm",
        lambda n: bool(re.search(r"rms_?norm|rms_normalize|layernorm", n, re.I)),
    ),
    ("  FP8 Quant", "quant", lambda n: "quant" in n.lower() or "Quant" in n),
    ("  TopK", "topk", lambda n: "topk" in n.lower()),
    ("  GEMM fp8", "gemm_fp8", _sub("deep_gemm")),
    ("  GEMM bf16", "gemm_bf16", _sub("nvjet")),
    ("  GEMM f32", "gemm_f32", _sub("sm80_xmma")),
    ("  Activation", "activation", _any_sub("silu_mul_clamp", "act_and_mul")),
    (
        "  RadixSort",
        "radixsort",
        lambda n: "radix_sort" in n.lower() or "RadixSort" in n,
    ),
]

_UNIVERSAL_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    ("void (anonymous namespace)::", ""),
    ("void at::native::", ""),
    ("void flashinfer::", ""),
    ("void deep_gemm::sm90_fp8_gemm_1d2d_impl", "deep_gemm::sm90_fp8_gemm_1d2d"),
    ("void fast_hadamard_transform_kernel", "fast_hadamard_transform_kernel"),
    ("void per_token_group_quant_8bit_kernel", "per_token_group_quant_8bit_kernel"),
    (
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        "ncclAllReduce_bf16_RING_LL",
    ),
    ("norm::RMSNormKernel", "RMSNormKernel"),
]


# ---------------------------------------------------------------------------
# DeepSeek-V4 CSA/HCA profile
# ---------------------------------------------------------------------------

_DSV4_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    # Model-specific rules (evaluated before universal rules)
    ("  mHC stats", "mhc_stats", _sub("_hc_mix_stats_")),
    ("  mHC Sinkhorn", "mhc_sinkhorn", _sub("_hc_mix_reduce_sinkhorn")),
    ("  Router scoring/top-k", "moe_gate", _sub("_router_triton_kernel")),
    ("  WO-A projection", "wo_a", _sub("wo_a_")),
    ("  MoE finalize + AllReduce", "allreduce", _sub("moe_finalize_all_reduce")),
    ("★ MLA Attention", "mla", _sub("flash_fwd_splitkv_mla")),
    ("★ MoE Fused", "moe", _sub("fused_moe_kernel")),
    ("  Hadamard Xform", "hadamard", lambda n: "hadamard" in n.lower()),
    ("  Indexer Cache", "indexer", lambda n: "indexer" in n.lower()),
    ("  Paged MQA", "paged_mqa", _sub("paged_mqa_logits")),
    ("  MLA Metadata", "mla_metadata", _sub("get_mla_metadata")),
    ("  C4 Prefill", "c4_prefill", _sub("c4_prefill")),
    ("  C128 Prefill", "c128_prefill", _sub("c128_prefill")),
    ("  RoPE", "rope", _any_sub("deepseek_rope", "fused_norm_rope")),
    ("  MHC Pre GEMM", "mhc_pre_gemm", _sub("mhc_pre_gemm_sqrsum")),
    ("  MHC Pre Fuse", "mhc_pre_fuse", _sub("mhc_pre_big_fuse")),
    ("  MHC Post", "mhc_post", _sub("mhc_post_tilelang")),
    ("  MoE Gate", "moe_gate", _sub("moe_fused_gate")),
    ("  MoE Align", "moe_align", _sub("moe_align_block")),
    ("  MoE Sort", "moe_sort", _sub("count_and_sort")),
    ("  MLA Cache Store", "mla_cache_store", _sub("fused_store_flashmla_cache")),
    ("  Indexer Store", "indexer_store", _sub("fused_store_indexer_cache")),
]

_DSV4_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    (
        "void deep_gemm::sm90_fp8_paged_mqa_logits",
        "deep_gemm::sm90_fp8_paged_mqa_logits",
    ),
    (
        "void deep_gemm::sched::smxx_paged_mqa_logits_metadata",
        "deep_gemm::paged_mqa_logits_metadata",
    ),
    (
        "void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel",
        "flash_fwd_splitkv_mla_fp8_sparse",
    ),
    ("void smxx::decode::flash_fwd_mla_combine_kernel", "flash_fwd_mla_combine"),
    ("void smxx::decode::get_mla_metadata_kernel", "get_mla_metadata"),
    ("mhc_pre_gemm_sqrsum_tilelang_kernel", "mhc_pre_gemm_sqrsum"),
    ("mhc_post_tilelang_kernel", "mhc_post_tilelang"),
    ("mhc_pre_big_fuse_tilelang_kernel", "mhc_pre_big_fuse"),
    ("fused_moe_kernel", "fused_moe_kernel"),
]


# ---------------------------------------------------------------------------
# DeepSeek-V3 MLA profile
# ---------------------------------------------------------------------------

_DSV3_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    ("★ MLA Attention", "mla", _sub("flash_fwd_splitkv_mla")),
    ("★ MoE Fused", "moe", _sub("fused_moe_kernel")),
    ("  MLA Metadata", "mla_metadata", _sub("get_mla_metadata")),
    ("  MLA Combine", "mla_combine", _sub("flash_fwd_mla_combine")),
    ("  RoPE", "rope", _any_sub("deepseek_rope", "fused_norm_rope")),
    ("  MoE Gate", "moe_gate", _sub("moe_fused_gate")),
    ("  MoE Align", "moe_align", _sub("moe_align_block")),
    ("  MoE Sort", "moe_sort", _sub("count_and_sort")),
    ("  MLA Cache Store", "mla_cache_store", _sub("fused_store_flashmla_cache")),
]

_DSV3_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    (
        "void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel",
        "flash_fwd_splitkv_mla_fp8_sparse",
    ),
    ("void smxx::decode::flash_fwd_mla_combine_kernel", "flash_fwd_mla_combine"),
    ("void smxx::decode::get_mla_metadata_kernel", "get_mla_metadata"),
    ("fused_moe_kernel", "fused_moe_kernel"),
]


# ---------------------------------------------------------------------------
# Built-in profile instances
# ---------------------------------------------------------------------------

PROFILE_DSV4_CSA_HCA = ModelProfile(
    name="dsv4_csa_hca",
    anchor_kernel="mhc_pre_big_fuse",
    blocks_per_layer=2,
    half_labels=["attn", "ffn"],
    category_rules=_DSV4_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES + _DSV4_SIMPLIFY_RULES,
    default_num_layers=1,
)

# Fused boundaries differ across V4.1 branches and batch sizes. Require a
# caller-verified once-per-layer anchor instead of assuming two mHC post calls.
PROFILE_DSV41 = ModelProfile(
    name="dsv41",
    anchor_kernel=None,
    blocks_per_layer=1,
    half_labels=["full"],
    category_rules=_DSV4_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES + _DSV4_SIMPLIFY_RULES,
)

PROFILE_DSV3_MLA = ModelProfile(
    name="dsv3_mla",
    anchor_kernel=None,
    blocks_per_layer=1,
    half_labels=["full"],
    category_rules=_DSV3_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES + _DSV3_SIMPLIFY_RULES,
    default_num_layers=1,
)

PROFILE_GENERIC = ModelProfile(
    name="generic",
    anchor_kernel=None,
    blocks_per_layer=1,
    half_labels=["full"],
    category_rules=_UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES,
    default_num_layers=1,
)


# ---------------------------------------------------------------------------
# Profile registry & inference
# ---------------------------------------------------------------------------

BUILTIN_PROFILES: Dict[str, ModelProfile] = {
    "dsv4_csa_hca": PROFILE_DSV4_CSA_HCA,
    "dsv41": PROFILE_DSV41,
    "dsv3_mla": PROFILE_DSV3_MLA,
    "generic": PROFILE_GENERIC,
}


def get_profile(name: str) -> ModelProfile:
    """Look up a built-in profile by name."""
    if name not in BUILTIN_PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {', '.join(BUILTIN_PROFILES)}"
        )
    return BUILTIN_PROFILES[name]


def normalize_compress_ratios(
    config: dict, num_layers: Optional[int] = None
) -> List[int]:
    """Return per-hidden-layer compress ratios, validating known config shapes.

    Some DeepSeek-V4 configs publish one extra ratio for next-token-prediction
    layers. The pipeline analyzers operate on transformer hidden layers only,
    so that trailing nextn ratio is intentionally excluded instead of being
    silently sliced by callers.
    """
    config = normalize_model_config(config)
    ratios = list(config.get("compress_ratios") or [])
    if not ratios:
        return []

    n_layers = num_layers or config.get("num_hidden_layers")
    if not n_layers:
        return ratios

    if len(ratios) == n_layers:
        return ratios

    nextn_layers = config.get("num_nextn_predict_layers", 0) or 0
    if nextn_layers and len(ratios) == n_layers + nextn_layers:
        return ratios[:n_layers]

    raise ValueError(
        "compress_ratios length mismatch: "
        f"got {len(ratios)}, expected num_hidden_layers={n_layers}"
        + (f" or + num_nextn_predict_layers={nextn_layers}" if nextn_layers else "")
    )


def normalize_model_config(raw: dict) -> dict:
    """Flatten public text configs and normalize HF dimension aliases.

    Preserve the wrapper identity as outer_model_type for profile inference.
    Derived shared width is total width, not width per shared expert.
    """
    config = dict(raw)
    for key in ("text_config", "language_config", "llm_config"):
        inner = raw.get(key)
        if isinstance(inner, dict):
            config["outer_model_type"] = raw.get("model_type", "")
            config.update(inner)
            break
    if not config.get("num_hidden_layers"):
        config["num_hidden_layers"] = config.get("num_layers") or len(
            config.get("layers_block_type")
            or config.get("hybrid_override_pattern")
            or []
        )
    has_total_shared_width = "shared_expert_intermediate_size" in config
    aliases = {
        "num_experts": ("n_routed_experts", "num_local_experts", "moe_num_experts"),
        "num_experts_per_tok": (
            "num_experts_per_token",
            "moe_topk",
            "moe_top_k",
            "top_k_experts",
        ),
        "num_shared_experts": ("n_shared_experts",),
        "routed_expert_intermediate_size": (
            "moe_intermediate_size",
            "expert_ffn_hidden_size",
        ),
        "shared_expert_intermediate_size": (
            "shared_intermediate_size",
            "moe_shared_expert_intermediate_size",
            "share_expert_dim",
        ),
    }
    for target, sources in aliases.items():
        if target not in config:
            for source in sources:
                if source in config:
                    config[target] = config[source]
                    break
    if "shared_expert_intermediate_size" not in config and "n_shared_experts" in config:
        config["shared_expert_intermediate_size"] = config.get(
            "moe_intermediate_size", 0
        ) * (config.get("n_shared_experts") or 0)
    linear_cfg = config.get("linear_attn_config") or {}
    for target, source in (
        ("kda_num_heads", "num_heads"),
        ("kda_head_dim", "head_dim"),
        ("kda_short_conv_kernel_size", "short_conv_kernel_size"),
    ):
        if source in linear_cfg:
            config.setdefault(target, linear_cfg[source])
    if "kda_layers" in linear_cfg:
        if config.get("model_type") == "kimi_linear":
            config.setdefault("kda_layers_1_indexed", linear_cfg["kda_layers"])
        else:
            config.setdefault("kda_layers", linear_cfg["kda_layers"])
    # HF Nemotron's explicit shared width is per expert; canonical index
    # entries already carry the total width and must not be multiplied again.
    if not has_total_shared_width and "moe_shared_expert_intermediate_size" in config:
        config["shared_expert_intermediate_size"] = config[
            "moe_shared_expert_intermediate_size"
        ] * (config.get("num_shared_experts") or 0)
    config["moe"] = bool(config.get("moe") or (config.get("num_experts") or 0) > 0)
    config["mhc"] = bool(
        config.get("mhc") or config.get("hc_mult") or config.get("hc_count")
    )
    return config


def infer_profile(config: dict) -> ModelProfile:
    """Dispatch model identity before structural MLA/compression fallbacks."""
    config = normalize_model_config(config)
    types = {config.get("model_type", ""), config.get("outer_model_type", "")}
    dispatch = [
        (("deepseek_v41", "deepseek_v41_text"), "dsv41"),
        (("deepseek_v4",), "dsv4_csa_hca"),
        (("kimi_k3",), "kimi_k3"),
        (("glm5_next", "glm5_next_text"), "glm5_next"),
        (("hy_v4",), "hy_v4"),
        (("qwen4_exp", "qwen4_exp_text"), "qwen4_exp"),
        (("nemotron_h",), "nemotron_h"),
        (("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"), "gemma4"),
        (("glm_moe_dsa", "deepseek_v32"), "mla_dsa"),
    ]
    if "kv_source_layer_ids" in config:
        return get_profile("dsv41")
    if "kimi_linear" in types and config.get("attn_res_block_size"):
        return get_profile("kimi_k3")
    for identities, profile in dispatch:
        if types.intersection(identities):
            return get_profile(profile)
    if any("longcatflash" in a.lower() for a in config.get("architectures", [])):
        return get_profile("longcat_flash")
    if config.get("compress_ratios"):
        return PROFILE_DSV4_CSA_HCA
    if config.get("index_topk") and config.get("kv_lora_rank"):
        return get_profile("mla_dsa")
    # Hybrid SWA/MLA is not a homogeneous V3 stack.
    if config.get("layer_types") or "dots3_note" in types:
        return get_profile("generic_addnorm")
    if config.get("kv_lora_rank", 0) > 0:
        return PROFILE_DSV3_MLA
    return (
        get_profile("generic_addnorm")
        if config.get("num_hidden_layers")
        else PROFILE_GENERIC
    )


_NEW_CATEGORY_RULES = [
    (
        "  Fused mHC",
        "mhc_fused",
        _any_sub(
            "mhc_fused_post_pre_fma",
            "mhc_fused_tilelang",
            "all_reduce_mhc",
            "mega_mhc",
            "mhc_post_split_h",
        ),
    ),
    (
        "  HC combine",
        "mhc_combine",
        _any_sub("hc_combine", "hc_mix", "hc_head_fuse", "ihc_", "attn_res"),
    ),
    ("  HC prenorm", "mhc_pre_gemm", _sub("hc_prenorm_gemm")),
    ("  Indexer Q preparation", "indexer", _sub("fused_q_indexer_rope_hadamard")),
    ("  Indexer top-k", "topk", _sub("topk_transform")),
    (
        "  Linear/SSM",
        "hybrid_linear",
        _any_sub("kda", "fused_recurrent", "chunk_gla", "causal_conv1d", "sconv"),
    ),
    ("  Activation", "activation", _sub("situ_and_mul")),
    ("  QKV norm", "rmsnorm", _any_sub("gemma_qkv_rmsnorm", "grouped_gemma_rmsnorm")),
    ("  Sparse attention", "mla", _any_sub("sparse_attn", "qsa", "fmhaSm100")),
    ("  Router", "moe_gate", _any_sub("inkling_gate", "gemma4_routing")),
]
for _name, _anchor, _bpl in [
    ("mla_dsa", None, 2),
    ("kimi_k3", None, 2),
    ("glm5_next", "mhc_pre_big_fuse", 2),
    ("hy_v4", None, 2),
    ("qwen4_exp", None, 2),
    ("nemotron_h", None, 1),
    ("longcat_flash", None, 4),
    ("gemma4", None, 1),
    ("generic_addnorm", None, 2),
]:
    BUILTIN_PROFILES[_name] = ModelProfile(
        _name,
        _anchor,
        _bpl,
        ["attn", "ffn"] if _bpl == 2 else [f"part{i}" for i in range(_bpl)],
        _NEW_CATEGORY_RULES + _DSV4_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
        _UNIVERSAL_SIMPLIFY_RULES,
    )
for _profile in (PROFILE_DSV4_CSA_HCA, PROFILE_DSV41):
    _profile.category_rules = _NEW_CATEGORY_RULES + _profile.category_rules


def layer_label(layer_id, compress_ratio, num_layers, num_hash_layers=0):
    """Composable work/routing flags; hash routing belongs to the prefix."""
    flags = []
    if layer_id == 0:
        flags.append("FIRST")
    if layer_id == num_layers - 1:
        flags.append("FINAL")
    flags.append(
        {0: "SWA_ONLY", 4: "CSA_C4", 128: "HCA_C128", 2: "V41_C2", 1: "V41_C1"}.get(
            compress_ratio, "GENERIC"
        )
    )
    if layer_id < num_hash_layers:
        flags.append("HASH")
    return "+".join(flags)


RESIDUAL_ANCHORS = (
    "FusedAddRMSNorm",
    "fused_add_rms_norm",
    "fused_add_rmsnorm",
    "allreduce_fusion",
)


def choose_anchor(gpu, profile):
    if profile.anchor_kernel:
        return profile.anchor_kernel
    if profile.name in ("generic", "generic_addnorm", "mla_dsa", "dsv3_mla"):
        for candidate in RESIDUAL_ANCHORS:
            if sum(candidate.lower() in e.get("name", "").lower() for e in gpu) >= 4:
                return candidate
    raise ValueError(
        "Supply a verified once-per-layer or sublayer --anchor-kernel and matching --config/--num-layers; phase and backend must be checked first"
    )


def select_gpu(events, pid=None, device=None):
    gpu = [e for e in events if "kernel" in e.get("cat", "").split(",")]
    if pid is not None:
        gpu = [e for e in gpu if str(e.get("pid")) == str(pid)]
    if device is not None:
        gpu = [e for e in gpu if str(e.get("args", {}).get("device")) == str(device)]
    if len({(e.get("pid"), e.get("args", {}).get("device")) for e in gpu}) > 1:
        raise ValueError("GPU kernels span ranks/devices; select --pid and --device")
    return sorted(gpu, key=lambda e: e.get("ts", 0))


def configure_anchor_profile(profile, anchor, blocks_per_layer=None):
    bpl = blocks_per_layer or (
        2
        if any(x.lower() in anchor.lower() for x in RESIDUAL_ANCHORS)
        else profile.blocks_per_layer
    )
    if bpl < 1:
        raise ValueError("--blocks-per-layer must be positive")
    return replace(
        profile,
        blocks_per_layer=bpl,
        half_labels=["attn", "ffn"] if bpl == 2 else [f"part{i}" for i in range(bpl)],
    )


def checked_anchor_indices(gpu, anchor, num_layers, bpl, offset=0):
    indices = [
        i for i, e in enumerate(gpu) if anchor.lower() in e.get("name", "").lower()
    ]
    if offset < 0 or offset >= len(indices):
        raise ValueError("--anchor-offset is outside the matching anchors")
    indices = indices[offset:]
    residue = len(indices) % (num_layers * bpl)
    if residue:
        raise ValueError(
            f"Anchor count {len(indices)} has residue {residue} for {num_layers} layers × {bpl} blocks; refuse automatic pass splitting"
        )
    # Final complete pass has no next-pass anchor: use end-of-selected-GPU sentinel.
    return indices + [len(gpu)]
