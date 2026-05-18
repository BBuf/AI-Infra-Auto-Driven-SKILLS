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

from dataclasses import dataclass, field
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
    return lambda n: any(s in n for s in substrings)


def _sub(s: str) -> Callable[[str], bool]:
    """Return a rule that matches a single substring."""
    return lambda n: s in n


# ---------------------------------------------------------------------------
# Universal (framework-level) rules
# ---------------------------------------------------------------------------

_UNIVERSAL_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    ("● NCCL AllReduce", "allreduce",     _sub("AllReduce")),
    ("  RMSNorm",        "rmsnorm",       _any_sub("RMSNorm", "rms_normalize")),
    ("  FP8 Quant",      "quant",         lambda n: "quant" in n.lower() or "Quant" in n),
    ("  TopK",           "topk",          lambda n: "topk" in n.lower()),
    ("  GEMM fp8",       "gemm_fp8",      _sub("deep_gemm")),
    ("  GEMM bf16",      "gemm_bf16",     _sub("nvjet")),
    ("  GEMM f32",       "gemm_f32",      _sub("sm80_xmma")),
    ("  Activation",     "activation",    _any_sub("silu_mul_clamp", "act_and_mul")),
    ("  RadixSort",      "radixsort",     lambda n: "radix_sort" in n.lower() or "RadixSort" in n),
]

_UNIVERSAL_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    ("void (anonymous namespace)::", ""),
    ("void at::native::", ""),
    ("void flashinfer::", ""),
    ("void deep_gemm::sm90_fp8_gemm_1d2d_impl", "deep_gemm::sm90_fp8_gemm_1d2d"),
    ("void fast_hadamard_transform_kernel", "fast_hadamard_transform_kernel"),
    ("void per_token_group_quant_8bit_kernel", "per_token_group_quant_8bit_kernel"),
    ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)", "ncclAllReduce_bf16_RING_LL"),
    ("norm::RMSNormKernel", "RMSNormKernel"),
]


# ---------------------------------------------------------------------------
# DeepSeek-V4 CSA/HCA profile
# ---------------------------------------------------------------------------

_DSV4_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    # Model-specific rules (evaluated before universal rules)
    ("★ MLA Attention",  "mla",           _sub("flash_fwd_splitkv_mla")),
    ("★ MoE Fused",      "moe",           _sub("fused_moe_kernel")),
    ("  Hadamard Xform", "hadamard",      lambda n: "hadamard" in n.lower()),
    ("  Indexer Cache",  "indexer",       lambda n: "indexer" in n.lower()),
    ("  Paged MQA",      "paged_mqa",     _sub("paged_mqa_logits")),
    ("  MLA Metadata",   "mla_metadata",  _sub("get_mla_metadata")),
    ("  C4 Prefill",     "c4_prefill",    _sub("c4_prefill")),
    ("  C128 Prefill",   "c128_prefill",  _sub("c128_prefill")),
    ("  RoPE",           "rope",          _any_sub("deepseek_rope", "fused_norm_rope")),
    ("  MHC Pre GEMM",   "mhc_pre_gemm",  _sub("mhc_pre_gemm_sqrsum")),
    ("  MHC Pre Fuse",   "mhc_pre_fuse",  _sub("mhc_pre_big_fuse")),
    ("  MHC Post",       "mhc_post",      _sub("mhc_post_tilelang")),
    ("  MoE Gate",       "moe_gate",      _sub("moe_fused_gate")),
    ("  MoE Align",      "moe_align",     _sub("moe_align_block")),
    ("  MoE Sort",       "moe_sort",      _sub("count_and_sort")),
    ("  MLA Cache Store", "mla_cache_store", _sub("fused_store_flashmla_cache")),
    ("  Indexer Store",  "indexer_store", _sub("fused_store_indexer_cache")),
]

_DSV4_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    ("void deep_gemm::sm90_fp8_paged_mqa_logits", "deep_gemm::sm90_fp8_paged_mqa_logits"),
    ("void deep_gemm::sched::smxx_paged_mqa_logits_metadata", "deep_gemm::paged_mqa_logits_metadata"),
    ("void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel", "flash_fwd_splitkv_mla_fp8_sparse"),
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
    ("★ MLA Attention",  "mla",           _sub("flash_fwd_splitkv_mla")),
    ("★ MoE Fused",      "moe",           _sub("fused_moe_kernel")),
    ("  MLA Metadata",   "mla_metadata",  _sub("get_mla_metadata")),
    ("  MLA Combine",    "mla_combine",   _sub("flash_fwd_mla_combine")),
    ("  RoPE",           "rope",          _any_sub("deepseek_rope", "fused_norm_rope")),
    ("  MoE Gate",       "moe_gate",      _sub("moe_fused_gate")),
    ("  MoE Align",      "moe_align",     _sub("moe_align_block")),
    ("  MoE Sort",       "moe_sort",      _sub("count_and_sort")),
    ("  MLA Cache Store", "mla_cache_store", _sub("fused_store_flashmla_cache")),
]

_DSV3_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    ("void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel", "flash_fwd_splitkv_mla_fp8_sparse"),
    ("void smxx::decode::flash_fwd_mla_combine_kernel", "flash_fwd_mla_combine"),
    ("void smxx::decode::get_mla_metadata_kernel", "get_mla_metadata"),
    ("fused_moe_kernel", "fused_moe_kernel"),
]


# ---------------------------------------------------------------------------
# Built-in profile instances
# ---------------------------------------------------------------------------

PROFILE_DSV4_CSA_HCA = ModelProfile(
    name="dsv4_csa_hca",
    anchor_kernel="mhc_post_tilelang",
    blocks_per_layer=2,
    half_labels=["attn", "ffn"],
    category_rules=_DSV4_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES + _DSV4_SIMPLIFY_RULES,
    default_num_layers=43,
)

PROFILE_DSV3_MLA = ModelProfile(
    name="dsv3_mla",
    anchor_kernel="flash_fwd_mla_combine",
    blocks_per_layer=1,
    half_labels=["full"],
    category_rules=_DSV3_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES + _DSV3_SIMPLIFY_RULES,
    default_num_layers=61,
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


def infer_profile(config: dict) -> ModelProfile:
    """Auto-detect the model profile from a config.json dict.

    Priority:
      1. Has non-empty ``compress_ratios`` → dsv4_csa_hca
      2. Has ``kv_lora_rank > 0`` → dsv3_mla
      3. Otherwise → generic
    """
    cr = config.get("compress_ratios", [])
    if cr:
        return PROFILE_DSV4_CSA_HCA

    kv_lora_rank = config.get("kv_lora_rank", 0)
    if kv_lora_rank and kv_lora_rank > 0:
        return PROFILE_DSV3_MLA

    return PROFILE_GENERIC
