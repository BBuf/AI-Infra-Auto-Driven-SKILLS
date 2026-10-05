#!/usr/bin/env python3
"""Model Compute Simulator — construct architecture, execution flow, tensor dims, FLOPs, MFU.

Usage:
  # List known models
  python3 model_compute_simulator.py --list-models

  # List known GPUs
  python3 model_compute_simulator.py --list-gpus

  # Simulate decode batch=1
  python3 model_compute_simulator.py "DeepSeek-V4-Flash" \
    --batch-size 1 --seq-len 1 --tp 8 --dp 1 --ep 8 --gpu b200 --dtype bf16

  # With MFU
  python3 model_compute_simulator.py "DeepSeek-V4-Flash" \
    --batch-size 1 --seq-len 1 --tp 8 --dp 1 --ep 8 --gpu b200 --dtype bf16 \
    --measured-ms 15.0
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config_normalization import normalize_model_config, profile_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(SCRIPT_DIR, "..", "references")
CONFIG_INDEX = os.path.join(REF_DIR, "model-config-index.json")
GPU_SPECS = os.path.join(REF_DIR, "gpu-specs.json")

# ---------------------------------------------------------------------------
# Alias map — fuzzy match user-facing names to config keys
# ---------------------------------------------------------------------------
ALIAS_MAP = {
    "deepseek-v3": "deepseek-v3",
    "deepseek-v4": "deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-flash-base": "deepseek-v4-flash-base",
    "deepseek-v4-base": "deepseek-v4-flash-base",
    "qwen3-235b": "qwen3-235b-a22b",
    "qwen3-235b-a22b": "qwen3-235b-a22b",
    "kimi-k2": "kimi-k2",
    "kimi-k2.5": "kimi-k2.5",
    "minimax-m2": "minimax-m2",
    "minimax-m3": "minimax-m3",
    "minimax-m3-mxfp8": "minimax-m3",
    "minimaxai/minimax-m3": "minimax-m3",
    "minimaxai/minimax-m3-mxfp8": "minimax-m3",
    "qwen3.6": "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b": "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-fp8": "qwen3.6-35b-a3b",
    "qwen/qwen3.6-35b-a3b-fp8": "qwen3.6-35b-a3b",
    "qwen3.8": "qwen3.8-27b",
    "qwen3.8-27b": "qwen3.8-27b",
    "qwen3.8-27b-fp8": "qwen3.8-27b",
    "qwen/qwen3.8-27b": "qwen3.8-27b",
    "qwen/qwen3.8-27b-fp8": "qwen3.8-27b",
    "glm-5": "glm-5",
}

GPU_ALIAS = {
    "h20": "h20",
    "h20-sxm": "h20",
    "h100": "h100-sxm-80gb",
    "h100-sxm": "h100-sxm-80gb",
    "h100-sxm5": "h100-sxm-80gb",
    "h100-80gb": "h100-sxm-80gb",
    "h100-sxm-80gb": "h100-sxm-80gb",
    "nvidia-h100": "h100-sxm-80gb",
    "h200": "h200-sxm-141gb",
    "h200-sxm": "h200-sxm-141gb",
    "h200-141gb": "h200-sxm-141gb",
    "h200-sxm-141gb": "h200-sxm-141gb",
    "nvidia-h200": "h200-sxm-141gb",
    "b200": "b200-sxm-180gb",
    "b200-sxm": "b200-sxm-180gb",
    "b200-180gb": "b200-sxm-180gb",
    "b200-sxm-180gb": "b200-sxm-180gb",
    "nvidia-b200": "b200-sxm-180gb",
    "b300": "b300-sxm-288gb",
    "hgx-b300": "b300-sxm-288gb",
    "rtx-pro-6000": "rtx-pro-6000-blackwell-server",
    "rtx6000-pro": "rtx-pro-6000-blackwell-server",
    "pro6000": "rtx-pro-6000-blackwell-server",
    "gb10": "dgx-spark-gb10",
    "spark": "dgx-spark-gb10",
    "dgx-spark": "dgx-spark-gb10",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Op:
    name: str
    flops: int  # per-token FLOPs (for one token)
    shape_in: str
    shape_out: str
    category: str  # "attention" | "moe" | "ffn" | "embed" | "residual" | "norm" | "mhc"


@dataclass
class LayerResult:
    layer_idx: int
    ops: List[Op] = field(default_factory=list)
    attention_flops: int = 0
    moe_ffn_flops: int = 0


@dataclass
class SimResult:
    model: str
    config_source: str
    batch_size: int
    seq_len: int
    tp: int
    dp: int
    ep: int
    gpu: str
    dtype: str
    context_len: Optional[int] = None
    all_token_logits: bool = False
    epilogue_flops: int = 0
    layers: List[LayerResult] = field(default_factory=list)
    total_flops: int = 0
    embed_flops: int = 0
    measured_ms: Optional[float] = None
    mfu_pct: Optional[float] = None
    per_layer_mfu_pct: Optional[float] = None
    per_op_mfu: list = field(default_factory=list)  # per-operator MFU details
    kernel_ms: Optional[dict] = None  # kernel-category → measured ms mapping
    kernel_flow: Optional[dict] = None  # raw measured kernel-flow evidence
    model_arch: dict = field(default_factory=dict)  # model architecture summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_json_argument(value: str) -> dict:
    """Load a JSON CLI argument from an inline document or ``@file``."""
    if value.startswith("@"):
        return load_json(value[1:])
    return json.loads(value)


def measured_ms_from_kernel_detail(detail: dict, num_layers: int) -> float:
    """Convert one measured layer's kernel duration into full-model latency."""
    layer_us = float(detail.get("metadata", {}).get("total_dur_us", 0))
    if layer_us <= 0:
        raise ValueError("kernel detail metadata.total_dur_us must be positive")
    return layer_us / 1000.0 * num_layers


def resolve_model(name: str, config_index: dict) -> Optional[dict]:
    key = name.lower().strip().replace(" ", "-")
    key = ALIAS_MAP.get(key, key)
    cfg = config_index.get(key)
    if cfg is None:
        cfg = next(
            (
                v
                for v in config_index.values()
                if v.get("hf_config_source", "").lower() == key
            ),
            None,
        )
    return cfg


def peak_dtype_key(dtype):
    if dtype in ("fp4", "mxfp4", "nvfp4"):
        return "fp4_tflops"
    return "bf16_tflops" if dtype == "bf16" else "fp8_tflops"


def resolve_gpu(name: str, gpu_specs: dict) -> Optional[dict]:
    key = name.lower().strip()
    key = GPU_ALIAS.get(key, key)
    return gpu_specs.get(key)


def normalize_compress_ratios(cfg: dict, n_layers: Optional[int] = None) -> List[int]:
    """Return per-hidden-layer compress ratios with explicit shape checks."""
    ratios = list(cfg.get("compress_ratios") or [])
    if not ratios:
        return []

    n_layers = n_layers or cfg.get("num_hidden_layers")
    if not n_layers:
        return ratios

    if len(ratios) == n_layers:
        return ratios

    nextn_layers = cfg.get("num_nextn_predict_layers", 0) or 0
    if nextn_layers and len(ratios) == n_layers + nextn_layers:
        return ratios[:n_layers]

    raise ValueError(
        "compress_ratios length mismatch: "
        f"got {len(ratios)}, expected num_hidden_layers={n_layers}"
        + (f" or + num_nextn_predict_layers={nextn_layers}" if nextn_layers else "")
    )


def matmul_flops(m, n, k):
    """FLOPs for C = A @ B where A:[m,k], B:[k,n] => 2*m*n*k."""
    return 2 * m * n * k


def fmt_shape(*dims):
    return "[" + "×".join(str(d) for d in dims if d is not None) + "]"


# ---------------------------------------------------------------------------
# Build per-layer ops
# ---------------------------------------------------------------------------
def _layer_moe(cfg, i):
    active = bool(cfg.get("moe")) and i >= cfg.get("first_k_dense_replace", 0)
    for key in ("mlp_layer_types", "moe_layer_freq"):
        pattern = cfg.get(key)
        if isinstance(pattern, list):
            value = pattern[i]
            active = active and value not in (0, "dense", "mlp")
        elif isinstance(pattern, int) and pattern > 0:
            active = active and i % pattern == 0
    if cfg.get("moe_layers_enum") is not None:
        active = active and i in cfg["moe_layers_enum"]
    if cfg.get("dense_mlp_idx") is not None:
        active = active and i not in cfg["dense_mlp_idx"]
    return active


def attention_pairs(S, context_len=None, limit=None):
    """Causal query/KV pairs: context_len is total KV for decode, prefix for extend."""
    if S == 1:
        return min(context_len or 1, limit) if limit else (context_len or 1)
    prefix = context_len or 0
    total = sum(
        min(prefix + q, limit) if limit else prefix + q for q in range(1, S + 1)
    )
    return total


def build_layer_ops(cfg, B, S, tp, ep, compress_ratio=0, layer_idx=0, context_len=None):
    """Public-config operator template; full-model FLOPs before TP/EP estimates.

    Dense attention uses causal pairs; sparse/linear/SSM scalar work is an
    architectural estimate, not an instruction count or benchmark result.
    """
    cfg = normalize_model_config(cfg)
    if min(B, S, tp, ep) < 1 or (context_len is not None and context_len < 1):
        raise ValueError(
            "Batch, sequence, parallelism and provided context must be positive"
        )
    H, T = cfg["hidden_size"], B * S
    nh = cfg["num_attention_heads"]
    nk = cfg.get("num_key_value_heads", nh)
    dh = cfg.get("head_dim", H // nh)
    n = cfg["num_hidden_layers"]
    lt = cfg.get("layer_types") or []
    if lt and len(lt) not in (n, n + cfg.get("num_nextn_predict_layers", 0)):
        raise ValueError("layer_types length mismatch")
    typ = lt[layer_idx] if lt else ""
    family = cfg.get("attention_type") or "gqa"
    if not cfg.get("attention_type"):
        profile = profile_name(cfg)
        family = {
            "dsv4_csa_hca": "csa_hca",
            "dsv41": "csa_hca_v41",
            "mla_dsa": "mla_dsa",
            "dsv3_mla": "mla",
            "kimi_k3": "hybrid_kda_mla",
            "glm5_next": "hybrid_kda_mla_dsa",
            "nemotron_h": "hybrid_mamba2_gqa",
            "qwen4_exp": "hybrid_gated_delta_qsa",
        }.get(profile, "mla" if cfg.get("kv_lora_rank") else "gqa")
    attn = family
    if family == "hybrid_kda_mla":
        kda_layers = cfg.get("kda_layers_1_indexed", cfg.get("kda_layers", []))
        attn = "kda" if layer_idx + 1 in kda_layers else "mla"
    elif family == "hybrid_kda_mla_dsa":
        attn = (
            "kda"
            if typ == "linear_attention" or layer_idx in cfg.get("kda_layers", [])
            else "mla_dsa"
        )
    elif family in ("hybrid_swa_gqa", "hybrid_swa_gqa_relpos"):
        attn = "gqa_qk_norm"
    elif family == "hybrid_swa_mla":
        attn = "mla"
    elif family == "hybrid_gated_delta_qsa":
        attn = "gated_delta" if typ == "linear_attention" else "qsa"
    elif family in ("hybrid_gated_delta_gqa_qk_norm", "hybrid_gated_delta_gqa"):
        attn = "gated_delta" if typ == "linear_attention" else "gqa_qk_norm"
    elif typ == "linear_attention":
        attn = "gated_delta"
    elif typ == "deepseek_sparse_attention":
        attn = "mla_dsa"
    block = None
    pattern = cfg.get("layers_block_type") or cfg.get("hybrid_override_pattern")
    if family == "hybrid_mamba2_gqa" and pattern:
        block = pattern[layer_idx]
        block = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}.get(
            block, block
        )
        attn = (
            "mamba" if block == "mamba" else "gqa" if block == "attention" else "none"
        )
    is_moe = _layer_moe(cfg, layer_idx)
    if block:
        is_moe = block == "moe"
    window = cfg.get("sliding_window") if typ == "sliding_attention" else None
    if typ == "sliding_attention":
        nh = cfg.get("swa_num_attention_heads", nh)
        nk = cfg.get("swa_num_key_value_heads", nk)
        dh = cfg.get("swa_head_dim", dh)
    elif cfg.get("global_head_dim"):
        dh = cfg["global_head_dim"]
        nk = cfg.get("num_global_key_value_heads", nk)
    pairs = attention_pairs(S, context_len, window)
    ctx = (context_len or 1) if S == 1 else (context_len or 0) + S
    ops = []

    def add(name, flops, din, output_dim, category="attention"):
        ops.append(
            Op(
                name,
                int(flops),
                fmt_shape(B, S, din),
                fmt_shape(B, S, output_dim),
                category,
            )
        )

    def proj(name, din, output_dim, category="attention", rows=T):
        add(name, matmul_flops(rows, output_dim, din), din, output_dim, category)

    add("rmsnorm", 6 * T * H, H, H, "norm")
    if attn in ("mla", "mla_dsa", "dsa"):
        qr = cfg.get("q_lora_rank", 0)
        kr = (
            cfg.get("swa_kv_lora_rank", cfg.get("kv_lora_rank", 512))
            if window
            else cfg.get("kv_lora_rank", 512)
        )
        nope = (
            cfg.get("swa_qk_nope_head_dim", cfg.get("qk_nope_head_dim", dh))
            if window
            else cfg.get("qk_nope_head_dim", dh)
        )
        rope = cfg.get("qk_rope_head_dim", 0)
        vd = (
            cfg.get("swa_v_head_dim", cfg.get("v_head_dim", nope))
            if window
            else cfg.get("v_head_dim", nope)
        )
        if qr:
            proj("q_lora_down", H, qr)
            proj("q_lora_up", qr, nh * (nope + rope))
        else:
            proj("q_proj", H, nh * (nope + rope))
        proj("kv_compress", H, kr + rope)
        if attn in ("mla_dsa", "dsa"):
            ih, idim = cfg.get("index_n_heads", 32), cfg.get("index_head_dim", 128)
            index_types = cfg.get("indexer_types")
            if not index_types or index_types[layer_idx] != "shared":
                proj("indexer_proj", qr or H, ih * idim)
                proj("indexer_k_proj", H, idim)
                add(
                    "paged_mqa",
                    2 * B * ih * attention_pairs(S, context_len) * idim,
                    ih * idim,
                    ctx,
                )
                add("indexer_topk", 4 * T * ctx, ctx, cfg.get("index_topk", 2048))
            pairs = attention_pairs(S, context_len, cfg.get("index_topk", 2048))
        if S == 1:
            add("q_absorb_uk", 2 * T * nh * nope * kr, nh * nope, nh * kr)
            add("attn_score", 2 * B * nh * pairs * (kr + rope), nh * (kr + rope), ctx)
            add("attn_v", 2 * B * nh * pairs * kr, ctx, nh * kr)
            add("v_absorb_uv", 2 * T * nh * kr * vd, nh * kr, nh * vd)
        else:
            kv_rows = B * ctx
            proj("kv_expand", kr, nh * (nope + vd), rows=kv_rows)
            add(
                "attn_score",
                2 * B * nh * pairs * (nope + rope),
                nh * (nope + rope),
                ctx,
            )
            add("attn_v", 2 * B * nh * pairs * vd, ctx, nh * vd)
        proj("o_proj", nh * vd, H)
        if cfg.get("gated_mla"):
            proj("mla_output_gate", H, nh)
        if cfg.get("attention_gate_type") or cfg.get("head_wise_attn_gate"):
            proj("attention_gate", H, nh)
    elif attn in ("csa_hca", "csa_hca_v41"):
        qr = cfg.get("q_lora_rank", 0)
        if qr:
            proj("q_lora_down", H, qr)
            proj("q_lora_up", qr, nh * dh)
        else:
            proj("q_proj", H, nh * dh)
        proj("kv_compress", H, nk * dh)
        ratio = compress_ratio
        kv_src = ratio in (4, 128) or layer_idx in cfg.get("kv_source_layer_ids", [])
        idx_src = ratio == 4 or layer_idx in cfg.get("index_source_layer_ids", [])
        if ratio and kv_src:
            # KV compressor: content + gate projections, each H -> head_dim.
            proj("compressor_proj", H, 2 * dh)
        if idx_src:
            ih, idim = cfg.get("index_n_heads", 32), cfg.get("index_head_dim", 128)
            proj("indexer_proj", qr or H, ih * idim)
            if ratio == 4:
                add(
                    "hadamard",
                    T * ih * idim * (idim.bit_length() - 1),
                    ih * idim,
                    ih * idim,
                )
            indexed = max(1, (ctx + max(ratio, 1) - 1) // max(ratio, 1))
            add("paged_mqa", 2 * T * ih * indexed * idim, ih * idim, indexed)
            add("indexer_topk", 4 * T * indexed, indexed, cfg.get("index_topk", 512))
        win = min(ctx, cfg.get("sliding_window", 128))
        compressed = (ctx + max(ratio, 1) - 1) // max(ratio, 1) if ratio else 0
        if ratio in (1, 2, 4):
            compressed = min(compressed, cfg.get("index_topk", 512))
        effective = win + compressed
        # Sparse rows are budget estimates; no second, fictitious HCA branch.
        sparse_pairs = (
            T * effective if ratio else B * attention_pairs(S, context_len, win)
        )
        add(
            (
                "c4_attn"
                if ratio in (1, 2, 4)
                else "hca_attn_score" if ratio == 128 else "csa_attn_score"
            ),
            2 * nh * sparse_pairs * dh,
            nh * dh,
            effective,
        )
        add(
            (
                "c4_attn_v"
                if ratio in (1, 2, 4)
                else "hca_attn_v" if ratio == 128 else "csa_attn_v"
            ),
            2 * nh * sparse_pairs * dh,
            effective,
            nh * dh,
        )
        groups, rank = cfg.get("o_groups", 1), cfg.get("o_lora_rank", 0)
        if rank:
            proj("o_lora_down", nh * dh // groups, rank, rows=T * groups)
            proj("o_lora_up", groups * rank, H)
        else:
            proj("o_proj", nh * dh, H)
    elif attn in ("gated_delta", "kda"):
        kh = cfg.get("linear_num_key_heads", cfg.get("kda_num_heads", nh))
        vh = cfg.get("linear_num_value_heads", kh)
        kd = cfg.get("linear_key_head_dim", cfg.get("kda_head_dim", 128))
        vd = cfg.get("linear_value_head_dim", kd)
        key, value = kh * kd, vh * vd
        proj("linear_in_proj_qkvz", H, 2 * key + 2 * value)
        proj("linear_in_proj_ba", H, value if attn == "kda" else 2 * vh)
        conv = cfg.get(
            "linear_conv_kernel_dim",
            cfg.get("kda_conv_kernel_dim", cfg.get("conv_kernel", 4)),
        )
        add(
            "linear_conv1d",
            2 * T * (2 * key + value) * conv,
            2 * key + value,
            2 * key + value,
        )
        add(
            "kda_attention" if attn == "kda" else "gated_delta_attention",
            8 * T * vh * kd * vd,
            key,
            value,
        )
        add("linear_gated_norm", 6 * T * value, value, value)
        proj("linear_out_proj", value, H)
    elif attn == "mamba":
        heads, hd = cfg.get("mamba_num_heads", 128), cfg.get("mamba_head_dim", 64)
        state, groups = cfg.get("ssm_state_size", 128), cfg.get("mamba_n_groups", 8)
        inner = heads * hd
        conv_dim = inner + 2 * groups * state
        proj("mamba_in_proj", H, 2 * inner + 2 * groups * state + heads)
        add(
            "mamba_conv1d",
            2 * T * conv_dim * cfg.get("conv_kernel", 4),
            conv_dim,
            conv_dim,
        )
        add("mamba_ssd_scan", 6 * T * inner * state, conv_dim, inner)
        proj("mamba_out_proj", inner, H)
    elif attn in ("gqa", "gqa_qk_norm", "qsa", "hybrid_mamba2_gqa"):
        proj("q_proj", H, nh * dh)
        proj("kv_proj", H, (1 if cfg.get("attention_k_eq_v") else 2) * nk * dh)
        if attn == "gqa_qk_norm":
            add("qk_norm", 6 * T * (nh + nk) * dh, (nh + nk) * dh, (nh + nk) * dh)
        sparse = cfg.get("sparse_attention_freq")
        is_sparse = bool(sparse and sparse[layer_idx]) or attn == "qsa"
        if is_sparse:
            ih = cfg.get(
                "sparse_num_index_heads",
                cfg.get("indexer_n_heads", cfg.get("index_n_heads", 4)),
            )
            idim = cfg.get(
                "sparse_index_dim",
                cfg.get("indexer_head_dim", cfg.get("index_head_dim", 128)),
            )
            disabled = cfg.get("sparse_disable_index_value")
            disable_v = bool(disabled and disabled[layer_idx])
            proj(
                "sparse_index_qk_proj" if disable_v else "sparse_index_qkv_proj",
                H,
                (ih + 1 + (not disable_v)) * idim,
            )
            add(
                "sparse_index_topk",
                2 * B * ih * attention_pairs(S, context_len) * idim,
                ih * idim,
                ctx,
            )
            budget = cfg.get(
                "sparse_topk", cfg.get("indexer_budget", cfg.get("index_topk", 2048))
            )
            pairs = attention_pairs(S, context_len, budget)
        add(
            "sparse_attn_score" if is_sparse else "attn_score",
            2 * B * nh * pairs * dh,
            nh * dh,
            ctx,
        )
        add(
            "sparse_attn_v" if is_sparse else "attn_v",
            2 * B * nh * pairs * dh,
            ctx,
            nh * dh,
        )
        proj("o_proj", nh * dh, H)
        if cfg.get("head_wise_attn_gate"):
            proj("attention_gate", H, nh)
        if cfg.get("d_rel"):
            add(
                "relative_position_bias",
                2 * B * nh * pairs * cfg["d_rel"],
                nh * cfg["d_rel"],
                ctx,
            )
    elif attn != "none":
        raise ValueError(f"Unknown attention_type: {attn}")
    if block not in ("mamba", "attention", "moe", "mlp"):
        add("rmsnorm", 6 * T * H, H, H, "norm")
        add("residual_add", T * H, H, H, "residual")
    if block not in ("mamba", "attention"):
        if is_moe:
            k, experts = cfg["num_experts_per_tok"], cfg["num_experts"]
            if layer_idx < cfg.get("num_hash_layers", 0):
                add("hash_route", 0, H, k, "moe")
            else:
                proj("router", H, experts + cfg.get("zero_expert_num", 0), "moe")
                add("topk", 4 * T * experts, experts, k, "moe")
            latent = (
                cfg.get("routed_expert_hidden_size", cfg.get("moe_latent_size", H)) or H
            )
            if latent != H:
                proj("routed_latent_down", H, latent, "moe")
                if cfg.get("latent_moe_use_norm"):
                    add("latent_rmsnorm", 6 * T * latent, latent, latent, "norm")
            inter = cfg.get("routed_expert_intermediate_size", 0)
            # Zero experts are identity slots; expected nonzero assignments are
            # unavailable without routing evidence, so count k as an upper bound.
            relu2 = cfg.get("mlp_hidden_act") == "relu2"
            add(
                "routed_experts_relu2" if relu2 else "routed_experts_swiglu",
                (4 if relu2 else 6) * T * k * latent * inter,
                k * latent,
                k * latent,
                "moe",
            )
            add(
                "activation",
                (2 if relu2 else 3) * T * k * inter,
                k * inter,
                k * inter,
                "moe",
            )
            if latent != H:
                proj("routed_latent_up", latent, H, "moe")
            shared = cfg.get("shared_expert_intermediate_size", 0)
            if cfg.get("num_shared_experts", 0) and shared:
                add("shared_experts_swiglu", 6 * T * H * shared, H, H, "moe")
            if cfg.get("parallel_dense_mlp"):
                add("ffn_swiglu", 6 * T * H * cfg["intermediate_size"], H, H, "ffn")
        else:
            inter = cfg.get("intermediate_size", 4 * H)
            relu2 = cfg.get("mlp_hidden_act") == "relu2"
            add(
                "ffn_relu2" if relu2 else "ffn_swiglu",
                (4 if relu2 else 6) * T * H * inter,
                H,
                H,
                "ffn",
            )
    add("residual_add", T * H, H, H, "residual")
    hc = cfg.get("hc_mult", cfg.get("hc_count", 0))
    if hc:
        if cfg.get("hc_lowrank"):
            rank = cfg["hc_lowrank"]
            add(
                "hc_lowrank_mix",
                4 * T * (hc * H * rank + rank * hc * hc),
                hc * H,
                hc * hc,
                "mhc",
            )
        elif profile_name(cfg) == "hy_v4" or cfg.get("ihc"):
            add("ihc_pre", 4 * T * hc * H * (2 * hc), hc * H, 2 * hc, "mhc")
        else:
            mix = (2 + hc) * hc
            add("mhc_pre_gemm", 4 * T * hc * H * mix, hc * H, mix, "mhc")
            add(
                "mhc_sinkhorn",
                2 * T * cfg.get("hc_sinkhorn_iters", 20) * hc * hc * 4,
                hc * hc,
                hc * hc,
                "mhc",
            )
        add("mhc_post_tilelang", 4 * T * hc * H, hc * H, hc * H, "mhc")
    if cfg.get("attn_res_block_size"):
        bank = min(8, 2 + 2 * layer_idx // cfg["attn_res_block_size"])
        add("attn_res", 8 * T * bank * H + 12 * T * H, bank * H, H, "mhc")
    if cfg.get("use_sconv"):
        add("short_conv", 4 * T * H * cfg.get("sconv_kernel_size", 4), H, H)
    if cfg.get("sublayers_per_layer"):
        # LongCat: two attention and dense branches, one shortcut MoE branch.
        attention_ops = [op for op in ops if op.category == "attention"]
        ops.extend(attention_ops)
        add("ffn_swiglu", 12 * T * H * cfg["intermediate_size"], H, H, "ffn")
    return ops


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
def simulate(
    cfg: dict,
    model_name: str,
    B: int,
    S: int,
    tp: int,
    dp: int,
    ep: int,
    gpu_name: str,
    dtype: str,
    measured_ms: Optional[float] = None,
    context_len: Optional[int] = None,
    all_token_logits: bool = False,
) -> SimResult:
    """Run the full simulation."""
    gpu_specs = load_json(GPU_SPECS)
    gpu_info = resolve_gpu(gpu_name, gpu_specs) if gpu_name else None

    cfg = normalize_model_config(cfg)
    n_layers = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    V = cfg.get("vocab_size", 0)

    # Embedding
    embed_flops = matmul_flops(B * (S if all_token_logits else 1), H, V) if V > 0 else 0

    # Per-layer (with per-layer compress_ratio)
    compress_ratios = normalize_compress_ratios(cfg, n_layers)

    total_layer_flops = 0
    layers = []
    for i in range(n_layers):
        cr = compress_ratios[i] if compress_ratios and i < len(compress_ratios) else 0
        layer_ops = build_layer_ops(
            cfg, B, S, tp, ep, compress_ratio=cr, layer_idx=i, context_len=context_len
        )
        lr = LayerResult(layer_idx=i)
        lr.compress_ratio = cr
        for op in layer_ops:
            lr.ops.append(op)
            if op.category == "attention":
                lr.attention_flops += op.flops
            elif op.category in ("moe", "ffn"):
                lr.moe_ffn_flops += op.flops
        layer_total = sum(op.flops for op in lr.ops)
        total_layer_flops += layer_total
        layers.append(lr)

    # Final norm/HC head and attention-residual output are pass-level work.
    epilogue = 6 * B * H
    hc = cfg.get("hc_mult", cfg.get("hc_count", 0))
    if hc:
        epilogue += 2 * B * hc * H * hc + 2 * B * hc * H
    if cfg.get("attn_res_block_size"):
        epilogue += 4 * B * min(8, 2 + 2 * n_layers // cfg["attn_res_block_size"]) * H
    total_flops = embed_flops + total_layer_flops + epilogue

    # Model architecture summary
    model_arch = {
        "num_hidden_layers": n_layers,
        "hidden_size": H,
        "num_attention_heads": cfg.get("num_attention_heads", 0),
        "num_key_value_heads": cfg.get("num_key_value_heads", 0),
        "head_dim": cfg.get("head_dim", H // cfg.get("num_attention_heads", 1)),
        "attention_type": cfg.get("attention_type", "unknown"),
        "moe": cfg.get("moe", False),
        "num_experts": cfg.get("num_experts", 0) if cfg.get("moe", False) else 0,
        "num_experts_per_tok": (
            cfg.get("num_experts_per_tok", 0) if cfg.get("moe", False) else 0
        ),
        "num_shared_experts": (
            cfg.get("num_shared_experts", 0) if cfg.get("moe", False) else 0
        ),
        "mhc": cfg.get("mhc", False),
        "vocab_size": V,
        "q_lora_rank": cfg.get("q_lora_rank", 0),
        "o_lora_rank": cfg.get("o_lora_rank", 0),
    }

    # MFU
    mfu_pct = None
    per_layer_mfu_pct = None
    per_op_mfu = []  # per-operator MFU for single-layer analysis
    if measured_ms is not None and gpu_info is not None:
        dtype_key = peak_dtype_key(dtype)
        peak_tflops = gpu_info.get(dtype_key) or 0
        if not peak_tflops:
            raise ValueError(
                f"No verified dense {dtype} peak for {gpu_name}; MFU unavailable"
            )
        if peak_tflops > 0:
            # Per-GPU FLOPs: attention/shared/MHC split by TP, routed MoE split by EP
            def flops_per_gpu_for_ops(ops_list):
                total = 0
                for op in ops_list:
                    if op.category == "moe" and op.name.startswith("routed_experts"):
                        total += op.flops / ep if ep > 0 else op.flops
                    else:
                        total += op.flops / tp if tp > 0 else op.flops
                return total

            # Overall MFU: measured_ms = total forward pass
            embed_per_gpu = embed_flops / tp if tp > 0 else embed_flops
            # Weight each layer's FLOPs by its compress_ratio
            total_per_gpu = embed_per_gpu + epilogue / tp
            for lr in layers:
                total_per_gpu += flops_per_gpu_for_ops(lr.ops)
            theoretical_time_s = total_per_gpu / (peak_tflops * 1e12)
            measured_time_s = measured_ms / 1000.0
            mfu_pct = (theoretical_time_s / measured_time_s) * 100.0

            # Per-layer MFU (uniform layer-time assumption)
            avg_layer_per_gpu = total_per_gpu / n_layers if n_layers > 0 else 0
            per_layer_ms = measured_ms / n_layers
            per_layer_mfu_pct = (
                (avg_layer_per_gpu / (peak_tflops * 1e12))
                / (per_layer_ms / 1000.0)
                * 100.0
            )

            # Per-operator FLOPs proportion — show a representative HCA_C128 layer if available
            # otherwise show layer 0
            repr_layer = None
            if compress_ratios:
                for i, cr in enumerate(compress_ratios):
                    if cr == 128:
                        repr_layer = layers[i]
                        break
            if repr_layer is None and layers:
                repr_layer = layers[0]
            if repr_layer is not None:
                l_ops = repr_layer.ops
                layer_total_flops = sum(op.flops for op in l_ops)
                for op in l_ops:
                    if op.category == "moe" and op.name.startswith("routed_experts"):
                        per_gpu = op.flops / ep if ep > 0 else op.flops
                    else:
                        per_gpu = op.flops / tp if tp > 0 else op.flops
                    per_op_mfu.append(
                        {
                            "name": op.name,
                            "flops": op.flops,
                            "flops_per_gpu": per_gpu,
                            "theo_us": per_gpu / (peak_tflops * 1e6),
                            "pct_of_layer": (
                                op.flops / layer_total_flops * 100
                                if layer_total_flops > 0
                                else 0
                            ),
                            "category": op.category,
                            "measured_us": None,  # filled later from --kernel-ms
                        }
                    )

    return SimResult(
        model=model_name,
        config_source=cfg.get("hf_config_source", ""),
        batch_size=B,
        seq_len=S,
        tp=tp,
        dp=dp,
        ep=ep,
        gpu=gpu_name or "",
        dtype=dtype,
        context_len=context_len,
        all_token_logits=all_token_logits,
        epilogue_flops=epilogue,
        layers=layers,
        total_flops=total_flops,
        embed_flops=embed_flops,
        measured_ms=measured_ms,
        mfu_pct=mfu_pct,
        per_layer_mfu_pct=per_layer_mfu_pct,
        per_op_mfu=per_op_mfu,
        kernel_ms=None,
        model_arch=model_arch,
    )


# ---------------------------------------------------------------------------
# Kernel category → operator mapping
# ---------------------------------------------------------------------------
# Maps measured kernel categories to simulator operator categories.
# Used to map kernel durations to per-operator MFU.

KERNEL_TO_OP_CATEGORY = {
    "mla": ["attention"],  # flash_fwd_splitkv_mla
    "moe": ["moe"],  # fused_moe_kernel
    "allreduce": [],  # NCCL — no compute FLOPs
    "hadamard": ["attention"],  # Hadamard transform (part of C128 attention)
    "indexer": ["attention"],  # Indexer cache (part of C128 attention)
    "paged_mqa": ["attention"],  # Paged MQA (hash layer attention)
    "mhc": ["mhc"],  # MHC pre/post
    "gemm_fp8": ["attention", "moe", "mhc", "norm"],  # GEMM spans multiple categories
    "gemm_bf16": ["attention", "moe", "mhc"],
    "rmsnorm": ["norm"],
    "quant": ["norm"],  # FP8 quantization
    "topk": ["moe"],  # TopK routing
    "rope": ["attention"],
    "activation": ["moe"],  # silu_mul in MoE
    "other": [],
    # Aliases from layer_kernel_breakdown.py
    "mla_metadata": ["attention"],
    "mhc_pre_gemm": ["mhc"],
    "mhc_pre_fuse": ["mhc"],
    "mhc_post": ["mhc"],
    "c4_prefill": ["attention"],
    "c128_prefill": ["attention"],
    "moe_gate": ["moe"],
    "moe_align": ["moe"],
    "moe_sort": ["moe"],
    "mla_cache_store": ["attention"],
    "indexer_store": ["attention"],
    "gemm_f32": [],
}


def map_kernel_ms_to_ops(kernel_ms: dict, ops: list, tp: int, ep: int) -> list:
    """Map kernel-category measured durations to per-operator measured durations.

    Strategy: distribute each kernel category's measured time proportionally
    across matching operators based on their FLOPs share.
    """
    # Step 1: Group ops by category and compute FLOPs share within each category
    cat_flops = {}  # category → total FLOPs
    for op in ops:
        cat = op.category
        cat_flops[cat] = cat_flops.get(cat, 0) + op.flops

    # Step 2: For each kernel category, find which op categories it maps to
    # and distribute the measured time proportionally
    op_measured_us = {}  # op index → measured microseconds

    for kernel_cat, duration_ms in kernel_ms.items():
        duration_us = duration_ms * 1000  # convert ms to μs
        mapped_op_cats = KERNEL_TO_OP_CATEGORY.get(kernel_cat, [])

        if not mapped_op_cats:
            # No compute FLOPs (e.g. allreduce, other) — skip
            continue

        # Sum FLOPs across all ops in the mapped categories
        total_cat_flops = sum(cat_flops.get(c, 0) for c in mapped_op_cats)
        if total_cat_flops == 0:
            continue

        # Distribute measured time to each op proportionally
        for i, op in enumerate(ops):
            if op.category in mapped_op_cats:
                share = op.flops / total_cat_flops
                op_measured_us[i] = op_measured_us.get(i, 0) + duration_us * share

    return op_measured_us


# ---------------------------------------------------------------------------
# Kernel-detail → operator mapping (precise, from --kernel-detail JSON)
# ---------------------------------------------------------------------------
# Maps kernel category (from layer_kernel_breakdown.py --format json)
# to template operator names.
#
# Three types of mapping:
#   - list of op names: direct assignment (kernel time → these ops)
#   - None: generic GEMM, distribute to remaining ops by FLOPs share
#   - empty list: no compute FLOPs (overhead only)

# Kernel categories that internally use fp8 compute even when --dtype bf16 is specified.
# For these kernels, the MFU denominator should use fp8 peak FLOPS (2x bf16).
FP8_INTERNAL_KERNEL_CATEGORIES = {"moe", "gemm_fp8"}

KERNEL_DETAIL_DIRECT_MAP = {
    # Fused kernels: directly map to specific operator groups
    "mla": [
        "csa_attn_score",
        "csa_attn_v",
        "hca_attn_score",
        "hca_attn_v",
        "c4_attn",
        "c4_attn_v",
        "attn_score",
        "attn_v",
    ],  # flash attention compute
    "moe": [
        "routed_experts_swiglu",
        "routed_experts_relu2",
    ],  # fused MoE: gate+up+silu+down
    "mhc_post": ["mhc_post_tilelang"],
    "mhc_pre_gemm": ["mhc_pre_gemm", "hc_lowrank_mix", "ihc_pre"],
    "mhc_pre_fuse": ["mhc_sinkhorn"],
    "mhc": [
        "mhc_post_tilelang",
        "mhc_pre_gemm",
        "mhc_sinkhorn",
        "hc_lowrank_mix",
        "ihc_pre",
        "attn_res",
    ],  # fallback
    "mhc_fused": ["mhc_pre_gemm", "mhc_sinkhorn", "mhc_post_tilelang"],
    "mhc_combine": ["mhc_post_tilelang", "ihc_pre", "hc_lowrank_mix", "attn_res"],
    "hybrid_linear": [
        "kda_attention",
        "gated_delta_attention",
        "linear_conv1d",
        "mamba_ssd_scan",
        "mamba_conv1d",
    ],
    "rmsnorm": ["rmsnorm"],
    "topk": ["topk"],
    "moe_gate": ["router"],
    # Direct-match operators (have FLOPs in template)
    "hadamard": ["hadamard"],
    "indexer": [],  # trace kernel is fused_store_indexer_cache only; proj compute is in gemm_bf16
    "paged_mqa": ["paged_mqa"],
    "c4_prefill": ["c4_attn", "c4_attn_v"],
    "c128_prefill": ["hca_attn_score", "hca_attn_v"],
    "rope": ["rope"],
    "quant": ["quant"],
    "activation": ["activation"],
    "mla_cache_store": ["mla_cache_store"],
    # Generic GEMM: distribute by FLOPs to remaining ops
    "gemm_fp8": None,
    "gemm_bf16": None,
    # No compute FLOPs (overhead)
    "allreduce": [],
    "mla_metadata": [],
    "moe_align": [],
    "moe_sort": [],
    "indexer_store": [],
    "gemm_f32": [],
    "radixsort": [],
    "other": [],
}


def map_kernel_detail_to_ops(kernel_detail: dict, ops: list, tp: int, ep: int) -> dict:
    """Map per-kernel measured durations to per-operator measured durations.

    This is more precise than map_kernel_ms_to_ops() because:
    1. Direct-match kernels (mla, moe, mhc, rmsnorm, topk) assign time directly
       to their corresponding ops, not proportionally by FLOPs.
    2. Generic GEMM kernels (gemm_fp8, gemm_bf16) only distribute to ops
       NOT already covered by direct-match, reducing FLOPs-proportional error.

    Args:
        kernel_detail: dict from layer_kernel_breakdown --format json output,
                       with 'kernels' list and 'category_summary'.
        ops: list of Op objects from build_layer_ops().
        tp: tensor parallelism.
        ep: expert parallelism.

    Returns:
        dict with:
          op_measured_us: {op_index: measured_us}
          overhead_us: total duration of kernels with no compute FLOPs
          direct_matched: {kernel_category: [op_names]} for info
          gemm_distributed: [op_names] for info
    """
    cat_summary = kernel_detail.get("category_summary", {})

    # Build op name → list of indices mapping (same name may appear multiple times)
    op_name_to_indices = {}
    for i, op in enumerate(ops):
        op_name_to_indices.setdefault(op.name, []).append(i)

    # Step 1: Direct-match assignment
    op_measured_us = {}  # op_index → measured_us
    direct_matched = {}  # kernel_cat → [op_names]  (for reporting)
    assigned_op_indices = set()  # ops that already got measured time
    overhead_us = 0.0

    for cat_key, info in cat_summary.items():
        cat_dur_us = info.get("dur_us", info.get("dur", 0))
        mapping = KERNEL_DETAIL_DIRECT_MAP.get(cat_key)

        if mapping is None:
            # Generic GEMM — will handle in Step 2
            continue
        elif len(mapping) == 0:
            # No compute FLOPs — overhead
            overhead_us += cat_dur_us
            continue

        # Direct match: distribute duration proportionally by FLOPs
        # across all matching op instances
        matched_indices = []
        matched_names = []
        for op_name in mapping:
            if op_name in op_name_to_indices:
                indices = [
                    idx
                    for idx in op_name_to_indices[op_name]
                    if idx not in assigned_op_indices
                ]
                for idx in indices:
                    matched_indices.append(idx)
                if indices:
                    matched_names.append(op_name)

        if matched_indices:
            # Compute FLOPs share within the matched ops
            total_matched_flops = sum(ops[idx].flops for idx in matched_indices)
            if total_matched_flops > 0:
                for idx in matched_indices:
                    share = ops[idx].flops / total_matched_flops
                    op_measured_us[idx] = (
                        op_measured_us.get(idx, 0) + cat_dur_us * share
                    )
                    assigned_op_indices.add(idx)
            else:
                # Equal distribution as fallback
                per_op_dur = cat_dur_us / len(matched_indices)
                for idx in matched_indices:
                    op_measured_us[idx] = op_measured_us.get(idx, 0) + per_op_dur
                    assigned_op_indices.add(idx)
            direct_matched[cat_key] = matched_names

    # Step 2: Distribute generic GEMM time to remaining ops by FLOPs share
    gemm_distributed = []
    for cat_key in ("gemm_fp8", "gemm_bf16"):
        info = cat_summary.get(cat_key)
        if not info:
            continue
        cat_dur_us = info.get("dur_us", info.get("dur", 0))
        if cat_dur_us <= 0:
            continue

        # Find ops NOT yet assigned that are GEMM-like (attention, moe, mhc, norm, ffn)
        # and could be executed by this GEMM type
        gemm_categories = ["attention", "moe", "mhc", "norm", "ffn"]
        if cat_key == "gemm_fp8":
            # fp8 GEMM could serve any category
            eligible_cats = gemm_categories
        else:
            # bf16 GEMM typically serves attention, mhc
            eligible_cats = ["attention", "mhc", "ffn"]

        # Compute FLOPs share among unassigned eligible ops
        eligible_flops = 0
        for i, op in enumerate(ops):
            if i not in assigned_op_indices and op.category in eligible_cats:
                eligible_flops += op.flops

        if eligible_flops == 0:
            # All eligible ops already assigned; distribute to any unassigned
            for i, op in enumerate(ops):
                if i not in assigned_op_indices:
                    eligible_flops += op.flops
            if eligible_flops == 0:
                overhead_us += cat_dur_us
                continue

        for i, op in enumerate(ops):
            if i not in assigned_op_indices and op.category in eligible_cats:
                share = op.flops / eligible_flops
                op_measured_us[i] = op_measured_us.get(i, 0) + cat_dur_us * share
                assigned_op_indices.add(i)
                gemm_distributed.append(op.name)

    # Step 3: Any remaining unassigned ops get no measured time
    # (they will show as MFU=N/A in the output)

    return {
        "op_measured_us": op_measured_us,
        "overhead_us": overhead_us,
        "direct_matched": direct_matched,
        "gemm_distributed": gemm_distributed,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_flops(f: int) -> str:
    if f >= 1e12:
        return f"{f / 1e12:.3f} TFLOPs"
    if f >= 1e9:
        return f"{f / 1e9:.3f} GFLOPs"
    if f >= 1e6:
        return f"{f / 1e6:.3f} MFLOPs"
    return f"{f} FLOPs"


def format_text(result: SimResult, skip_compute_flow: bool = False) -> str:
    lines = []

    # ---- 1. Model Architecture Summary ----
    lines.append("=" * 70)
    lines.append("  MODEL ARCHITECTURE")
    lines.append("=" * 70)
    arch = result.model_arch
    lines.append(f"  Model:           {result.model}")
    lines.append(f"  Config source:   {result.config_source}")
    lines.append(f"  Layers:          {arch.get('num_hidden_layers', '?')}")
    lines.append(f"  Hidden size:     {arch.get('hidden_size', '?')}")
    lines.append(
        f"  Attention heads: {arch.get('num_attention_heads', '?')} (KV: {arch.get('num_key_value_heads', '?')})"
    )
    lines.append(f"  Head dim:        {arch.get('head_dim', '?')}")
    lines.append(f"  Attention type:  {arch.get('attention_type', '?')}")
    if arch.get("q_lora_rank", 0) > 0:
        lines.append(f"  Q LoRA rank:     {arch.get('q_lora_rank')}")
        lines.append(f"  O LoRA rank:     {arch.get('o_lora_rank')}")
    if arch.get("moe", False):
        lines.append(
            f"  MoE:             {arch.get('num_experts')} experts, top-{arch.get('num_experts_per_tok')}, {arch.get('num_shared_experts')} shared"
        )
    if arch.get("mhc", False):
        lines.append(f"  MHC:             enabled")
    lines.append(f"  Vocab size:      {arch.get('vocab_size', '?')}")

    # ---- 2. Serving Configuration ----
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  SERVING CONFIGURATION")
    lines.append(f"{'=' * 70}")
    lines.append(
        f"  B={result.batch_size}  S={result.seq_len}  context/prefix={result.context_len}  TP={result.tp}  DP={result.dp}  EP={result.ep}"
    )
    lines.append(f"  GPU={result.gpu}  dtype={result.dtype}")

    # Embedding
    if result.embed_flops > 0:
        lines.append(f"[LM head]  FLOPs: {fmt_flops(result.embed_flops)}")

    # Per-layer (skip when kernel-flow mode — redundant with kernel-flow table)
    if result.layers and not skip_compute_flow:
        # Find the representative layer that per_op_mfu is based on
        repr_idx = 0
        for i, lr in enumerate(result.layers):
            if getattr(lr, "compress_ratio", 0) == 128:
                repr_idx = i
                break
        l0 = result.layers[repr_idx]
        cr_val = getattr(l0, "compress_ratio", 0)
        cr_label = f" (compress_ratio={cr_val})" if cr_val else ""
        type_label = {0: "SWA_ONLY", 4: "CSA_C4", 128: "HCA_C128"}.get(cr_val, "")
        layer_type_str = f" [{type_label}]" if type_label else ""
        lines.append(f"\n--- Layer {repr_idx} (detail){layer_type_str}{cr_label} ---")
        attn_pct = (
            l0.attention_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100
        )
        moe_pct = l0.moe_ffn_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100
        lines.append(
            f"  Attention FLOPs: {fmt_flops(l0.attention_flops)}  ({attn_pct:.1f}%)"
        )
        lines.append(
            f"  MoE/FFN FLOPs:   {fmt_flops(l0.moe_ffn_flops)}  ({moe_pct:.1f}%)"
        )
        lines.append(f"  Per-op sequence:")
        for op in l0.ops:
            lines.append(
                f"    {op.name:30s}  FLOPs: {fmt_flops(op.flops):>16s}  {op.shape_in} -> {op.shape_out}"
            )

        # Summary for remaining layers
        n_layers = len(result.layers)
        if n_layers > 1:
            layer_total = sum(o.flops for o in l0.ops)
            lines.append(f"\n--- Layers 1-{n_layers - 1} (same as Layer 0) ---")
            lines.append(f"  Per-layer FLOPs: {fmt_flops(layer_total)}")

    # Total
    lines.append(f"\n{'='*70}")
    lines.append(f"  TOTAL MODEL FLOPs")
    lines.append(f"{'='*70}")
    lines.append(f"  Total (1 forward pass): {fmt_flops(result.total_flops)}")
    if result.embed_flops > 0:
        lines.append(f"  LM head:               {fmt_flops(result.embed_flops)}")
    lines.append(f"  Pass epilogue:         {fmt_flops(result.epilogue_flops)}")
    layer_total = result.total_flops - result.embed_flops - result.epilogue_flops
    lines.append(f"  Transformer layers:    {fmt_flops(layer_total)}")

    # MFU
    if result.mfu_pct is not None:
        lines.append(f"\n{'='*70}")
        lines.append(f"  MFU ANALYSIS")
        lines.append(f"{'='*70}")
        lines.append(f"  Measured latency: {result.measured_ms:.2f} ms")
        lines.append(f"  Overall MFU:      {result.mfu_pct:.2f}%")
        if result.per_layer_mfu_pct is not None:
            lines.append(f"  Per-layer MFU:    {result.per_layer_mfu_pct:.2f}%")
        # Dominant category
        if result.layers:
            l0 = result.layers[0]
            if l0.moe_ffn_flops > l0.attention_flops:
                lines.append(
                    f"  Dominant: MoE/FFN ({l0.moe_ffn_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100:.1f}%)"
                )
            else:
                lines.append(
                    f"  Dominant: Attention ({l0.attention_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100:.1f}%)"
                )
        if result.mfu_pct < 5:
            lines.append(f"  Note: decode B=1 is memory-bound; low MFU is expected.")

        # Per-operator MFU breakdown with kernel-measured times
        if result.per_op_mfu:
            has_kernel = (
                result.kernel_ms is not None
                or getattr(result, "_kernel_detail_meta", None) is not None
            )
            lines.append(
                f"\n  Per-operator MFU detail (TP={result.tp}, EP={result.ep}):"
            )
            if has_kernel:
                lines.append(
                    f"  {'Op':28s} {'Cat':>6s} {'FLOPs':>14s} {'Per-GPU':>14s} {'Theo(μs)':>10s} {'Meas(μs)':>10s} {'MFU%':>7s} {'Layer%':>8s}"
                )
            else:
                lines.append(
                    f"  {'Op':28s} {'Cat':>6s} {'FLOPs':>14s} {'Per-GPU':>14s} {'Theo(μs)':>10s} {'Layer%':>8s}"
                )
            lines.append(f"  {'-'*90}")
            for op in result.per_op_mfu:
                if (
                    has_kernel
                    and op.get("measured_us") is not None
                    and op["measured_us"] > 0
                ):
                    mfu_op = op["theo_us"] / op["measured_us"] * 100
                    lines.append(
                        f"  {op['name']:28s} {op['category']:>6s} {fmt_flops(op['flops']):>14s} {fmt_flops(op['flops_per_gpu']):>14s} {op['theo_us']:>9.1f} {op['measured_us']:>9.1f} {mfu_op:>6.1f}% {op['pct_of_layer']:>7.1f}%"
                    )
                else:
                    lines.append(
                        f"  {op['name']:28s} {op['category']:>6s} {fmt_flops(op['flops']):>14s} {fmt_flops(op['flops_per_gpu']):>14s} {op['theo_us']:>9.1f} {op['pct_of_layer']:>7.1f}%"
                    )

        # Kernel category mapping explanation
        kdm = getattr(result, "_kernel_detail_meta", None)
        if kdm is not None:
            # Precise kernel-detail mapping
            lines.append(
                f"\n  Kernel-detail → operator mapping (precise, from --kernel-detail):"
            )
            for kcat, op_names in kdm.get("direct_matched", {}).items():
                lines.append(f"    {kcat:20s} → {', '.join(op_names)} (direct)")
            if kdm.get("gemm_distributed"):
                lines.append(
                    f"    {'gemm_fp8/bf16':20s} → {', '.join(kdm['gemm_distributed'])} (FLOPs-proportional)"
                )
            overhead = kdm.get("overhead_us", 0)
            if overhead > 0:
                lines.append(
                    f"    {'overhead':20s} {overhead/1000:8.3f} ms (allreduce/quant/rope/etc.)"
                )
            # Unassigned ops
            if result.per_op_mfu:
                unassigned = [
                    op["name"]
                    for op in result.per_op_mfu
                    if op.get("measured_us") is None
                ]
                if unassigned:
                    lines.append(
                        f"    {'unassigned':20s} {', '.join(unassigned)} (no kernel match)"
                    )
        elif result.kernel_ms:
            lines.append(f"\n  Kernel category → operator mapping:")
            for kcat, dur_ms in sorted(result.kernel_ms.items(), key=lambda x: -x[1]):
                mapped = KERNEL_TO_OP_CATEGORY.get(kcat, [])
                lines.append(
                    f"    {kcat:20s} {dur_ms:8.3f} ms → {', '.join(mapped) if mapped else '(no compute FLOPs)'}"
                )

    # One-line summary
    if result.mfu_pct is not None and result.layers:
        l0 = result.layers[0]
        dom = "MoE" if l0.moe_ffn_flops > l0.attention_flops else "Attn"
        lines.append(
            f"\n  ▸ Summary: MFU={result.mfu_pct:.1f}% | {dom}-dominant | {result.gpu} | TP={result.tp} EP={result.ep}"
        )

    return "\n".join(lines)


def format_kernel_flow(
    kernel_detail: dict,
    ops: list,
    peak_tflops: float,
    tp: int,
    ep: int,
    compress_ratio: int = 0,
    fp8_peak_tflops: float = 0,
) -> str:
    """Format kernel-level MFU table that preserves every kernel row.

    This uses per-kernel timing rows and adds:
    - Mapped Op: which operator this kernel maps to
    - FLOPs: operator's total FLOPs
    - Theo(us): theoretical minimum time
    - MFU%: measured FLOPs utilization
    - shape_in→shape_out: operator tensor dimensions
    """
    lines = []
    kernels = kernel_detail.get("kernels", [])
    cat_summary = kernel_detail.get("category_summary", {})
    meta = kernel_detail.get("metadata", {})

    # Step 1: Run the full operator-level mapping to get per-operator measured time
    mapping_result = map_kernel_detail_to_ops(kernel_detail, ops, tp, ep)
    op_measured_us = mapping_result["op_measured_us"]

    # Step 2: Build op name → {flops, theo_us, measured_us, shape_in, shape_out} dict
    # For ops with measured time, compute overall MFU per operator
    op_info = {}
    op_idx_by_name = {}
    for i, op in enumerate(ops):
        per_gpu = (
            op.flops / ep
            if (
                op.category == "moe" and op.name.startswith("routed_experts") and ep > 0
            )
            else op.flops / tp if tp > 0 else op.flops
        )
        theo_us = per_gpu / (peak_tflops * 1e6) if peak_tflops > 0 else 0
        meas_us = op_measured_us.get(i, None)
        mfu = (
            (theo_us / meas_us * 100)
            if (meas_us and meas_us > 0 and theo_us > 0)
            else None
        )
        op_info[i] = {
            "name": op.name,
            "flops": op.flops,
            "per_gpu": per_gpu,
            "theo_us": theo_us,
            "measured_us": meas_us,
            "mfu": mfu,
            "shape_in": op.shape_in,
            "shape_out": op.shape_out,
            "category": op.category,
        }
        # Track first index for each op name
        if op.name not in op_idx_by_name:
            op_idx_by_name[op.name] = i

    # Step 3: For each kernel category, determine the mapped operator index
    # Build category → [op_indices] mapping
    cat_to_op_indices = {}
    assigned_op_indices = set()

    for cat_key in cat_summary:
        mapping = KERNEL_DETAIL_DIRECT_MAP.get(cat_key)
        if mapping is None:
            # Generic GEMM — will resolve later
            cat_to_op_indices[cat_key] = None
        elif len(mapping) == 0:
            # Overhead
            cat_to_op_indices[cat_key] = []
        else:
            # Direct match
            indices = []
            for op_name in mapping:
                if op_name in op_idx_by_name:
                    idx = op_idx_by_name[op_name]
                    indices.append(idx)
                    assigned_op_indices.add(idx)
            cat_to_op_indices[cat_key] = indices

    # Resolve GEMM categories — both fp8 and bf16 share the same pool of
    # unassigned GEMM-like ops; don't mark them as exclusively assigned
    gemm_eligible_indices = []
    for cat_key in ("gemm_fp8", "gemm_bf16"):
        if cat_key not in cat_summary:
            continue
        gemm_categories = ["attention", "moe", "mhc", "norm", "ffn"]

        eligible_indices = [
            i
            for i in range(len(ops))
            if i not in assigned_op_indices and ops[i].category in gemm_categories
        ]
        if not eligible_indices:
            eligible_indices = [
                i for i in range(len(ops)) if i not in assigned_op_indices
            ]
        cat_to_op_indices[cat_key] = eligible_indices
        gemm_eligible_indices.extend(eligible_indices)
    # Deduplicate
    gemm_eligible_indices = list(dict.fromkeys(gemm_eligible_indices))
    for idx in gemm_eligible_indices:
        assigned_op_indices.add(idx)

    # Step 4: For each kernel, find the mapped operator(s)
    def get_mapped_ops_info(cat_key):
        """Return list of (op_idx, flops_share) for a kernel category.

        For direct-match: returns the mapped ops with FLOPs-proportional shares.
        For GEMM: returns all eligible ops with FLOPs-proportional shares.
        For overhead: returns empty list.
        """
        indices = cat_to_op_indices.get(cat_key)
        if indices is None or len(indices) == 0:
            return []
        # Compute FLOPs share among the mapped ops
        total_flops = sum(ops[i].flops for i in indices if i < len(ops))
        if total_flops == 0:
            return [(indices[0], 1.0 / len(indices))] * len(indices)
        return [(i, ops[i].flops / total_flops) for i in indices if i < len(ops)]

    # Step 5: Compute per-kernel FLOPs allocation
    # For each kernel, allocate a proportional share of the mapped operator's FLOPs
    # based on the kernel's duration relative to the total duration of its category.
    # This prevents MFU > 100% when multiple kernels share the same operator.

    # Build category → total duration mapping
    cat_total_dur = {}
    for k in kernels:
        ck = k.get("category", "other")
        cat_total_dur[ck] = cat_total_dur.get(ck, 0) + k.get("dur_us", 0)

    # For GEMM categories, they share the same operator pool,
    # so we need to combine their durations
    gemm_total_dur = sum(cat_total_dur.get(gk, 0) for gk in ("gemm_fp8", "gemm_bf16"))

    # Step 6: Format the kernel-flow table
    total_dur = meta.get("total_dur_us", sum(k.get("dur_us", 0) for k in kernels))
    layer_id = meta.get("layer_id", 0)
    cr = meta.get("compress_ratio", compress_ratio)
    cr_label = f" (compress_ratio={cr})" if cr >= 0 else ""
    type_label = {0: "SWA_ONLY", 4: "CSA_C4", 128: "HCA_C128"}.get(cr, "")
    type_str = f" [{type_label}]" if type_label else ""

    lines.append(f"\n{'=' * 140}")
    lines.append(
        f"  Kernel-Level MFU: Layer {layer_id}{type_str}{cr_label}"
        f"  —  {total_dur:.0f}us ({total_dur/1000:.2f}ms), {len(kernels)} kernels"
    )
    lines.append(f"{'=' * 140}")

    # Header
    lines.append(
        f"  {'#':>3s} {'Half':>4s} {'Category':<16s} {'Simplified Name':<42s}"
        f" {'dur(us)':>8s} {'%':>5s} {'Mapped Op':<24s}"
        f" {'FLOPs':>10s} {'Theo(us)':>9s} {'MFU%':>7s} {'shape_in→out':<22s}"
    )
    lines.append(f"  {'-' * 138}")

    for i, k in enumerate(kernels):
        name = k.get("simplified_name", k.get("name", ""))
        cat_key = k.get("category", "other")
        half = k.get("half", "?")
        dur_us = k.get("dur_us", 0)
        pct = dur_us / total_dur * 100 if total_dur > 0 else 0

        # Truncate name
        if len(name) > 42:
            name = name[:39] + "..."

        # Find mapped operator(s) and compute per-kernel FLOPs allocation
        mapped_ops = get_mapped_ops_info(cat_key)
        if mapped_ops:
            # Determine the reference duration for proportional FLOPs splitting
            # For GEMM categories, use combined gemm duration since they share the pool
            if cat_key in ("gemm_fp8", "gemm_bf16"):
                ref_dur = (
                    gemm_total_dur
                    if gemm_total_dur > 0
                    else cat_total_dur.get(cat_key, dur_us)
                )
            else:
                ref_dur = cat_total_dur.get(cat_key, dur_us)
            kernel_share = dur_us / ref_dur if ref_dur > 0 else 1.0

            # Show the primary (largest share) operator for the row
            primary_idx = max(mapped_ops, key=lambda x: x[1])[0]
            oi = op_info[primary_idx]
            # Build mapped name showing shares if multiple ops
            if len(mapped_ops) > 1:
                # Show top-2 ops with share percentages
                sorted_ops = sorted(mapped_ops, key=lambda x: -x[1])[:2]
                parts = []
                for idx, share in sorted_ops:
                    parts.append(f"{op_info[idx]['name']}({share*100:.0f}%)")
                mapped_name = "+".join(parts)
            else:
                mapped_name = oi["name"]
            if len(mapped_name) > 24:
                mapped_name = mapped_name[:21] + "..."
            # Per-kernel FLOPs: operator's FLOPs * this kernel's share of the category
            kernel_flops = oi["per_gpu"] * kernel_share
            # Use fp8 peak for kernels known to compute internally in fp8
            effective_peak = (
                fp8_peak_tflops
                if cat_key in FP8_INTERNAL_KERNEL_CATEGORIES and fp8_peak_tflops > 0
                else peak_tflops
            )
            kernel_theo_us = (
                kernel_flops / (effective_peak * 1e6) if effective_peak > 0 else 0
            )
            mfu = (
                (kernel_theo_us / dur_us * 100)
                if (dur_us > 0 and kernel_theo_us > 0)
                else None
            )
            flops_str = fmt_flops_short(int(kernel_flops))
            theo_str = f"{kernel_theo_us:.1f}"
            mfu_str = f"{mfu:.1f}%" if mfu is not None else "N/A"
            # Add fp8 indicator for kernels using fp8 peak
            if cat_key in FP8_INTERNAL_KERNEL_CATEGORIES and fp8_peak_tflops > 0:
                mfu_str = f"{mfu:.1f}%⁸" if mfu is not None else "N/A"
            shape_str = f"{oi['shape_in']}→{oi['shape_out']}"
            if len(shape_str) > 22:
                shape_str = shape_str[:19] + "..."
        else:
            mapped_name = "—"
            flops_str = "—"
            theo_str = "—"
            mfu_str = "N/A"
            shape_str = "—"

        lines.append(
            f"  {i:>3d} {half:>4s} {cat_key:<16s} {name:<42s}"
            f" {dur_us:>7.1f} {pct:>4.1f}% {mapped_name:<24s}"
            f" {flops_str:>10s} {theo_str:>9s} {mfu_str:>7s} {shape_str:<22s}"
        )

    return "\n".join(lines)


def fmt_flops_short(f: int) -> str:
    """Compact FLOPs formatting for table cells."""
    if f >= 1e12:
        return f"{f / 1e12:.1f}T"
    if f >= 1e9:
        return f"{f / 1e9:.1f}G"
    if f >= 1e6:
        return f"{f / 1e6:.1f}M"
    if f >= 1e3:
        return f"{f / 1e3:.1f}K"
    return str(f)


def format_json(result: SimResult) -> str:
    data = {
        "model": result.model,
        "config_source": result.config_source,
        "batch_size": result.batch_size,
        "seq_len": result.seq_len,
        "context_len": result.context_len,
        "all_token_logits": result.all_token_logits,
        "epilogue_flops": result.epilogue_flops,
        "tp": result.tp,
        "dp": result.dp,
        "ep": result.ep,
        "gpu": result.gpu,
        "dtype": result.dtype,
        "total_flops": result.total_flops,
        "embed_flops": result.embed_flops,
        "layer_template": (
            [
                {
                    "name": op.name,
                    "flops": op.flops,
                    "shape_in": op.shape_in,
                    "shape_out": op.shape_out,
                    "category": op.category,
                }
                for op in result.layers[0].ops
            ]
            if result.layers
            else []
        ),
        "num_layers": len(result.layers),
        "measured_ms": result.measured_ms,
        "mfu_pct": result.mfu_pct,
        "per_layer_mfu_pct": result.per_layer_mfu_pct,
        "per_op_mfu": result.per_op_mfu if result.per_op_mfu else [],
        "kernel_flow": result.kernel_flow,
        "model_arch": result.model_arch,
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model Compute Simulator")
    parser.add_argument("model", nargs="?", help="Model name (e.g. DeepSeek-V4-Flash)")
    parser.add_argument(
        "--list-models", action="store_true", help="List known model IDs"
    )
    parser.add_argument("--list-gpus", action="store_true", help="List known GPU types")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--seq-len", type=int, default=1, help="Sequence length (1 for decode)"
    )
    parser.add_argument("--config", type=Path, help="Public or local model config.json")
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Total KV length for decode; prefix KV length for extend",
    )
    parser.add_argument("--all-token-logits", action="store_true")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism")
    parser.add_argument("--ep", type=int, default=8, help="Expert parallelism")
    parser.add_argument("--gpu", default="h20", help="GPU type")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp8", "mxfp8", "fp4", "mxfp4", "nvfp4"],
        help="Data type",
    )
    parser.add_argument(
        "--measured-ms",
        type=float,
        default=None,
        help="Measured forward-pass latency (ms)",
    )
    parser.add_argument(
        "--per-layer-ms",
        type=float,
        default=None,
        help="Measured per-layer latency (ms); overrides --measured-ms for per-layer MFU",
    )
    parser.add_argument(
        "--kernel-ms",
        default=None,
        help="JSON object mapping kernel categories to measured ms",
    )
    parser.add_argument(
        "--kernel-detail",
        default=None,
        help="JSON string or @file path with per-kernel detail for precise per-operator MFU",
    )
    parser.add_argument(
        "--kernel-flow",
        default=None,
        help="JSON string or @file path with per-kernel detail; produces kernel-level MFU table preserving all kernel rows",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        default="text",
        choices=["text", "json"],
        help="Output format",
    )

    args = parser.parse_args()

    config_index = load_json(CONFIG_INDEX)
    for key, value in config_index.items():
        ALIAS_MAP[value.get("hf_config_source", key).lower()] = key
    gpu_specs = load_json(GPU_SPECS)

    if args.list_models:
        print("Known model IDs:")
        for key, val in config_index.items():
            aliases = [k for k, v in ALIAS_MAP.items() if v == key and k != key]
            alias_str = f"  (aliases: {', '.join(aliases)})" if aliases else ""
            print(f"  {key}: {val['display_name']}{alias_str}")
        return

    if args.list_gpus:
        print("Known GPU types:")
        for key, val in gpu_specs.items():
            aliases = [k for k, v in GPU_ALIAS.items() if v == key and k != key]
            alias_str = f"  (aliases: {', '.join(aliases)})" if aliases else ""
            print(
                f"  {key}: {val['display_name']}  BF16={val['bf16_tflops'] if val['bf16_tflops'] is not None else 'unknown'} TFLOPS{alias_str}"
            )
        return

    if not args.model and not args.config:
        parser.error(
            "model name is required (use --list-models to see available models)"
        )

    cfg = (
        normalize_model_config(load_json(args.config))
        if args.config
        else resolve_model(args.model, config_index)
    )
    if args.config and not args.model:
        args.model = cfg.get("model_type", args.config.stem)
    print(
        f"Assumptions: B={args.batch_size}, S={args.seq_len}, KV/prefix={args.context_len}, TP={args.tp}, DP={args.dp}, EP={args.ep}, GPU={args.gpu}, dtype={args.dtype}. Sparse/SSM work and zero-expert top-k are estimates; LM head uses last-token logits unless --all-token-logits.",
        file=sys.stderr,
    )
    if args.context_len is not None and args.context_len < 1:
        parser.error("--context-len must be positive")
    if cfg is None:
        print(
            f"Error: model '{args.model}' not found in config index.", file=sys.stderr
        )
        print(f"Use --list-models to see available models.", file=sys.stderr)
        sys.exit(1)

    specific_timing_inputs = [
        args.per_layer_ms is not None,
        args.kernel_ms is not None,
        args.kernel_detail is not None,
        args.kernel_flow is not None,
    ]
    if sum(specific_timing_inputs) > 1:
        parser.error(
            "choose only one of --per-layer-ms, --kernel-ms, "
            "--kernel-detail, or --kernel-flow"
        )

    measured_ms = args.measured_ms
    kernel_ms = None
    kernel_detail = None
    kernel_flow_detail = None
    kernel_detail_result = None  # result from map_kernel_detail_to_ops
    n_layers = cfg.get("num_hidden_layers", 1)

    try:
        if args.kernel_flow is not None:
            kernel_flow_detail = load_json_argument(args.kernel_flow)
            measured_ms = measured_ms_from_kernel_detail(kernel_flow_detail, n_layers)
        elif args.kernel_detail is not None:
            kernel_detail = load_json_argument(args.kernel_detail)
            measured_ms = measured_ms_from_kernel_detail(kernel_detail, n_layers)
        elif args.kernel_ms is not None:
            kernel_ms = json.loads(args.kernel_ms)
            layer_measured = sum(float(value) for value in kernel_ms.values())
            if layer_measured <= 0:
                raise ValueError("--kernel-ms durations must sum to a positive value")
            measured_ms = layer_measured * n_layers
        elif args.per_layer_ms is not None:
            if args.per_layer_ms <= 0:
                raise ValueError("--per-layer-ms must be positive")
            measured_ms = args.per_layer_ms * n_layers
        elif measured_ms is not None and measured_ms <= 0:
            raise ValueError("--measured-ms must be positive")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    result = simulate(
        cfg,
        args.model,
        args.batch_size,
        args.seq_len,
        args.tp,
        args.dp,
        args.ep,
        args.gpu,
        args.dtype,
        measured_ms,
        context_len=args.context_len,
        all_token_logits=args.all_token_logits,
    )
    # If per-layer-ms was used, show the per-layer measured time in output
    if args.per_layer_ms is not None:
        result.measured_ms = measured_ms  # total
        result._per_layer_measured_ms = args.per_layer_ms

    # If kernel-detail was provided, map measured durations to per-operator (precise)
    if kernel_flow_detail is not None:
        result.kernel_flow = kernel_flow_detail
    elif kernel_detail is not None:
        result.kernel_detail = kernel_detail
        if result.layers and result.per_op_mfu:
            # Use the same representative layer that per_op_mfu is based on
            repr_layer = None
            for lr in result.layers:
                if getattr(lr, "compress_ratio", 0) == 128:
                    repr_layer = lr
                    break
            if repr_layer is None and result.layers:
                repr_layer = result.layers[0]
            if repr_layer is not None:
                kernel_detail_result = map_kernel_detail_to_ops(
                    kernel_detail, repr_layer.ops, args.tp, args.ep
                )
                for i, measured_us in kernel_detail_result["op_measured_us"].items():
                    if i < len(result.per_op_mfu):
                        result.per_op_mfu[i]["measured_us"] = measured_us
                result._kernel_detail_meta = kernel_detail_result
    elif kernel_ms is not None:
        result.kernel_ms = kernel_ms
        if result.layers and result.per_op_mfu:
            repr_layer = None
            for lr in result.layers:
                if getattr(lr, "compress_ratio", 0) == 128:
                    repr_layer = lr
                    break
            if repr_layer is None and result.layers:
                repr_layer = result.layers[0]
            if repr_layer is not None:
                op_measured = map_kernel_ms_to_ops(
                    kernel_ms, repr_layer.ops, args.tp, args.ep
                )
                for i, measured_us in op_measured.items():
                    if i < len(result.per_op_mfu):
                        result.per_op_mfu[i]["measured_us"] = measured_us

    if args.fmt == "json":
        print(format_json(result))
        return
    else:
        print(format_text(result, skip_compute_flow=kernel_flow_detail is not None))

    # Kernel-flow output (appended after the main output)
    # When kernel-flow is used, skip the per-op static template in format_text
    if kernel_flow_detail is not None:
        # Find the representative layer matching the kernel detail
        cr_meta = kernel_flow_detail.get("metadata", {}).get("compress_ratio", 0)
        repr_layer = None
        for lr in result.layers:
            if getattr(lr, "compress_ratio", 0) == cr_meta:
                repr_layer = lr
                break
        if repr_layer is None and result.layers:
            repr_layer = result.layers[0]

        if repr_layer is not None:
            gpu_info = resolve_gpu(args.gpu, gpu_specs) if args.gpu else None
            peak_tflops = 0
            fp8_peak_tflops = 0
            if gpu_info:
                dtype_key = peak_dtype_key(args.dtype)
                peak_tflops = gpu_info.get(dtype_key) or 0
                fp8_peak_tflops = gpu_info.get("fp8_tflops") or 0

            print(
                format_kernel_flow(
                    kernel_flow_detail,
                    repr_layer.ops,
                    peak_tflops,
                    args.tp,
                    args.ep,
                    compress_ratio=cr_meta,
                    fp8_peak_tflops=fp8_peak_tflops,
                )
            )


if __name__ == "__main__":
    main()
