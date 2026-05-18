#!/usr/bin/env python3
"""Model Compute Simulator — construct architecture, execution flow, tensor dims, FLOPs, MFU.

Usage:
  # List known models
  python3 model_compute_simulator.py --list-models

  # List known GPUs
  python3 model_compute_simulator.py --list-gpus

  # Simulate decode batch=1
  python3 model_compute_simulator.py "DeepSeek-V4-Flash" \
    --batch-size 1 --seq-len 1 --tp 8 --dp 1 --ep 8 --gpu h20 --dtype bf16

  # With MFU
  python3 model_compute_simulator.py "DeepSeek-V4-Flash" \
    --batch-size 1 --seq-len 1 --tp 8 --dp 1 --ep 8 --gpu h20 --dtype bf16 \
    --measured-ms 15.0
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

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
    "glm-5": "glm-5",
}

GPU_ALIAS = {
    "h20": "h20",
    "h20-sxm": "h20",
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
    layers: List[LayerResult] = field(default_factory=list)
    total_flops: int = 0
    embed_flops: int = 0
    measured_ms: Optional[float] = None
    mfu_pct: Optional[float] = None
    per_layer_mfu_pct: Optional[float] = None
    per_op_mfu: list = field(default_factory=list)  # per-operator MFU details
    kernel_ms: Optional[dict] = None  # kernel-category → measured ms mapping
    model_arch: dict = field(default_factory=dict)  # model architecture summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def resolve_model(name: str, config_index: dict) -> Optional[dict]:
    key = name.lower().strip().replace(" ", "-")
    key = ALIAS_MAP.get(key, key)
    return config_index.get(key)


def resolve_gpu(name: str, gpu_specs: dict) -> Optional[dict]:
    key = name.lower().strip()
    key = GPU_ALIAS.get(key, key)
    return gpu_specs.get(key)


def matmul_flops(m, n, k):
    """FLOPs for C = A @ B where A:[m,k], B:[k,n] => 2*m*n*k."""
    return 2 * m * n * k


def fmt_shape(*dims):
    return "[" + "×".join(str(d) for d in dims if d is not None) + "]"


# ---------------------------------------------------------------------------
# Build per-layer ops
# ---------------------------------------------------------------------------
def build_layer_ops(cfg: dict, B: int, S: int, tp: int, ep: int,
                    compress_ratio: int = 0, layer_idx: int = 0) -> List[Op]:
    """Build operator list for one transformer layer.

    Args:
        cfg: model configuration dict
        B: batch size
        S: sequence length
        tp: tensor parallelism
        ep: expert parallelism
        compress_ratio: 0=full_attn, 4=C4_LIGHT, 128=C128_HEAVY (from compress_ratios)
        layer_idx: layer index (for hash layer detection)
    """
    ops = []
    H = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    d_head = cfg.get("head_dim", H // n_heads)
    attn_type = cfg.get("attention_type", "gqa")
    is_moe = cfg.get("moe", False)
    n_layers = cfg.get("num_hidden_layers", 1)
    n_hash_layers = cfg.get("num_hash_layers", 0)
    is_hash = n_hash_layers > 0 and layer_idx >= n_layers - n_hash_layers

    # ---- Input Layernorm (RMSNorm) ----
    ops.append(Op("rmsnorm", 2 * H * 3,  # mean + (x-mean)*rsqrt(var+eps) + *gamma
                  fmt_shape(B, S, H), fmt_shape(B, S, H), "norm"))

    # ---- Attention ----
    if attn_type == "mla":
        # MLA: compress KV then attend
        compress_dim = cfg.get("mla_compress_dim", 512)
        # Q projection
        q_flops = matmul_flops(B * S, n_heads * d_head, H)
        ops.append(Op("q_proj", q_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, n_heads, d_head), "attention"))
        # KV compress
        kv_comp_flops = matmul_flops(B * S, 2 * compress_dim, H)
        ops.append(Op("kv_compress", kv_comp_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, 2, compress_dim), "attention"))
        # Attention score: Q @ K^T
        attn_score_flops = matmul_flops(B * n_heads, S, d_head) if S > 1 else 0
        if attn_score_flops > 0:
            ops.append(Op("attn_score", attn_score_flops,
                          fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, S), "attention"))
        # Attn @ V
        attn_v_flops = matmul_flops(B * n_heads, S, compress_dim // n_heads) if S > 1 else matmul_flops(B * n_heads, 1, compress_dim // n_heads)
        ops.append(Op("attn_v", attn_v_flops,
                      fmt_shape(B, n_heads, S, compress_dim // n_heads), fmt_shape(B, n_heads, S, d_head), "attention"))
        # O projection
        o_flops = matmul_flops(B * S, H, n_heads * d_head)
        ops.append(Op("o_proj", o_flops,
                      fmt_shape(B, S, n_heads, d_head), fmt_shape(B, S, H), "attention"))
        # KV upcast (decompress)
        kv_decomp_flops = matmul_flops(B * S, 2 * H, compress_dim)
        ops.append(Op("kv_decompress", kv_decomp_flops,
                      fmt_shape(B, S, compress_dim), fmt_shape(B, S, 2 * H), "attention"))

    elif attn_type == "csa_hca":
        # CSA/HCA: Compressed Sparse Attention + Heavily Compressed Attention
        # Standard QKV when q_lora_rank=0; LoRA Q/O projections when q_lora_rank > 0
        q_lora_rank = cfg.get("q_lora_rank", 0)
        o_lora_rank = cfg.get("o_lora_rank", 0)
        o_groups = cfg.get("o_groups", 1)
        rope_dim = cfg.get("qk_rope_head_dim", 0)

        if q_lora_rank > 0:
            # MLA-style Q: H → q_lora_rank → n_heads * d_head
            q_down_flops = matmul_flops(B * S, q_lora_rank, H)
            ops.append(Op("q_lora_down", q_down_flops,
                          fmt_shape(B, S, H), fmt_shape(B, S, q_lora_rank), "attention"))
            q_up_flops = matmul_flops(B * S, n_heads * d_head, q_lora_rank)
            ops.append(Op("q_lora_up", q_up_flops,
                          fmt_shape(B, S, q_lora_rank), fmt_shape(B, S, n_heads, d_head), "attention"))
            # KV compress (MLA): H → n_kv_heads * d_head
            kv_dim = n_kv_heads * d_head
            kv_comp_flops = matmul_flops(B * S, kv_dim, H)
            ops.append(Op("kv_compress", kv_comp_flops,
                          fmt_shape(B, S, H), fmt_shape(B, S, kv_dim), "attention"))
        else:
            # Standard QKV projection
            qkv_flops = matmul_flops(B * S, (n_heads + 2 * n_kv_heads) * d_head, H)
            ops.append(Op("qkv_proj", qkv_flops,
                          fmt_shape(B, S, H), fmt_shape(B, S, (n_heads + 2 * n_kv_heads) * d_head), "attention"))

        # FP8 quantization (before attention)
        quant_flops = 2 * B * S * H  # scale + quantize per element
        ops.append(Op("quant", quant_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, H), "quant"))

        # RoPE (rotary position embedding)
        if rope_dim > 0:
            rope_flops = 2 * B * S * rope_dim * 4  # cos/sin multiply for Q and K
            ops.append(Op("rope", rope_flops,
                          fmt_shape(B, S, rope_dim), fmt_shape(B, S, rope_dim), "attention"))

        # ---- NSA-specific ops for C128 layers ----
        index_n_heads = cfg.get("index_n_heads", 0)
        index_head_dim = cfg.get("index_head_dim", 0)
        index_topk = cfg.get("index_topk", 0)
        sliding_window = cfg.get("sliding_window", 128)

        if compress_ratio == 128 and index_n_heads > 0:
            # Hadamard transform (before indexer)
            hadamard_dim = n_heads * d_head  # transform dimension
            hadamard_flops = B * S * hadamard_dim * 14  # ~N*log2(N)/N=14 for 16K
            ops.append(Op("hadamard", hadamard_flops,
                          fmt_shape(B, S, hadamard_dim), fmt_shape(B, S, hadamard_dim), "attention"))

            # Indexer Q/K projection
            indexer_q_flops = matmul_flops(B * S, index_n_heads * index_head_dim, q_lora_rank)
            ops.append(Op("indexer_proj", indexer_q_flops,
                          fmt_shape(B, S, q_lora_rank), fmt_shape(B, S, index_n_heads, index_head_dim), "attention"))

            # Paged MQA (indexer attention: index_n_heads Q, 1 KV head, topk blocks)
            n_blocks = (S + sliding_window - 1) // sliding_window
            paged_mqa_kv_len = min(index_topk * sliding_window, S)
            paged_mqa_flops = matmul_flops(B * index_n_heads, S, index_head_dim) if S > 1 else 0
            paged_mqa_flops += matmul_flops(B * index_n_heads, S if S > 1 else 1, index_head_dim)
            if paged_mqa_flops > 0:
                ops.append(Op("paged_mqa", paged_mqa_flops,
                              fmt_shape(B, index_n_heads, S, index_head_dim),
                              fmt_shape(B, index_n_heads, S, index_head_dim), "attention"))

        # C4 sparse attention (for layers with compress_ratio > 0)
        if compress_ratio > 0 and S > 1:
            c4_kv_len = max(S // compress_ratio, sliding_window)
            c4_attn = matmul_flops(B * n_heads, S * c4_kv_len, d_head) // S  # per-query-token
            c4_attn_total = matmul_flops(B * n_heads, S, d_head)  # score: Q @ K^T
            ops.append(Op("c4_attn", c4_attn_total,
                          fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, c4_kv_len), "attention"))
            c4_v = matmul_flops(B * n_heads, S, d_head)
            ops.append(Op("c4_attn_v", c4_v,
                          fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, d_head), "attention"))
        elif S > 1:
            # Full attention (compress_ratio == 0)
            csa_attn = matmul_flops(B * n_heads, S, d_head)
            ops.append(Op("csa_attn_score", csa_attn,
                          fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, S), "attention"))
            csa_v = matmul_flops(B * n_heads, S, d_head)
            ops.append(Op("csa_attn_v", csa_v,
                          fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, d_head), "attention"))

        # HCA attention (heavily compressed, always present for csa_hca)
        hca_attn = matmul_flops(B * n_kv_heads, S, d_head) if S > 1 else 0
        if hca_attn > 0:
            ops.append(Op("hca_attn_score", hca_attn,
                          fmt_shape(B, n_kv_heads, S, d_head), fmt_shape(B, n_kv_heads, S, S), "attention"))
        hca_v = matmul_flops(B * n_kv_heads, S if S > 1 else 1, d_head)
        ops.append(Op("hca_attn_v", hca_v,
                      fmt_shape(B, n_kv_heads, S, d_head), fmt_shape(B, n_kv_heads, S, d_head), "attention"))

        # MLA cache store
        mla_cache_flops = 2 * B * S * (n_kv_heads * d_head + rope_dim)  # write KV cache
        ops.append(Op("mla_cache_store", mla_cache_flops,
                      fmt_shape(B, S, n_kv_heads * d_head), fmt_shape(B, S, n_kv_heads * d_head), "attention"))

        # O projection
        if o_lora_rank > 0:
            # Grouped O-LoRA: each group d_head → o_lora_rank/o_groups, then up → H
            per_group_down = o_lora_rank // o_groups
            o_down_flops = matmul_flops(B * S * o_groups, per_group_down, d_head)
            ops.append(Op("o_lora_down", o_down_flops,
                          fmt_shape(B, S, o_groups, d_head), fmt_shape(B, S, o_groups, per_group_down), "attention"))
            o_up_flops = matmul_flops(B * S, H, o_lora_rank)
            ops.append(Op("o_lora_up", o_up_flops,
                          fmt_shape(B, S, o_lora_rank), fmt_shape(B, S, H), "attention"))
        else:
            o_flops = matmul_flops(B * S, H, n_heads * d_head)
            ops.append(Op("o_proj", o_flops,
                          fmt_shape(B, S, n_heads * d_head), fmt_shape(B, S, H), "attention"))

    elif attn_type == "gqa" or attn_type == "gqa_qk_norm":
        # GQA: grouped-query attention
        q_flops = matmul_flops(B * S, n_heads * d_head, H)
        ops.append(Op("q_proj", q_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, n_heads, d_head), "attention"))
        kv_flops = matmul_flops(B * S, 2 * n_kv_heads * d_head, H)
        ops.append(Op("kv_proj", kv_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, 2, n_kv_heads, d_head), "attention"))
        # QK norm (optional)
        if attn_type == "gqa_qk_norm":
            ops.append(Op("qk_norm", 2 * (n_heads + n_kv_heads) * d_head * 3,
                          fmt_shape(B, n_heads, d_head), fmt_shape(B, n_heads, d_head), "attention"))
        # Attention
        n_groups = n_heads // n_kv_heads
        attn_score_flops = matmul_flops(B * n_heads, S, d_head) if S > 1 else 0
        if attn_score_flops > 0:
            ops.append(Op("attn_score", attn_score_flops,
                          fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, S), "attention"))
        attn_v_flops = matmul_flops(B * n_heads, S if S > 1 else 1, d_head)
        ops.append(Op("attn_v", attn_v_flops,
                      fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, d_head), "attention"))
        # O projection
        o_flops = matmul_flops(B * S, H, n_heads * d_head)
        ops.append(Op("o_proj", o_flops,
                      fmt_shape(B, S, n_heads, d_head), fmt_shape(B, S, H), "attention"))

    elif attn_type == "dsa":
        # DSA: DeepSeek-style sparse attention (similar to GQA for FLOPs)
        q_flops = matmul_flops(B * S, n_heads * d_head, H)
        ops.append(Op("q_proj", q_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, n_heads, d_head), "attention"))
        kv_flops = matmul_flops(B * S, 2 * n_kv_heads * d_head, H)
        ops.append(Op("kv_proj", kv_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, 2, n_kv_heads, d_head), "attention"))
        attn_score_flops = matmul_flops(B * n_heads, S, d_head) if S > 1 else 0
        if attn_score_flops > 0:
            ops.append(Op("attn_score", attn_score_flops,
                          fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, S), "attention"))
        attn_v_flops = matmul_flops(B * n_heads, S if S > 1 else 1, d_head)
        ops.append(Op("attn_v", attn_v_flops,
                      fmt_shape(B, n_heads, S, d_head), fmt_shape(B, n_heads, S, d_head), "attention"))
        o_flops = matmul_flops(B * S, H, n_heads * d_head)
        ops.append(Op("o_proj", o_flops,
                      fmt_shape(B, S, n_heads, d_head), fmt_shape(B, S, H), "attention"))
    else:
        raise ValueError(f"Unknown attention_type: {attn_type}")

    # ---- Post-attention RMSNorm ----
    ops.append(Op("rmsnorm", 2 * H * 3,
                  fmt_shape(B, S, H), fmt_shape(B, S, H), "norm"))

    # ---- Residual add ----
    ops.append(Op("residual_add", H,
                  fmt_shape(B, S, H), fmt_shape(B, S, H), "residual"))

    # ---- MoE / FFN ----
    if is_moe:
        topk = cfg["num_experts_per_tok"]
        n_experts = cfg["num_experts"]
        routed_inter = cfg.get("routed_expert_intermediate_size", 0)
        shared_inter = cfg.get("shared_expert_intermediate_size", 0)
        n_shared = cfg.get("num_shared_experts", 0)

        # Router
        router_flops = matmul_flops(B * S, n_experts, H)
        ops.append(Op("router", router_flops,
                      fmt_shape(B, S, H), fmt_shape(B, S, n_experts), "moe"))

        # TopK selection
        topk_flops = B * S * n_experts * 4  # comparison + selection per token
        ops.append(Op("topk", topk_flops,
                      fmt_shape(B, S, n_experts), fmt_shape(B, S, topk), "moe"))

        # Routed experts (top-k selected per token, SwiGLU)
        # Per expert: gate(H→inter) + up(H→inter) + silu_mul + down(inter→H)
        routed_gate = matmul_flops(B * S * topk, routed_inter, H)
        routed_up = matmul_flops(B * S * topk, routed_inter, H)
        routed_down = matmul_flops(B * S * topk, H, routed_inter)
        routed_total = routed_gate + routed_up + routed_down
        ops.append(Op("routed_experts_swiglu", routed_total,
                      fmt_shape(B, S, topk, H), fmt_shape(B, S, topk, H), "moe"))

        # Activation (silu_mul in MoE)
        act_flops = 3 * B * S * topk * routed_inter  # silu(x)*gate(x)
        ops.append(Op("activation", act_flops,
                      fmt_shape(B, S, topk, routed_inter), fmt_shape(B, S, topk, routed_inter), "moe"))

        # Shared experts (if any)
        if n_shared > 0 and shared_inter > 0:
            shared_gate = matmul_flops(B * S, shared_inter, H)
            shared_up = matmul_flops(B * S, shared_inter, H)
            shared_down = matmul_flops(B * S, H, shared_inter)
            shared_total = shared_gate + shared_up + shared_down
            ops.append(Op("shared_experts_swiglu", shared_total,
                          fmt_shape(B, S, H), fmt_shape(B, S, H), "moe"))
    else:
        # Dense FFN (SwiGLU)
        inter = cfg.get("intermediate_size", 4 * H)
        gate_flops = matmul_flops(B * S, inter, H)
        up_flops = matmul_flops(B * S, inter, H)
        down_flops = matmul_flops(B * S, H, inter)
        ffn_total = gate_flops + up_flops + down_flops
        ops.append(Op("ffn_swiglu", ffn_total,
                      fmt_shape(B, S, H), fmt_shape(B, S, H), "ffn"))

    # ---- Residual add ----
    ops.append(Op("residual_add", H,
                  fmt_shape(B, S, H), fmt_shape(B, S, H), "residual"))

    # ---- MHC (Manifold-Constrained Hyper-Connections) ----
    if cfg.get("mhc", False):
        mhc_dim = cfg.get("mhc_bottleneck_dim", H // 4)
        mhc_down = matmul_flops(B * S, mhc_dim, H)
        mhc_up = matmul_flops(B * S, H, mhc_dim)
        mhc_total = mhc_down + mhc_up
        ops.append(Op("mhc_post_tilelang", mhc_total,
                      fmt_shape(B, S, H), fmt_shape(B, S, H), "mhc"))

    return ops


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
def simulate(cfg: dict, model_name: str, B: int, S: int, tp: int, dp: int, ep: int,
             gpu_name: str, dtype: str, measured_ms: Optional[float] = None) -> SimResult:
    """Run the full simulation."""
    gpu_specs = load_json(GPU_SPECS)
    gpu_info = resolve_gpu(gpu_name, gpu_specs) if gpu_name else None

    n_layers = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    V = cfg.get("vocab_size", 0)

    # Embedding
    embed_flops = matmul_flops(B * S, H, V) if V > 0 else 0

    # Per-layer (with per-layer compress_ratio)
    compress_ratios = cfg.get("compress_ratios", None)
    layer_ops_cache = {}  # compress_ratio → ops list (cache identical layers)

    total_layer_flops = 0
    layers = []
    for i in range(n_layers):
        cr = compress_ratios[i] if compress_ratios and i < len(compress_ratios) else 0
        if cr not in layer_ops_cache:
            layer_ops_cache[cr] = build_layer_ops(cfg, B, S, tp, ep,
                                                   compress_ratio=cr, layer_idx=i)
        layer_ops = layer_ops_cache[cr]

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

    total_flops = embed_flops + total_layer_flops

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
        "num_experts_per_tok": cfg.get("num_experts_per_tok", 0) if cfg.get("moe", False) else 0,
        "num_shared_experts": cfg.get("num_shared_experts", 0) if cfg.get("moe", False) else 0,
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
        dtype_key = "bf16_tflops" if dtype == "bf16" else "fp8_tflops"
        peak_tflops = gpu_info.get(dtype_key, 0)
        if peak_tflops > 0:
            # Per-GPU FLOPs: attention/shared/MHC split by TP, routed MoE split by EP
            def flops_per_gpu_for_ops(ops_list):
                total = 0
                for op in ops_list:
                    if op.category == "moe" and "routed" in op.name:
                        total += op.flops / ep if ep > 0 else op.flops
                    else:
                        total += op.flops / tp if tp > 0 else op.flops
                return total

            # Overall MFU: measured_ms = total forward pass
            embed_per_gpu = embed_flops / tp if tp > 0 else embed_flops
            # Weight each layer's FLOPs by its compress_ratio
            total_per_gpu = embed_per_gpu
            for lr in layers:
                total_per_gpu += flops_per_gpu_for_ops(lr.ops)
            theoretical_time_s = total_per_gpu / (peak_tflops * 1e12)
            measured_time_s = measured_ms / 1000.0
            mfu_pct = (theoretical_time_s / measured_time_s) * 100.0

            # Per-layer MFU (uniform layer-time assumption)
            avg_layer_per_gpu = total_per_gpu / n_layers if n_layers > 0 else 0
            per_layer_ms = measured_ms / n_layers
            per_layer_mfu_pct = (avg_layer_per_gpu / (peak_tflops * 1e12)) / (per_layer_ms / 1000.0) * 100.0

            # Per-operator FLOPs proportion — show a representative C128_HEAVY layer if available
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
                    if op.category == "moe" and "routed" in op.name:
                        per_gpu = op.flops / ep if ep > 0 else op.flops
                    else:
                        per_gpu = op.flops / tp if tp > 0 else op.flops
                    per_op_mfu.append({
                        "name": op.name,
                        "flops": op.flops,
                        "flops_per_gpu": per_gpu,
                        "theo_us": per_gpu / (peak_tflops * 1e6),
                        "pct_of_layer": op.flops / layer_total_flops * 100 if layer_total_flops > 0 else 0,
                        "category": op.category,
                        "measured_us": None,  # filled later from --kernel-ms
                    })

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
# Maps llm-pipeline-analysis kernel categories to model-compute-simulation
# operator categories. Used to map measured kernel durations to per-operator MFU.

KERNEL_TO_OP_CATEGORY = {
    "mla":           ["attention"],           # flash_fwd_splitkv_mla
    "moe":           ["moe"],                 # fused_moe_kernel
    "allreduce":     [],                       # NCCL — no compute FLOPs
    "hadamard":      ["attention"],           # Hadamard transform (part of C128 attention)
    "indexer":       ["attention"],           # Indexer cache (part of C128 attention)
    "paged_mqa":     ["attention"],           # Paged MQA (hash layer attention)
    "mhc":           ["mhc"],                 # MHC pre/post
    "gemm_fp8":      ["attention", "moe", "mhc", "norm"],  # GEMM spans multiple categories
    "gemm_bf16":     ["attention", "moe", "mhc"],
    "rmsnorm":       ["norm"],
    "quant":         ["norm"],                # FP8 quantization
    "topk":          ["moe"],                 # TopK routing
    "rope":          ["attention"],
    "activation":    ["moe"],                 # silu_mul in MoE
    "other":         [],
    # Aliases from layer_kernel_breakdown.py
    "mla_metadata":  ["attention"],
    "mhc_pre_gemm":  ["mhc"],
    "mhc_pre_fuse":  ["mhc"],
    "mhc_post":      ["mhc"],
    "c4_prefill":    ["attention"],
    "c128_prefill":  ["attention"],
    "moe_gate":      ["moe"],
    "moe_align":     ["moe"],
    "moe_sort":      ["moe"],
    "mla_cache_store": ["attention"],
    "indexer_store": ["attention"],
    "gemm_f32":      [],
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
    "mla":           ["csa_attn_score", "csa_attn_v", "hca_attn_score", "hca_attn_v",
                      "c4_attn", "c4_attn_v", "attn_score", "attn_v"],  # flash attention compute
    "moe":           ["routed_experts_swiglu"],  # fused MoE: gate+up+silu+down
    "mhc_post":      ["mhc_post_tilelang"],
    "mhc_pre_gemm":  ["mhc_pre"],
    "mhc_pre_fuse":  ["mhc_pre"],
    "mhc":           ["mhc_post_tilelang", "mhc_pre"],  # fallback
    "rmsnorm":       ["rmsnorm"],
    "topk":          ["topk"],
    "moe_gate":      ["router"],
    # Direct-match operators (have FLOPs in template)
    "hadamard":      ["hadamard"],
    "indexer":       [],        # trace kernel is fused_store_indexer_cache only; proj compute is in gemm_bf16
    "paged_mqa":     ["paged_mqa"],
    "c4_prefill":    ["c4_attn", "c4_attn_v"],
    "c128_prefill":  ["c4_attn", "c4_attn_v"],
    "rope":          ["rope"],
    "quant":         ["quant"],
    "activation":    ["activation"],
    "mla_cache_store": ["mla_cache_store"],
    # Generic GEMM: distribute by FLOPs to remaining ops
    "gemm_fp8":      None,
    "gemm_bf16":     None,
    # No compute FLOPs (overhead)
    "allreduce":     [],
    "mla_metadata":  [],
    "moe_align":     [],
    "moe_sort":      [],
    "indexer_store": [],
    "gemm_f32":      [],
    "radixsort":     [],
    "other":         [],
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
    kernels = kernel_detail.get("kernels", [])
    cat_summary = kernel_detail.get("category_summary", {})

    # Build op name → list of indices mapping (same name may appear multiple times)
    op_name_to_indices = {}
    for i, op in enumerate(ops):
        op_name_to_indices.setdefault(op.name, []).append(i)

    # Step 1: Direct-match assignment
    op_measured_us = {}       # op_index → measured_us
    direct_matched = {}       # kernel_cat → [op_names]  (for reporting)
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
                indices = [idx for idx in op_name_to_indices[op_name]
                           if idx not in assigned_op_indices]
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
                    op_measured_us[idx] = op_measured_us.get(idx, 0) + cat_dur_us * share
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
    lines.append(f"  Attention heads: {arch.get('num_attention_heads', '?')} (KV: {arch.get('num_key_value_heads', '?')})")
    lines.append(f"  Head dim:        {arch.get('head_dim', '?')}")
    lines.append(f"  Attention type:  {arch.get('attention_type', '?')}")
    if arch.get('q_lora_rank', 0) > 0:
        lines.append(f"  Q LoRA rank:     {arch.get('q_lora_rank')}")
        lines.append(f"  O LoRA rank:     {arch.get('o_lora_rank')}")
    if arch.get('moe', False):
        lines.append(f"  MoE:             {arch.get('num_experts')} experts, top-{arch.get('num_experts_per_tok')}, {arch.get('num_shared_experts')} shared")
    if arch.get('mhc', False):
        lines.append(f"  MHC:             enabled")
    lines.append(f"  Vocab size:      {arch.get('vocab_size', '?')}")

    # ---- 2. Serving Configuration ----
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  SERVING CONFIGURATION")
    lines.append(f"{'=' * 70}")
    lines.append(f"  B={result.batch_size}  S={result.seq_len}  TP={result.tp}  DP={result.dp}  EP={result.ep}")
    lines.append(f"  GPU={result.gpu}  dtype={result.dtype}")

    # Embedding
    if result.embed_flops > 0:
        lines.append(f"[Embedding]  FLOPs: {fmt_flops(result.embed_flops)}")

    # Per-layer (skip when kernel-flow mode — redundant with kernel-flow table)
    if result.layers and not skip_compute_flow:
        # Find the representative layer that per_op_mfu is based on
        repr_idx = 0
        for i, lr in enumerate(result.layers):
            if getattr(lr, 'compress_ratio', 0) == 128:
                repr_idx = i
                break
        l0 = result.layers[repr_idx]
        cr_val = getattr(l0, 'compress_ratio', 0)
        cr_label = f" (compress_ratio={cr_val})" if cr_val else ""
        type_label = {0: "FULL_ATTN", 4: "C4_LIGHT", 128: "C128_HEAVY"}.get(cr_val, "")
        layer_type_str = f' [{type_label}]' if type_label else ''
        lines.append(f"\n--- Layer {repr_idx} (detail){layer_type_str}{cr_label} ---")
        attn_pct = l0.attention_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100
        moe_pct = l0.moe_ffn_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100
        lines.append(f"  Attention FLOPs: {fmt_flops(l0.attention_flops)}  ({attn_pct:.1f}%)")
        lines.append(f"  MoE/FFN FLOPs:   {fmt_flops(l0.moe_ffn_flops)}  ({moe_pct:.1f}%)")
        lines.append(f"  Per-op sequence:")
        for op in l0.ops:
            lines.append(f"    {op.name:30s}  FLOPs: {fmt_flops(op.flops):>16s}  {op.shape_in} -> {op.shape_out}")

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
        lines.append(f"  Embedding:             {fmt_flops(result.embed_flops)}")
    layer_total = result.total_flops - result.embed_flops
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
                lines.append(f"  Dominant: MoE/FFN ({l0.moe_ffn_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100:.1f}%)")
            else:
                lines.append(f"  Dominant: Attention ({l0.attention_flops / max(1, l0.attention_flops + l0.moe_ffn_flops) * 100:.1f}%)")
        if result.mfu_pct < 5:
            lines.append(f"  Note: decode B=1 is memory-bound; low MFU is expected.")

        # Per-operator MFU breakdown with kernel-measured times
        if result.per_op_mfu:
            has_kernel = result.kernel_ms is not None or getattr(result, '_kernel_detail_meta', None) is not None
            lines.append(f"\n  Per-operator MFU detail (TP={result.tp}, EP={result.ep}):")
            if has_kernel:
                lines.append(f"  {'Op':28s} {'Cat':>6s} {'FLOPs':>14s} {'Per-GPU':>14s} {'Theo(μs)':>10s} {'Meas(μs)':>10s} {'MFU%':>7s} {'Layer%':>8s}")
            else:
                lines.append(f"  {'Op':28s} {'Cat':>6s} {'FLOPs':>14s} {'Per-GPU':>14s} {'Theo(μs)':>10s} {'Layer%':>8s}")
            lines.append(f"  {'-'*90}")
            for op in result.per_op_mfu:
                if has_kernel and op.get('measured_us') is not None and op['measured_us'] > 0:
                    mfu_op = op['theo_us'] / op['measured_us'] * 100
                    lines.append(f"  {op['name']:28s} {op['category']:>6s} {fmt_flops(op['flops']):>14s} {fmt_flops(op['flops_per_gpu']):>14s} {op['theo_us']:>9.1f} {op['measured_us']:>9.1f} {mfu_op:>6.1f}% {op['pct_of_layer']:>7.1f}%")
                else:
                    lines.append(f"  {op['name']:28s} {op['category']:>6s} {fmt_flops(op['flops']):>14s} {fmt_flops(op['flops_per_gpu']):>14s} {op['theo_us']:>9.1f} {op['pct_of_layer']:>7.1f}%")

        # Kernel category mapping explanation
        kdm = getattr(result, '_kernel_detail_meta', None)
        if kdm is not None:
            # Precise kernel-detail mapping
            lines.append(f"\n  Kernel-detail → operator mapping (precise, from --kernel-detail):")
            for kcat, op_names in kdm.get("direct_matched", {}).items():
                lines.append(f"    {kcat:20s} → {', '.join(op_names)} (direct)")
            if kdm.get("gemm_distributed"):
                lines.append(f"    {'gemm_fp8/bf16':20s} → {', '.join(kdm['gemm_distributed'])} (FLOPs-proportional)")
            overhead = kdm.get("overhead_us", 0)
            if overhead > 0:
                lines.append(f"    {'overhead':20s} {overhead/1000:8.3f} ms (allreduce/quant/rope/etc.)")
            # Unassigned ops
            if result.per_op_mfu:
                unassigned = [op['name'] for op in result.per_op_mfu if op.get('measured_us') is None]
                if unassigned:
                    lines.append(f"    {'unassigned':20s} {', '.join(unassigned)} (no kernel match)")
        elif result.kernel_ms:
            lines.append(f"\n  Kernel category → operator mapping (from llm-pipeline-analysis):")
            for kcat, dur_ms in sorted(result.kernel_ms.items(), key=lambda x: -x[1]):
                mapped = KERNEL_TO_OP_CATEGORY.get(kcat, [])
                lines.append(f"    {kcat:20s} {dur_ms:8.3f} ms → {', '.join(mapped) if mapped else '(no compute FLOPs)'}")

    # One-line summary
    if result.mfu_pct is not None and result.layers:
        l0 = result.layers[0]
        dom = 'MoE' if l0.moe_ffn_flops > l0.attention_flops else 'Attn'
        lines.append(f"\n  ▸ Summary: MFU={result.mfu_pct:.1f}% | {dom}-dominant | {result.gpu} | TP={result.tp} EP={result.ep}")

    return "\n".join(lines)


def format_kernel_flow(kernel_detail: dict, ops: list, peak_tflops: float,
                       tp: int, ep: int, compress_ratio: int = 0,
                       fp8_peak_tflops: float = 0) -> str:
    """Format kernel-level MFU table that preserves every kernel row.

    This builds on the compute flow from llm-pipeline-analysis and adds:
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
        per_gpu = op.flops / ep if (op.category == "moe" and "routed" in op.name and ep > 0) else op.flops / tp if tp > 0 else op.flops
        theo_us = per_gpu / (peak_tflops * 1e6) if peak_tflops > 0 else 0
        meas_us = op_measured_us.get(i, None)
        mfu = (theo_us / meas_us * 100) if (meas_us and meas_us > 0 and theo_us > 0) else None
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

        eligible_indices = [i for i in range(len(ops))
                          if i not in assigned_op_indices and ops[i].category in gemm_categories]
        if not eligible_indices:
            eligible_indices = [i for i in range(len(ops)) if i not in assigned_op_indices]
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
    type_label = {0: "FULL_ATTN", 4: "C4_LIGHT", 128: "C128_HEAVY"}.get(cr, "")
    type_str = f" [{type_label}]" if type_label else ""

    lines.append(f"\n{'=' * 140}")
    lines.append(f"  Kernel-Level MFU: Layer {layer_id}{type_str}{cr_label}"
                 f"  —  {total_dur:.0f}us ({total_dur/1000:.2f}ms), {len(kernels)} kernels")
    lines.append(f"{'=' * 140}")

    # Header
    lines.append(f"  {'#':>3s} {'Half':>4s} {'Category':<16s} {'Simplified Name':<42s}"
                 f" {'dur(us)':>8s} {'%':>5s} {'Mapped Op':<24s}"
                 f" {'FLOPs':>10s} {'Theo(us)':>9s} {'MFU%':>7s} {'shape_in→out':<22s}")
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
                ref_dur = gemm_total_dur if gemm_total_dur > 0 else cat_total_dur.get(cat_key, dur_us)
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
            effective_peak = fp8_peak_tflops if cat_key in FP8_INTERNAL_KERNEL_CATEGORIES and fp8_peak_tflops > 0 else peak_tflops
            kernel_theo_us = kernel_flops / (effective_peak * 1e6) if effective_peak > 0 else 0
            mfu = (kernel_theo_us / dur_us * 100) if (dur_us > 0 and kernel_theo_us > 0) else None
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

        lines.append(f"  {i:>3d} {half:>4s} {cat_key:<16s} {name:<42s}"
                     f" {dur_us:>7.1f} {pct:>4.1f}% {mapped_name:<24s}"
                     f" {flops_str:>10s} {theo_str:>9s} {mfu_str:>7s} {shape_str:<22s}")

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
        "tp": result.tp,
        "dp": result.dp,
        "ep": result.ep,
        "gpu": result.gpu,
        "dtype": result.dtype,
        "total_flops": result.total_flops,
        "embed_flops": result.embed_flops,
        "layer_template": [
            {
                "name": op.name,
                "flops": op.flops,
                "shape_in": op.shape_in,
                "shape_out": op.shape_out,
                "category": op.category,
            }
            for op in result.layers[0].ops
        ] if result.layers else [],
        "num_layers": len(result.layers),
        "measured_ms": result.measured_ms,
        "mfu_pct": result.mfu_pct,
        "per_layer_mfu_pct": result.per_layer_mfu_pct,
        "per_op_mfu": result.per_op_mfu if result.per_op_mfu else [],
        "model_arch": result.model_arch,
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Model Compute Simulator")
    parser.add_argument("model", nargs="?", help="Model name (e.g. DeepSeek-V4-Flash)")
    parser.add_argument("--list-models", action="store_true", help="List known model IDs")
    parser.add_argument("--list-gpus", action="store_true", help="List known GPU types")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1, help="Sequence length (1 for decode)")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism")
    parser.add_argument("--ep", type=int, default=8, help="Expert parallelism")
    parser.add_argument("--gpu", default="h20", help="GPU type")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp8"], help="Data type")
    parser.add_argument("--measured-ms", type=float, default=None, help="Measured forward-pass latency (ms)")
    parser.add_argument("--per-layer-ms", type=float, default=None, help="Measured per-layer latency (ms); overrides --measured-ms for per-layer MFU")
    parser.add_argument("--kernel-ms", default=None, help="JSON object mapping kernel categories to measured ms (from llm-pipeline-analysis)")
    parser.add_argument("--kernel-detail", default=None,
                        help="JSON string or @file path from layer_kernel_breakdown --format json; provides per-kernel detail for precise per-operator MFU")
    parser.add_argument("--kernel-flow", default=None,
                        help="JSON string or @file path from layer_kernel_breakdown --format json; produces kernel-level MFU table preserving all kernel rows")
    parser.add_argument("--format", dest="fmt", default="text", choices=["text", "json"], help="Output format")

    args = parser.parse_args()

    config_index = load_json(CONFIG_INDEX)
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
            print(f"  {key}: {val['display_name']}  BF16={val['bf16_tflops']} TFLOPS{alias_str}")
        return

    if not args.model:
        parser.error("model name is required (use --list-models to see available models)")

    cfg = resolve_model(args.model, config_index)
    if cfg is None:
        print(f"Error: model '{args.model}' not found in config index.", file=sys.stderr)
        print(f"Use --list-models to see available models.", file=sys.stderr)
        sys.exit(1)

    # If per-layer-ms is provided, compute measured-ms from it
    measured_ms = args.measured_ms
    kernel_ms = None
    kernel_detail = None
    kernel_detail_result = None  # result from map_kernel_detail_to_ops

    if args.kernel_detail is not None:
        # Load kernel detail JSON (from string or @file)
        kd_input = args.kernel_detail
        if kd_input.startswith("@"):
            with open(kd_input[1:], "r") as f:
                kernel_detail = json.load(f)
        else:
            kernel_detail = json.loads(kd_input)
        # Compute measured_ms from kernel detail
        meta = kernel_detail.get("metadata", {})
        layer_dur_ms = meta.get("total_dur_us", 0) / 1000.0
        n_layers = cfg.get("num_hidden_layers", 1)
        measured_ms = layer_dur_ms * n_layers
    elif args.kernel_ms is not None:
        kernel_ms = json.loads(args.kernel_ms)
        # Sum all kernel categories for per-layer measured time
        layer_measured = sum(kernel_ms.values())
        measured_ms = layer_measured * cfg.get("num_hidden_layers", 1)
    elif args.per_layer_ms is not None and cfg is not None:
        n_layers = cfg.get("num_hidden_layers", 1)
        measured_ms = args.per_layer_ms * n_layers

    result = simulate(
        cfg, args.model, args.batch_size, args.seq_len,
        args.tp, args.dp, args.ep, args.gpu, args.dtype, measured_ms,
    )
    # If per-layer-ms was used, show the per-layer measured time in output
    if args.per_layer_ms is not None:
        result.measured_ms = measured_ms  # total
        result._per_layer_measured_ms = args.per_layer_ms

    # If kernel-detail was provided, map measured durations to per-operator (precise)
    kernel_flow_detail = None  # for --kernel-flow output
    if getattr(args, 'kernel_flow', None) is not None:
        # Load kernel flow JSON (same format as kernel-detail)
        kf_input = args.kernel_flow
        if kf_input.startswith("@"):
            with open(kf_input[1:], "r") as f:
                kernel_flow_detail = json.load(f)
        else:
            kernel_flow_detail = json.loads(kf_input)
        # Compute measured_ms from kernel detail
        meta = kernel_flow_detail.get("metadata", {})
        layer_dur_ms = meta.get("total_dur_us", 0) / 1000.0
        n_layers_kf = cfg.get("num_hidden_layers", 1)
        measured_ms = layer_dur_ms * n_layers_kf
    elif kernel_detail is not None:
        result.kernel_detail = kernel_detail
        if result.layers and result.per_op_mfu:
            # Use the same representative layer that per_op_mfu is based on
            repr_layer = None
            for lr in result.layers:
                if getattr(lr, 'compress_ratio', 0) == 128:
                    repr_layer = lr
                    break
            if repr_layer is None and result.layers:
                repr_layer = result.layers[0]
            if repr_layer is not None:
                kernel_detail_result = map_kernel_detail_to_ops(
                    kernel_detail, repr_layer.ops, args.tp, args.ep)
                for i, measured_us in kernel_detail_result["op_measured_us"].items():
                    if i < len(result.per_op_mfu):
                        result.per_op_mfu[i]["measured_us"] = measured_us
                result._kernel_detail_meta = kernel_detail_result
    elif kernel_ms is not None:
        result.kernel_ms = kernel_ms
        if result.layers and result.per_op_mfu:
            repr_layer = None
            for lr in result.layers:
                if getattr(lr, 'compress_ratio', 0) == 128:
                    repr_layer = lr
                    break
            if repr_layer is None and result.layers:
                repr_layer = result.layers[0]
            if repr_layer is not None:
                op_measured = map_kernel_ms_to_ops(kernel_ms, repr_layer.ops, args.tp, args.ep)
                for i, measured_us in op_measured.items():
                    if i < len(result.per_op_mfu):
                        result.per_op_mfu[i]["measured_us"] = measured_us

    if args.fmt == "json":
        print(format_json(result))
    else:
        print(format_text(result, skip_compute_flow=kernel_flow_detail is not None))

    # Kernel-flow output (appended after the main output)
    # When kernel-flow is used, skip the per-op static template in format_text
    if kernel_flow_detail is not None:
        # Find the representative layer matching the kernel detail
        cr_meta = kernel_flow_detail.get("metadata", {}).get("compress_ratio", 0)
        repr_layer = None
        for lr in result.layers:
            if getattr(lr, 'compress_ratio', 0) == cr_meta:
                repr_layer = lr
                break
        if repr_layer is None and result.layers:
            repr_layer = result.layers[0]

        if repr_layer is not None:
            gpu_info = resolve_gpu(args.gpu, gpu_specs) if args.gpu else None
            peak_tflops = 0
            fp8_peak_tflops = 0
            if gpu_info:
                dtype_key = "bf16_tflops" if args.dtype == "bf16" else "fp8_tflops"
                peak_tflops = gpu_info.get(dtype_key, 0)
                fp8_peak_tflops = gpu_info.get("fp8_tflops", 0)

            print(format_kernel_flow(
                kernel_flow_detail, repr_layer.ops, peak_tflops,
                args.tp, args.ep, compress_ratio=cr_meta,
                fp8_peak_tflops=fp8_peak_tflops))


if __name__ == "__main__":
    main()
