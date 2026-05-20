#!/usr/bin/env python3
"""LLM Serving Capacity Analyzer — parse serving logs, decompose GPU memory, estimate max concurrency.

Usage:
  # Analyze from log file (SGLang)
  python3 capacity_analyzer.py --log-file /path/to/sglang.log

  # With nvidia-smi per-rank data
  python3 capacity_analyzer.py --log-file /path/to/sglang.log --nvidia-smi-file /path/to/smi.txt

  # With model config for KV cache byte estimation
  python3 capacity_analyzer.py --log-file /path/to/sglang.log --config-json /path/to/config.json

  # Specify GPU type (if not inferable from log)
  python3 capacity_analyzer.py --log-file /path/to/sglang.log --gpu h20

  # JSON output for automation
  python3 capacity_analyzer.py --log-file /path/to/sglang.log --format json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths — reuse model-compute-simulation references as read-only
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.join(SCRIPT_DIR, "..")
SHARED_REF = os.path.join(SCRIPT_DIR, "..", "..", "model-compute-simulation", "references")
GPU_SPECS_PATH = os.path.join(SHARED_REF, "gpu-specs.json")

# ---------------------------------------------------------------------------
# GPU specs loader
# ---------------------------------------------------------------------------

def load_gpu_specs() -> Dict:
    """Load GPU specs from model-compute-simulation references."""
    if os.path.exists(GPU_SPECS_PATH):
        with open(GPU_SPECS_PATH) as f:
            return json.load(f)
    return {}


GPU_ALIAS = {
    "h20": "h20",
    "h20-sxm": "h20",
    "l20n": "l20n",
    "l20": "l20n",
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
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServerArgs:
    """Key serving parameters extracted from server_args line."""
    model_path: str = ""
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    mem_fraction_static: float = 0.88
    kv_cache_dtype: str = "auto"
    cuda_graph_max_bs: int = 0
    disable_radix_cache: bool = False
    context_length: Optional[int] = None
    page_size: int = 1
    swa_full_tokens_ratio: Optional[float] = None


@dataclass
class MemCheckpoint:
    """A memory data point extracted from the log."""
    rank: int
    label: str          # e.g. "load_weight_begin", "memory_pool_end", "cuda_graph_end"
    avail_gb: float
    extra: Dict = field(default_factory=dict)


@dataclass
class MemoryProfiling:
    """Parsed from 'Memory profiling:' line (newer sglang versions)."""
    rank: int
    available_gpu_memory_gb: float
    total_gpu_memory_gb: float
    mem_fraction_static: float
    rest_memory_gb: float


@dataclass
class SwKvMemoryCalc:
    """Parsed from 'SW KV memory calculation:' line (SWA models)."""
    rank: int
    bytes_per_full_token: float
    available_bytes_gb: float
    full_token: int


@dataclass
class CudaGraphInfo:
    """Parsed from 'Capture cuda graph end.' line."""
    rank: int
    mem_usage_gb: float
    avail_gb: float
    elapsed_s: float = 0.0


@dataclass
class FinalInfo:
    """Parsed from the final 'max_total_num_tokens=' line."""
    max_total_num_tokens: int = 0
    chunked_prefill_size: int = 0
    max_prefill_tokens: int = 0
    max_running_requests: int = 0
    context_len: int = 0
    available_gpu_mem_gb: float = 0.0


@dataclass
class NvidiaSmiEntry:
    """One row from nvidia-smi query output."""
    index: int
    memory_used_mib: int
    memory_free_mib: int


@dataclass
class ModelConfig:
    """Key model parameters from config.json for KV cache calculation."""
    num_hidden_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    attention_type: str = "gqa"  # "gqa", "mla", "csa_hca"
    # MLA specific
    kv_lora_rank: int = 0        # aka compress_dim / q_lora_rank for MLA
    # SWA/CSA/HCA specific
    compress_ratios: List = field(default_factory=list)
    mhc_bottleneck_dim: int = 0
    # MoE
    moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_server_args(line: str) -> ServerArgs:
    """Extract key serving parameters from the server_args= line."""
    args = ServerArgs()

    m = re.search(r"model_path='([^']*)'", line)
    if m:
        args.model_path = m.group(1)

    for param, attr, conv in [
        (r"tp_size=(\d+)", "tp_size", int),
        (r"pp_size=(\d+)", "pp_size", int),
        (r"dp_size=(\d+)", "dp_size", int),
        (r"ep_size=(\d+)", "ep_size", int),
        (r"mem_fraction_static=([\d.]+)", "mem_fraction_static", float),
        (r"kv_cache_dtype='([^']*)'", "kv_cache_dtype", str),
        (r"cuda_graph_max_bs=(\d+)", "cuda_graph_max_bs", int),
        (r"page_size=(\d+)", "page_size", int),
    ]:
        m = re.search(param, line)
        if m:
            setattr(args, attr, conv(m.group(1)))

    if "disable_radix_cache=True" in line:
        args.disable_radix_cache = True

    m = re.search(r"context_length=(\d+)", line)
    if m and m.group(1) != "None":
        args.context_length = int(m.group(1))

    m = re.search(r"swa_full_tokens_ratio=([\d.]+)", line)
    if m and m.group(1) != "None":
        args.swa_full_tokens_ratio = float(m.group(1))

    return args


def parse_load_weight_begin(line: str) -> Optional[MemCheckpoint]:
    """Parse: [timestamp TP0] Load weight begin. avail mem=93.61 GB"""
    m = re.search(r"\[.*TP(\d+)\]\s+Load weight begin\.\s+avail mem=([\d.]+)\s+GB", line)
    if m:
        return MemCheckpoint(rank=int(m.group(1)), label="load_weight_begin", avail_gb=float(m.group(2)))
    return None


def parse_memory_profiling(line: str) -> Optional[MemoryProfiling]:
    """Parse: [timestamp TP0] Memory profiling: available_gpu_memory=57.01 GB, total_gpu_memory=93.58 GB, mem_fraction_static=0.60, rest_memory=19.58 GB"""
    m = re.search(
        r"\[.*TP(\d+)\]\s+Memory profiling:\s+"
        r"available_gpu_memory=([\d.]+)\s+GB,\s+"
        r"total_gpu_memory=([\d.]+)\s+GB,\s+"
        r"mem_fraction_static=([\d.]+),\s+"
        r"rest_memory=([\d.]+)\s+GB",
        line,
    )
    if m:
        return MemoryProfiling(
            rank=int(m.group(1)),
            available_gpu_memory_gb=float(m.group(2)),
            total_gpu_memory_gb=float(m.group(3)),
            mem_fraction_static=float(m.group(4)),
            rest_memory_gb=float(m.group(5)),
        )
    return None


def parse_sw_kv_memory_calc(line: str) -> Optional[SwKvMemoryCalc]:
    """Parse: [timestamp TP0] SW KV memory calculation: bytes_per_full_token=15955.85, available_bytes=19.58 GB, full_token=1317632

    Also matches legacy format: [timestamp TP0] DSv4 memory calculation: ...
    """
    m = re.search(
        r"\[.*TP(\d+)\]\s+(?:DSv4|SW KV) memory calculation:\s+"
        r"bytes_per_full_token=([\d.]+),\s+"
        r"available_bytes=([\d.]+)\s+GB,\s+"
        r"full_token=(\d+)",
        line,
    )
    if m:
        return SwKvMemoryCalc(
            rank=int(m.group(1)),
            bytes_per_full_token=float(m.group(2)),
            available_bytes_gb=float(m.group(3)),
            full_token=int(m.group(4)),
        )
    return None


def parse_memory_pool_end(line: str) -> Optional[MemCheckpoint]:
    """Parse: [timestamp TP0] Memory pool end. avail mem=10.18 GB"""
    m = re.search(r"\[.*TP(\d+)\]\s+Memory pool end\.\s+avail mem=([\d.]+)\s+GB", line)
    if m:
        return MemCheckpoint(rank=int(m.group(1)), label="memory_pool_end", avail_gb=float(m.group(2)))
    return None


def parse_cuda_graph_end(line: str) -> Optional[CudaGraphInfo]:
    """Parse: [timestamp TP0] Capture cuda graph end. Time elapsed: 54.61 s. mem usage=1.93 GB. avail mem=8.16 GB."""
    m = re.search(
        r"\[.*TP(\d+)\]\s+Capture cuda graph end\.\s+"
        r"Time elapsed:\s+([\d.]+)\s+s\.\s+"
        r"mem usage=([\d.]+)\s+GB\.\s+"
        r"avail mem=([\d.]+)\s+GB",
        line,
    )
    if m:
        return CudaGraphInfo(
            rank=int(m.group(1)),
            elapsed_s=float(m.group(2)),
            mem_usage_gb=float(m.group(3)),
            avail_gb=float(m.group(4)),
        )
    return None


def parse_max_total_tokens(line: str) -> Optional[FinalInfo]:
    """Parse: [timestamp TP0] max_total_num_tokens=3080960, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=256, context_len=1048576, available_gpu_mem=8.16 GB"""
    m = re.search(
        r"max_total_num_tokens=(\d+),\s+"
        r"chunked_prefill_size=(\d+),\s+"
        r"max_prefill_tokens=(\d+),\s+"
        r"max_running_requests=(\d+),\s+"
        r"context_len=(\d+),\s+"
        r"available_gpu_mem=([\d.]+)\s+GB",
        line,
    )
    if m:
        return FinalInfo(
            max_total_num_tokens=int(m.group(1)),
            chunked_prefill_size=int(m.group(2)),
            max_prefill_tokens=int(m.group(3)),
            max_running_requests=int(m.group(4)),
            context_len=int(m.group(5)),
            available_gpu_mem_gb=float(m.group(6)),
        )
    return None


def parse_nvidia_smi(text: str) -> List[NvidiaSmiEntry]:
    """Parse nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader output."""
    entries = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                idx = int(parts[0])
                used = int(re.sub(r"\s*MiB", "", parts[1]))
                free = int(re.sub(r"\s*MiB", "", parts[2]))
                entries.append(NvidiaSmiEntry(index=idx, memory_used_mib=used, memory_free_mib=free))
            except (ValueError, IndexError):
                continue
    return entries


# ---------------------------------------------------------------------------
# Log processing
# ---------------------------------------------------------------------------

@dataclass
class ParsedLog:
    """All extracted data from a serving log."""
    server_args: Optional[ServerArgs] = None
    checkpoints: List[MemCheckpoint] = field(default_factory=list)
    memory_profilings: List[MemoryProfiling] = field(default_factory=list)
    sw_kv_calcs: List[SwKvMemoryCalc] = field(default_factory=list)
    cuda_graphs: List[CudaGraphInfo] = field(default_factory=list)
    final_info: Optional[FinalInfo] = None
    framework: str = "unknown"  # "sglang" or "vllm"


def parse_log(text: str) -> ParsedLog:
    """Parse an entire serving log and extract all memory-related data points."""
    result = ParsedLog()

    for line in text.splitlines():
        # Detect framework
        if result.framework == "unknown":
            if "sglang" in line.lower() or "server_args=ServerArgs" in line:
                result.framework = "sglang"
        if result.framework == "unknown":
            if "vllm" in line.lower() or "vllm_engine" in line.lower():
                result.framework = "vllm"

        # server_args (only parse once, from the first occurrence)
        if result.server_args is None and "server_args=ServerArgs(" in line:
            result.server_args = parse_server_args(line)

        # Load weight begin
        ck = parse_load_weight_begin(line)
        if ck:
            result.checkpoints.append(ck)

        # Memory profiling (newer sglang)
        mp = parse_memory_profiling(line)
        if mp:
            result.memory_profilings.append(mp)

        # SW KV memory calculation
        dc = parse_sw_kv_memory_calc(line)
        if dc:
            result.sw_kv_calcs.append(dc)

        # Memory pool end
        ck = parse_memory_pool_end(line)
        if ck:
            result.checkpoints.append(ck)

        # CUDA graph end
        cg = parse_cuda_graph_end(line)
        if cg:
            result.cuda_graphs.append(cg)

        # max_total_num_tokens
        fi = parse_max_total_tokens(line)
        if fi:
            result.final_info = fi

    return result


# ---------------------------------------------------------------------------
# Model config loader
# ---------------------------------------------------------------------------

def load_model_config(config_json_path: str) -> ModelConfig:
    """Load model parameters from a HuggingFace config.json."""
    with open(config_json_path) as f:
        cfg = json.load(f)

    mc = ModelConfig()
    mc.num_hidden_layers = cfg.get("num_hidden_layers", 0)
    mc.hidden_size = cfg.get("hidden_size", 0)
    mc.num_attention_heads = cfg.get("num_attention_heads", 0)
    mc.num_key_value_heads = cfg.get("num_key_value_heads", mc.num_attention_heads)
    mc.head_dim = cfg.get("head_dim", cfg.get("hidden_size", 0) // cfg.get("num_attention_heads", 1))
    mc.moe = cfg.get("is_moe", False) or "num_experts" in cfg
    mc.num_experts = cfg.get("num_experts", 0)
    mc.num_experts_per_tok = cfg.get("num_experts_per_tok", cfg.get("num_selected_experts", 0))

    # MLA (DeepSeek-V3 style)
    mc.kv_lora_rank = cfg.get("kv_lora_rank", 0)
    if mc.kv_lora_rank > 0:
        mc.attention_type = "mla"

    # SWA/CSA/HCA
    mc.compress_ratios = cfg.get("compress_ratios", [])
    if mc.compress_ratios:
        mc.attention_type = "csa_hca"
    mc.mhc_bottleneck_dim = cfg.get("mhc_bottleneck_dim", 0)

    return mc


# ---------------------------------------------------------------------------
# KV cache byte calculation
# ---------------------------------------------------------------------------

def kv_dtype_bytes(dtype_str: str) -> int:
    """Return bytes per element for a KV cache dtype string."""
    dtype_str = dtype_str.lower().strip("'\"")
    if "fp8" in dtype_str or "e4m3" in dtype_str or "e5m2" in dtype_str:
        return 1
    if "fp16" in dtype_str or "bf16" in dtype_str or "float16" in dtype_str or "bfloat16" in dtype_str:
        return 2
    if "fp32" in dtype_str or "float32" in dtype_str:
        return 4
    return 2  # default to bf16


def calc_kv_bytes_per_token(
    mc: ModelConfig,
    tp_size: int,
    kv_dtype_bytes: int = 2,
) -> float:
    """Calculate per-token KV cache bytes (per GPU card) based on model architecture.

    Returns bytes per token per GPU.
    """
    L = mc.num_hidden_layers

    if mc.attention_type == "mla":
        # MLA: KV is the compressed latent, replicated when kv_heads < tp
        # per_token = 2 * L * kv_lora_rank * dtype_bytes
        # With TP, MLA latent is replicated (not split) when kv_lora_rank < tp
        per_token = 2 * L * mc.kv_lora_rank * kv_dtype_bytes
        return per_token  # MLA latent is replicated, no TP division

    if mc.attention_type == "csa_hca":
        # SWA/CSA/HCA: the actual per-token bytes are calculated by the framework
        # with SWA compression ratios. We cannot accurately compute this without
        # the full compress_ratios and SWA window sizes. Return 0 to indicate
        # that the log's bytes_per_full_token should be used instead.
        return 0.0

    # Standard GQA/MHA:
    # per_token = 2 * L * kv_heads * head_dim * dtype_bytes / tp_size
    # But when kv_heads < tp_size, KV is replicated, so no TP division for KV
    if mc.num_key_value_heads >= tp_size:
        kv_heads_per_gpu = mc.num_key_value_heads / tp_size
    else:
        kv_heads_per_gpu = mc.num_key_value_heads  # replicated

    per_token = 2 * L * kv_heads_per_gpu * mc.head_dim * kv_dtype_bytes
    return per_token


# ---------------------------------------------------------------------------
# Memory decomposition
# ---------------------------------------------------------------------------

@dataclass
class MemoryBreakdown:
    """Decomposed GPU memory for a single rank."""
    rank: int = 0

    # Source data
    gpu_hbm_gib: float = 0.0      # total GPU HBM from spec
    gpu_hbm_mib: float = 0.0      # total GPU HBM in MiB

    # Decomposed categories (GiB)
    framework_overhead_gib: float = 0.0
    model_weights_gib: float = 0.0
    kv_pool_gib: float = 0.0
    cuda_graph_gib: float = 0.0
    other_gib: float = 0.0

    # nvidia-smi actual (MiB)
    smi_used_mib: int = 0
    smi_free_mib: int = 0

    # Derived
    total_used_gib: float = 0.0
    weight_pct: float = 0.0
    kv_pool_pct: float = 0.0
    cuda_graph_pct: float = 0.0
    framework_pct: float = 0.0
    other_pct: float = 0.0

    # Derivation notes
    derivation: Dict[str, str] = field(default_factory=dict)


def decompose_memory(
    parsed: ParsedLog,
    gpu_hbm_gib: float,
    smi_entries: Optional[List[NvidiaSmiEntry]] = None,
    target_rank: int = 0,
) -> MemoryBreakdown:
    """Decompose GPU memory into categories for a target rank.

    Two code paths depending on whether 'Memory profiling' lines exist.
    """
    bd = MemoryBreakdown()
    bd.rank = target_rank
    bd.gpu_hbm_gib = gpu_hbm_gib
    bd.gpu_hbm_mib = gpu_hbm_gib * 1024

    # Find nvidia-smi data for this rank
    if smi_entries:
        for e in smi_entries:
            if e.index == target_rank:
                bd.smi_used_mib = e.memory_used_mib
                bd.smi_free_mib = e.memory_free_mib
                break

    # Find load_weight_begin for target rank
    avail_before_weight = None
    for ck in parsed.checkpoints:
        if ck.label == "load_weight_begin" and ck.rank == target_rank:
            avail_before_weight = ck.avail_gb
            break
    # Fallback: use rank 0
    if avail_before_weight is None:
        for ck in parsed.checkpoints:
            if ck.label == "load_weight_begin":
                avail_before_weight = ck.avail_gb
                break

    # Find memory_pool_end for target rank
    avail_after_pool = None
    for ck in parsed.checkpoints:
        if ck.label == "memory_pool_end" and ck.rank == target_rank:
            avail_after_pool = ck.avail_gb
            break
    if avail_after_pool is None:
        for ck in parsed.checkpoints:
            if ck.label == "memory_pool_end":
                avail_after_pool = ck.avail_gb
                break

    # Find CUDA graph for target rank
    cg_info = None
    for cg in parsed.cuda_graphs:
        if cg.rank == target_rank:
            cg_info = cg
            break
    if cg_info is None and parsed.cuda_graphs:
        cg_info = parsed.cuda_graphs[0]

    # Framework overhead
    if avail_before_weight is not None:
        bd.framework_overhead_gib = gpu_hbm_gib - avail_before_weight
        bd.derivation["framework_overhead"] = f"{gpu_hbm_gib:.2f} - {avail_before_weight:.2f} (HBM - avail_before_weight)"
    else:
        bd.derivation["framework_overhead"] = "avail_before_weight not found in log"

    # KV pool size
    kv_pool_gb = None

    # Method 1: from Memory profiling line
    mp_info = None
    for mp in parsed.memory_profilings:
        if mp.rank == target_rank:
            mp_info = mp
            break
    if mp_info is None and parsed.memory_profilings:
        mp_info = parsed.memory_profilings[0]

    if mp_info:
        kv_pool_gb = mp_info.rest_memory_gb
        bd.derivation["kv_pool"] = f"rest_memory from Memory profiling line (rank {mp_info.rank})"

    # Method 2: from SW KV memory calculation
    if kv_pool_gb is None:
        sw_info = None
        for dc in parsed.sw_kv_calcs:
            if dc.rank == target_rank:
                sw_info = dc
                break
        if sw_info is None and parsed.sw_kv_calcs:
            sw_info = parsed.sw_kv_calcs[0]
        if sw_info:
            kv_pool_gb = sw_info.available_bytes_gb
            bd.derivation["kv_pool"] = f"available_bytes from SW KV memory calculation (rank {sw_info.rank})"

    if kv_pool_gb is not None:
        bd.kv_pool_gib = kv_pool_gb

    # Model weights
    if avail_before_weight is not None and avail_after_pool is not None and kv_pool_gb is not None:
        # weight + kv_pool = avail_before_weight - avail_after_pool
        # weight = (avail_before_weight - avail_after_pool) - kv_pool
        bd.model_weights_gib = (avail_before_weight - avail_after_pool) - kv_pool_gb
        bd.derivation["model_weights"] = (
            f"({avail_before_weight:.2f} - {avail_after_pool:.2f}) - {kv_pool_gb:.2f} "
            f"= (avail_before_weight - avail_after_pool) - kv_pool"
        )
    elif mp_info and avail_before_weight is not None:
        # With Memory profiling: weight = avail_before_weight - total_gpu_memory_after_weight
        # But total_gpu_memory in mp is the same as avail_before_weight minus weight,
        # so: weight = avail_before_weight - mp.total_gpu_memory_gb (if they differ)
        # Actually: mp.total_gpu_memory is avail mem AFTER weight loading, before KV pool
        bd.model_weights_gib = avail_before_weight - mp_info.total_gpu_memory_gb
        bd.derivation["model_weights"] = (
            f"{avail_before_weight:.2f} - {mp_info.total_gpu_memory_gb:.2f} "
            f"= avail_before_weight - total_gpu_memory (from Memory profiling)"
        )
    else:
        bd.derivation["model_weights"] = "insufficient data to calculate"

    # CUDA Graph
    if cg_info:
        bd.cuda_graph_gib = cg_info.mem_usage_gb
        bd.derivation["cuda_graph"] = f"reported mem_usage from Capture cuda graph end (rank {cg_info.rank})"

    # Total used
    bd.total_used_gib = (
        bd.framework_overhead_gib
        + bd.model_weights_gib
        + bd.kv_pool_gib
        + bd.cuda_graph_gib
    )

    # Other (from nvidia-smi residual)
    if bd.smi_used_mib > 0:
        smi_used_gib = bd.smi_used_mib / 1024.0
        bd.other_gib = smi_used_gib - bd.total_used_gib
        if bd.other_gib < 0:
            bd.other_gib = 0.0
        bd.total_used_gib = smi_used_gib
        bd.derivation["other"] = f"{smi_used_gib:.2f} - {bd.total_used_gib - bd.other_gib:.2f} (smi_used - sum_of_known)"
    else:
        bd.derivation["other"] = "nvidia-smi data not available"

    # Percentages
    total = bd.total_used_gib if bd.total_used_gib > 0 else 1.0
    bd.weight_pct = bd.model_weights_gib / total * 100
    bd.kv_pool_pct = bd.kv_pool_gib / total * 100
    bd.cuda_graph_pct = bd.cuda_graph_gib / total * 100
    bd.framework_pct = bd.framework_overhead_gib / total * 100
    bd.other_pct = bd.other_gib / total * 100

    return bd


# ---------------------------------------------------------------------------
# Concurrency estimation
# ---------------------------------------------------------------------------

@dataclass
class ConcurrencyEstimate:
    """Max concurrent request estimates under different scenarios."""
    request_tokens: int
    max_total_num_tokens: int
    max_running_requests: int
    max_concurrent: int        # min(max_total_num_tokens / request_tokens, max_running_requests)
    kv_pool_gib: float
    bytes_per_full_token: float
    kv_dtype: str


def estimate_concurrency(
    final_info: FinalInfo,
    kv_pool_gib: float,
    bytes_per_full_token: float,
    kv_dtype: str,
    request_tokens_list: Optional[List[int]] = None,
) -> List[ConcurrencyEstimate]:
    """Estimate max concurrent requests for different token lengths."""
    if request_tokens_list is None:
        request_tokens_list = [4096, 6144, 8192]

    results = []
    for rt in request_tokens_list:
        token_limit = final_info.max_total_num_tokens // rt if rt > 0 else 0
        max_conc = min(token_limit, final_info.max_running_requests)
        results.append(ConcurrencyEstimate(
            request_tokens=rt,
            max_total_num_tokens=final_info.max_total_num_tokens,
            max_running_requests=final_info.max_running_requests,
            max_concurrent=max_conc,
            kv_pool_gib=kv_pool_gib,
            bytes_per_full_token=bytes_per_full_token,
            kv_dtype=kv_dtype,
        ))
    return results


# ---------------------------------------------------------------------------
# GPU identification
# ---------------------------------------------------------------------------

def infer_gpu_from_log(parsed: ParsedLog, gpu_specs: Dict) -> Optional[str]:
    """Try to infer the GPU model from the log content."""
    # Check for device_name in MoE config warnings
    for line_text in []:  # we don't have raw lines here, rely on --gpu flag
        pass

    # If avail_before_weight is known, try to match GPU HBM
    avail_before = None
    for ck in parsed.checkpoints:
        if ck.label == "load_weight_begin":
            avail_before = ck.avail_gb
            break

    if avail_before is not None:
        for gpu_key, spec in gpu_specs.items():
            hbm = spec.get("hbm_gb", 0)
            # avail should be slightly less than HBM (framework overhead ~2GB)
            if abs(hbm - avail_before) < 5.0 and hbm > avail_before:
                return gpu_key

    return None


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_breakdown_table(bd: MemoryBreakdown) -> str:
    """Format memory breakdown as a text table."""
    lines = []
    lines.append("=" * 90)
    lines.append(f"Memory Breakdown — Rank {bd.rank}")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"{'Category':<28} {'GiB':>8} {'MiB':>10} {'%':>7}  {'Derivation'}")
    lines.append("-" * 90)

    rows = [
        ("Model Weights", bd.model_weights_gib, bd.derivation.get("model_weights", "")),
        ("KV Cache Pool", bd.kv_pool_gib, bd.derivation.get("kv_pool", "")),
        ("CUDA Graph", bd.cuda_graph_gib, bd.derivation.get("cuda_graph", "")),
        ("Framework/NCCL/Driver", bd.framework_overhead_gib, bd.derivation.get("framework_overhead", "")),
        ("Other (DeepGEMM etc.)", bd.other_gib, bd.derivation.get("other", "")),
    ]

    total_gib = 0.0
    for name, gib, deriv in rows:
        mib = gib * 1024
        total_gib += gib
        pct = gib / bd.total_used_gib * 100 if bd.total_used_gib > 0 else 0
        lines.append(f"{name:<28} {gib:>8.2f} {mib:>10.0f} {pct:>6.1f}%  {deriv}")

    lines.append("-" * 90)
    lines.append(f"{'TOTAL':<28} {bd.total_used_gib:>8.2f} {bd.total_used_gib * 1024:>10.0f} {'100.0%':>7}")
    if bd.smi_used_mib > 0:
        lines.append(f"  nvidia-smi used: {bd.smi_used_mib} MiB = {bd.smi_used_mib / 1024:.2f} GiB")
    lines.append(f"  GPU HBM total:   {bd.gpu_hbm_gib:.2f} GiB")
    lines.append(f"  GPU HBM free:    {bd.smi_free_mib / 1024:.2f} GiB" if bd.smi_free_mib > 0 else "")
    lines.append("")
    return "\n".join(lines)


def format_smi_table(smi_entries: List[NvidiaSmiEntry]) -> str:
    """Format nvidia-smi per-rank data as a text table."""
    if not smi_entries:
        return ""

    lines = []
    lines.append("=" * 60)
    lines.append("nvidia-smi Per-Rank Comparison")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'Rank':>4}  {'Used (MiB)':>12}  {'Used (GiB)':>12}  {'Free (MiB)':>12}")
    lines.append("-" * 60)

    for e in smi_entries:
        lines.append(f"{e.index:>4}  {e.memory_used_mib:>12}  {e.memory_used_mib / 1024:>12.2f}  {e.memory_free_mib:>12}")

    lines.append("-" * 60)
    avg = sum(e.memory_used_mib for e in smi_entries) / len(smi_entries)
    lines.append(f"{'Avg':>4}  {avg:>12.0f}  {avg / 1024:>12.2f}")
    lines.append("")
    return "\n".join(lines)


def format_concurrency_table(estimates: List[ConcurrencyEstimate], kv_dtype: str) -> str:
    """Format concurrency estimates as a text table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Max Concurrent Requests Estimate (KV dtype: {kv_dtype})")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Req Tokens':>12}  {'Token Limit':>12}  {'Req Limit':>12}  {'Max Concurrent':>15}")
    lines.append("-" * 60)

    for est in estimates:
        token_limit = est.max_total_num_tokens // est.request_tokens if est.request_tokens > 0 else 0
        lines.append(
            f"{est.request_tokens:>12}  {token_limit:>12}  {est.max_running_requests:>12}  {est.max_concurrent:>15}"
        )

    lines.append("")
    if estimates:
        e0 = estimates[0]
        lines.append(f"  max_total_num_tokens = {e0.max_total_num_tokens}")
        lines.append(f"  max_running_requests = {e0.max_running_requests}")
        if e0.bytes_per_full_token > 0:
            lines.append(f"  bytes_per_full_token = {e0.bytes_per_full_token:.2f} ({e0.bytes_per_full_token / 1024:.2f} KB)")
        lines.append(f"  KV pool size         = {e0.kv_pool_gib:.2f} GiB")
    lines.append("")
    return "\n".join(lines)


def format_kv_pool_detail(
    parsed: ParsedLog,
    bd: MemoryBreakdown,
    mc: Optional[ModelConfig],
    kv_dtype: str,
    tp_size: int,
) -> str:
    """Format KV pool configuration detail."""
    lines = []
    lines.append("=" * 80)
    lines.append("KV Cache Pool Detail")
    lines.append("=" * 80)
    lines.append("")

    if parsed.sw_kv_calcs:
        dc = parsed.sw_kv_calcs[0]
        lines.append(f"  Framework:          SGLang (SWA mode)")
        lines.append(f"  bytes_per_full_token: {dc.bytes_per_full_token:.2f} bytes ({dc.bytes_per_full_token / 1024:.2f} KB)")
        lines.append(f"  KV Pool capacity:   {dc.available_bytes_gb:.2f} GiB")
        lines.append(f"  full_token count:   {dc.full_token}")
        lines.append(f"  KV dtype:           {kv_dtype}")
        if mc:
            lines.append(f"  Attention type:     {mc.attention_type}")
            if mc.compress_ratios:
                lines.append(f"  compress_ratios:    {mc.compress_ratios[:10]}{'...' if len(mc.compress_ratios) > 10 else ''}")
                lines.append(f"  (SWA compression: theoretical KV >> actual due to sliding window)")
    elif parsed.memory_profilings:
        mp = parsed.memory_profilings[0]
        lines.append(f"  Framework:          SGLang")
        lines.append(f"  mem_fraction_static: {mp.mem_fraction_static}")
        lines.append(f"  available_gpu_mem:  {mp.available_gpu_memory_gb:.2f} GiB (after weight loading)")
        lines.append(f"  rest_memory (KV):   {mp.rest_memory_gb:.2f} GiB")
        lines.append(f"  KV dtype:           {kv_dtype}")
    else:
        lines.append(f"  KV Pool size:       {bd.kv_pool_gib:.2f} GiB (from log)")
        lines.append(f"  KV dtype:           {kv_dtype}")

    # KV head replication
    if mc and mc.num_key_value_heads > 0:
        lines.append("")
        lines.append(f"  kv_heads:           {mc.num_key_value_heads}")
        lines.append(f"  tp_size:            {tp_size}")
        if mc.num_key_value_heads < tp_size:
            replication = tp_size // mc.num_key_value_heads
            lines.append(f"  Replication factor: {replication}x (kv_heads < tp_size, KV replicated across all TP ranks)")
        else:
            lines.append(f"  KV split across TP ranks (kv_heads >= tp_size)")

    # Manual KV bytes calculation
    if mc and mc.attention_type not in ("csa_hca",):
        kv_bytes = kv_dtype_bytes(kv_dtype)
        per_token = calc_kv_bytes_per_token(mc, tp_size, kv_bytes)
        if per_token > 0:
            lines.append("")
            lines.append(f"  Theoretical per-token KV: {per_token:.0f} bytes ({per_token / 1024:.2f} KB)")
            lines.append(f"    = 2 x {mc.num_hidden_layers} layers x kv_heads_per_gpu x {mc.head_dim} head_dim x {kv_bytes} bytes")

    lines.append("")
    return "\n".join(lines)


def format_tuning_suggestions(
    bd: MemoryBreakdown,
    parsed: ParsedLog,
    kv_dtype: str,
    gpu_hbm_gib: float,
    mc: Optional[ModelConfig] = None,
    tp_size: int = 1,
) -> str:
    """Format tuning suggestions."""
    lines = []
    lines.append("=" * 80)
    lines.append("Tuning Suggestions")
    lines.append("=" * 80)
    lines.append("")

    avail_free_gib = bd.smi_free_mib / 1024.0 if bd.smi_free_mib > 0 else 0
    mfs = parsed.server_args.mem_fraction_static if parsed.server_args else 0.88

    suggestions = []

    # Check if there's significant free memory
    if avail_free_gib > 10:
        suggestions.append(
            f"1. Free GPU memory: {avail_free_gib:.2f} GiB — consider increasing --mem-fraction-static\n"
            f"   Current mfs={mfs}, suggest trying mfs={min(mfs + 0.1, 0.95):.2f}\n"
            f"   Expected KV pool increase: ~{avail_free_gib * 0.8:.1f} GiB"
        )

    # FP8 KV suggestion
    if "fp8" not in kv_dtype.lower():
        est_tokens = parsed.final_info.max_total_num_tokens * 2 if parsed.final_info else "N/A"
        suggestions.append(
            f"2. Switch to fp8 KV cache (--kv-cache-dtype fp8_e4m3)\n"
            f"   Current KV dtype: {kv_dtype}\n"
            f"   FP8 KV would roughly double the KV pool capacity\n"
            f"   Expected max_total_num_tokens: ~{est_tokens}"
        )

    # KV head replication
    if mc and mc.num_key_value_heads > 0 and mc.num_key_value_heads < tp_size:
        suggestions.append(
            "3. KV head replication detected (kv_heads < tp_size, KV replicated across all TP ranks)\n"
            "   Consider EP (Expert Parallelism) to reduce TP size and KV replication overhead"
        )

    if suggestions:
        for s in suggestions:
            lines.append("")
            lines.append(s)
    else:
        lines.append("")
        lines.append("  Current configuration appears well-tuned.")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def generate_report(
    parsed: ParsedLog,
    bd: MemoryBreakdown,
    smi_entries: List[NvidiaSmiEntry],
    mc: Optional[ModelConfig],
    concurrency: List[ConcurrencyEstimate],
    gpu_key: str,
    gpu_hbm_gib: float,
) -> str:
    """Generate the full text report."""
    sections = []

    # Header
    sa = parsed.server_args
    header = []
    header.append("=" * 80)
    header.append("LLM Serving Capacity Analysis Report")
    header.append("=" * 80)
    header.append("")
    header.append(f"  Framework:          {parsed.framework}")
    header.append(f"  Model:              {sa.model_path if sa else 'N/A'}")
    header.append(f"  GPU:                {gpu_key} ({gpu_hbm_gib:.0f} GiB HBM)")
    header.append(f"  TP / PP / EP:       {sa.tp_size if sa else '?'}/{sa.pp_size if sa else '?'}/{sa.ep_size if sa else '?'}")
    header.append(f"  mem-fraction-static: {sa.mem_fraction_static if sa else '?'}")
    header.append(f"  kv-cache-dtype:     {sa.kv_cache_dtype if sa else '?'}")
    if sa and sa.disable_radix_cache:
        header.append(f"  radix-cache:        disabled")
    header.append("")

    sections.append("\n".join(header))

    # Memory breakdown
    sections.append(format_breakdown_table(bd))

    # Per-rank comparison
    if smi_entries:
        sections.append(format_smi_table(smi_entries))

    # KV pool detail
    if sa:
        sections.append(format_kv_pool_detail(
            parsed, bd, mc,
            kv_dtype=sa.kv_cache_dtype,
            tp_size=sa.tp_size,
        ))

    # Concurrency
    if concurrency:
        sections.append(format_concurrency_table(concurrency, kv_dtype=sa.kv_cache_dtype if sa else "unknown"))

    # Tuning suggestions
    sections.append(format_tuning_suggestions(bd, parsed, sa.kv_cache_dtype if sa else "unknown", gpu_hbm_gib, mc, sa.tp_size if sa else 1))

    return "\n".join(sections)


def generate_json_report(
    parsed: ParsedLog,
    bd: MemoryBreakdown,
    smi_entries: List[NvidiaSmiEntry],
    mc: Optional[ModelConfig],
    concurrency: List[ConcurrencyEstimate],
    gpu_key: str,
    gpu_hbm_gib: float,
) -> str:
    """Generate JSON report."""
    report = {
        "framework": parsed.framework,
        "model": parsed.server_args.model_path if parsed.server_args else None,
        "gpu": gpu_key,
        "gpu_hbm_gib": gpu_hbm_gib,
        "tp_size": parsed.server_args.tp_size if parsed.server_args else None,
        "pp_size": parsed.server_args.pp_size if parsed.server_args else None,
        "ep_size": parsed.server_args.ep_size if parsed.server_args else None,
        "mem_fraction_static": parsed.server_args.mem_fraction_static if parsed.server_args else None,
        "kv_cache_dtype": parsed.server_args.kv_cache_dtype if parsed.server_args else None,
        "memory_breakdown": {
            "rank": bd.rank,
            "model_weights_gib": round(bd.model_weights_gib, 2),
            "kv_pool_gib": round(bd.kv_pool_gib, 2),
            "cuda_graph_gib": round(bd.cuda_graph_gib, 2),
            "framework_overhead_gib": round(bd.framework_overhead_gib, 2),
            "other_gib": round(bd.other_gib, 2),
            "total_used_gib": round(bd.total_used_gib, 2),
            "weight_pct": round(bd.weight_pct, 1),
            "kv_pool_pct": round(bd.kv_pool_pct, 1),
        },
        "nvidia_smi": [
            {"rank": e.index, "used_mib": e.memory_used_mib, "free_mib": e.memory_free_mib}
            for e in smi_entries
        ],
        "kv_pool": {},
        "concurrency": [],
    }

    if parsed.sw_kv_calcs:
        dc = parsed.sw_kv_calcs[0]
        report["kv_pool"] = {
            "bytes_per_full_token": dc.bytes_per_full_token,
            "available_bytes_gib": dc.available_bytes_gb,
            "full_token_count": dc.full_token,
        }
    elif parsed.memory_profilings:
        mp = parsed.memory_profilings[0]
        report["kv_pool"] = {
            "available_gpu_memory_gib": mp.available_gpu_memory_gb,
            "total_gpu_memory_gib": mp.total_gpu_memory_gb,
            "rest_memory_gib": mp.rest_memory_gb,
        }

    if parsed.final_info:
        report["max_total_num_tokens"] = parsed.final_info.max_total_num_tokens
        report["max_running_requests"] = parsed.final_info.max_running_requests

    for est in concurrency:
        report["concurrency"].append({
            "request_tokens": est.request_tokens,
            "max_concurrent": est.max_concurrent,
        })

    if mc:
        report["model_config"] = {
            "num_hidden_layers": mc.num_hidden_layers,
            "hidden_size": mc.hidden_size,
            "num_key_value_heads": mc.num_key_value_heads,
            "head_dim": mc.head_dim,
            "attention_type": mc.attention_type,
        }

    return json.dumps(report, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Serving Capacity Analyzer — parse serving logs, decompose GPU memory, estimate max concurrency",
    )
    parser.add_argument("--log-file", required=True, help="Path to SGLang/vLLM startup log file")
    parser.add_argument("--nvidia-smi-file", default=None, help="Path to nvidia-smi CSV output file (optional)")
    parser.add_argument("--gpu", default=None, help="GPU type (e.g. h20, h100, h200, b200). Auto-detected if omitted.")
    parser.add_argument("--config-json", default=None, help="Path to model config.json for KV cache byte estimation")
    parser.add_argument("--request-tokens", default=None, help="Comma-separated request token lengths for concurrency estimate (default: 4096,6144,8192)")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")
    parser.add_argument("--target-rank", type=int, default=0, help="Target TP rank for breakdown (default: 0)")

    args = parser.parse_args()

    # Read log file
    if not os.path.exists(args.log_file):
        print(f"Error: log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.log_file) as f:
        log_text = f.read()

    # Parse log
    parsed = parse_log(log_text)

    # Load nvidia-smi
    smi_entries = []
    if args.nvidia_smi_file and os.path.exists(args.nvidia_smi_file):
        with open(args.nvidia_smi_file) as f:
            smi_text = f.read()
        smi_entries = parse_nvidia_smi(smi_text)

    # Load GPU specs
    gpu_specs = load_gpu_specs()

    # Determine GPU type
    gpu_key = None
    if args.gpu:
        gpu_key = GPU_ALIAS.get(args.gpu.lower(), args.gpu.lower())
    else:
        gpu_key = infer_gpu_from_log(parsed, gpu_specs)

    # Get HBM
    gpu_hbm_gib = 0.0
    if gpu_key and gpu_key in gpu_specs:
        gpu_hbm_gib = gpu_specs[gpu_key].get("hbm_gb", 0)
    elif gpu_key == "l20n":
        gpu_hbm_gib = 72  # L20N has 72GB HBM
    else:
        # Try to infer from nvidia-smi or log
        if smi_entries:
            total_mib = smi_entries[0].memory_used_mib + smi_entries[0].memory_free_mib
            gpu_hbm_gib = total_mib / 1024.0
        elif parsed.checkpoints:
            # avail_before_weight + ~2GB framework = approximate HBM
            for ck in parsed.checkpoints:
                if ck.label == "load_weight_begin":
                    gpu_hbm_gib = ck.avail_gb + 2.0  # rough estimate
                    break

    if gpu_hbm_gib == 0:
        print("Warning: Could not determine GPU HBM size. Use --gpu flag.", file=sys.stderr)
        gpu_hbm_gib = 96  # fallback

    # Load model config
    mc = None
    if args.config_json and os.path.exists(args.config_json):
        mc = load_model_config(args.config_json)

    # Decompose memory
    bd = decompose_memory(parsed, gpu_hbm_gib, smi_entries, args.target_rank)

    # Concurrency estimation
    concurrency = []
    if parsed.final_info:
        kv_pool_gib = bd.kv_pool_gib
        bytes_per_full_token = 0.0
        if parsed.sw_kv_calcs:
            bytes_per_full_token = parsed.sw_kv_calcs[0].bytes_per_full_token

        kv_dtype = parsed.server_args.kv_cache_dtype if parsed.server_args else "bf16"

        request_tokens_list = [4096, 6144, 8192]
        if args.request_tokens:
            request_tokens_list = [int(x.strip()) for x in args.request_tokens.split(",")]

        concurrency = estimate_concurrency(
            parsed.final_info, kv_pool_gib, bytes_per_full_token, kv_dtype, request_tokens_list
        )

    # Output
    if args.format == "json":
        print(generate_json_report(parsed, bd, smi_entries, mc, concurrency, gpu_key or "unknown", gpu_hbm_gib))
    else:
        print(generate_report(parsed, bd, smi_entries, mc, concurrency, gpu_key or "unknown", gpu_hbm_gib))


if __name__ == "__main__":
    main()
