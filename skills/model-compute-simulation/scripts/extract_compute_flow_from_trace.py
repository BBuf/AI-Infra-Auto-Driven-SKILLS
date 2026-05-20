#!/usr/bin/env python3
"""Extract compute flow from torch profiler trace (record_shapes=True).

This script loads a torch profiler trace file, filters cpu_op events for
compute-related operators (matmul, linear, attention, etc.), extracts tensor
dimensions, computes FLOPs, and attributes each operator to a module scope
using python_function events.

The output serves as ground-truth validation and supplement for the static
compute-flow templates in model_compute_simulator.py.

Usage:
  # Basic extraction to JSONL
  python3 extract_compute_flow_from_trace.py \\
    --input /path/to/trace.json.gz

  # Human-readable table
  python3 extract_compute_flow_from_trace.py \\
    --input /path/to/trace.json \\
    --format text

  # Summary only (category-level FLOPs aggregation)
  python3 extract_compute_flow_from_trace.py \\
    --input /path/to/trace.json.gz \\
    --format summary

  # Compare with static template from model_compute_simulator
  python3 extract_compute_flow_from_trace.py \\
    --input /path/to/trace.json.gz \\
    --compare qwen3-235b-a22b \
    --batch-size 1 --seq-len 1 --tp 8 --ep 8

  # Filter small ops and limit layers
  python3 extract_compute_flow_from_trace.py \\
    --input /path/to/trace.json.gz \\
    --min-flops 1000 --layer-limit 4
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — import from sibling model_compute_simulator.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    from model_compute_simulator import (
        CONFIG_INDEX,
        build_layer_ops,
        fmt_flops,
        load_json,
        matmul_flops,
        resolve_model,
    )

    _HAS_SIMULATOR = True
except ImportError:
    _HAS_SIMULATOR = False

# ---------------------------------------------------------------------------
# Compute operators we care about
# ---------------------------------------------------------------------------
COMPUTE_OPS = frozenset(
    {
        "aten::mm",
        "aten::bmm",
        "aten::matmul",
        "aten::linear",
        "aten::addmm",
        "aten::baddbmm",
        "aten::scaled_dot_product_attention",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_math_attention",
        "aten::mv",
        # Elementwise / reduce that contribute to FLOPs accounting
        "aten::silu",
        "aten::mul",
        "aten::add",
        "aten::rms_norm",
        "aten::layer_norm",
        "aten::softmax",
        "aten::_softmax",
        "aten::sigmoid",
        "aten::gelu",
        "aten::relu",
        "aten::topk",
        "aten::scatter",
        "aten::index_add",
        "aten::embedding",
    }
)

# Subset that are matmul-like and have 2*M*N*K FLOPs
MATMUL_OPS = frozenset(
    {
        "aten::mm",
        "aten::bmm",
        "aten::matmul",
        "aten::linear",
        "aten::addmm",
        "aten::baddbmm",
    }
)

# ---------------------------------------------------------------------------
# Scope → Category mapping
# ---------------------------------------------------------------------------
ATTENTION_KEYWORDS = (
    "self_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "kv_compress",
    "attention",
    "q_lora",
    "o_lora",
    "qkv_proj",
    "kv_proj",
    "attn_score",
    "attn_v",
    "csa_attn",
    "hca_attn",
    "kv_decompress",
)

MOE_FFN_KEYWORDS = (
    "mlp",
    "gate_proj",
    "up_proj",
    "down_proj",
    "experts",
    "shared_expert",
    "routed_expert",
    "ffn",
)

NORM_KEYWORDS = (
    "norm",
    "rmsnorm",
    "layernorm",
    "input_layernorm",
    "post_attention_layernorm",
)

EMBED_KEYWORDS = (
    "embed",
    "lm_head",
    "embed_tokens",
)

MOE_ROUTER_KEYWORDS = (
    "router",
    "gate",
)


def scope_to_category(scope: str, is_moe_model: bool = False) -> str:
    """Map a Python scope name to a compute category.

    Args:
        scope: The python_function scope string.
        is_moe_model: Whether the model is MoE (affects moe vs ffn distinction).

    Returns:
        One of: "attention", "moe", "ffn", "norm", "embed", "mhc", "other".
    """
    scope_lower = scope.lower()

    # Check embed first (typically has no layer_idx, early exit)
    for kw in EMBED_KEYWORDS:
        if kw in scope_lower:
            return "embed"

    # Check norm
    for kw in NORM_KEYWORDS:
        if kw in scope_lower:
            return "norm"

    # Check attention
    for kw in ATTENTION_KEYWORDS:
        if kw in scope_lower:
            return "attention"

    # Check MoE router specifically (must be before MOE_FFN check because
    # "gate" can match both "gate" (router) and "gate_proj" (FFN projection))
    for kw in MOE_ROUTER_KEYWORDS:
        if kw in scope_lower:
            # Disambiguate "gate" from "gate_proj"
            if kw == "gate" and "gate_proj" in scope_lower:
                pass  # falls through to MOE_FFN check below
            else:
                return "moe"

    # Check MoE/FFN
    for kw in MOE_FFN_KEYWORDS:
        if kw in scope_lower:
            if is_moe_model or "expert" in scope_lower:
                return "moe"
            return "ffn"

    # Check MHC
    if "mhc" in scope_lower:
        return "mhc"

    return "other"


# ---------------------------------------------------------------------------
# Layer index extraction
# ---------------------------------------------------------------------------
_LAYER_IDX_RE = re.compile(r"layers\.(\d+)")


def extract_layer_idx(scope: str) -> int:
    """Extract transformer layer index from scope string.

    Examples:
        "model.layers.0.self_attn.q_proj" → 0
        "model.layers.5.mlp.gate_proj" → 5
        "model.embed_tokens" → -1

    Returns:
        Layer index, or -1 if not found (e.g. embed/lm_head).
    """
    m = _LAYER_IDX_RE.search(scope)
    if m:
        return int(m.group(1))
    return -1


# ---------------------------------------------------------------------------
# FLOPs calculation
# ---------------------------------------------------------------------------
def compute_op_flops(
    op_name: str,
    input_dims: List[List[int]],
    output_dims: Optional[List[List[int]]] = None,
) -> int:
    """Compute FLOPs for a given operator and its tensor dimensions.

    Args:
        op_name: The aten operator name.
        input_dims: List of input dimension lists from trace.
        output_dims: Optional output dimension lists from trace.

    Returns:
        FLOPs count, or 0 if cannot be determined.
    """
    if op_name in MATMUL_OPS:
        return _matmul_flops_from_dims(op_name, input_dims)
    # Non-matmul ops: not computing FLOPs for now
    return 0


def _matmul_flops_from_dims(op_name: str, input_dims: List[List[int]]) -> int:
    """Calculate FLOPs for matmul-like operators from input dimensions.

    For aten::mm / aten::matmul: A[m,k] @ B[k,n] → 2*m*n*k
    For aten::bmm: A[b,m,k] @ B[b,k,n] → 2*b*m*n*k
    For aten::linear: A[m,k] @ W[n,k]^T + bias → 2*m*n*k
    For aten::addmm: bias + A[m,k] @ B[k,n] → 2*m*n*k
    """
    if len(input_dims) < 2:
        return 0

    # For addmm, the first input is the bias (1D); mat1/mat2 follow.
    # We need operator-aware extraction before dimension validation.
    if op_name == "aten::addmm":
        # addmm(bias, mat1, mat2)
        if len(input_dims) < 3:
            return 0
        mat1_dims = input_dims[1]
        mat2_dims = input_dims[2]
        if len(mat1_dims) < 2 or len(mat2_dims) < 2:
            return 0
        m = mat1_dims[0]
        # Batched addmm
        if len(mat1_dims) >= 3:
            m = 1
            for d in mat1_dims[:-2]:
                m *= d
            m *= mat1_dims[-2]
            k_a = mat1_dims[-1]
        else:
            k_a = mat1_dims[1]
        k_b = mat2_dims[0] if len(mat2_dims) >= 2 else 0
        n = mat2_dims[1] if len(mat2_dims) >= 2 else 0
        return matmul_flops(m, n, k_a)

    a_dims = input_dims[0]
    b_dims = input_dims[1]

    if len(a_dims) < 2 or len(b_dims) < 2:
        return 0

    # aten::bmm has 3D inputs
    if op_name == "aten::bmm":
        if len(a_dims) >= 3 and len(b_dims) >= 3:
            b_batch = a_dims[0]
            m = a_dims[1]
            k_a = a_dims[2]
            k_b = b_dims[1]
            n = b_dims[2]
            if k_a != k_b:
                return 0
            return matmul_flops(b_batch * m, n, k_a)
        return 0

    # aten::linear has W as [out_features, in_features]
    # and input as [..., in_features]
    if op_name == "aten::linear":
        m = 1
        for d in a_dims[:-1]:
            m *= d
        k_a = a_dims[-1]
        n = b_dims[0]
        k_b = b_dims[1] if len(b_dims) >= 2 else 0
        if k_a != k_b:
            # Dimensions may not align; best effort
            return matmul_flops(m, n, k_a)
        return matmul_flops(m, n, k_a)

    # aten::mm / aten::matmul: A[m,k] @ B[k,n]
    m = a_dims[0]
    # Handle batched matmul in mm (some traces have 3D)
    if len(a_dims) >= 3:
        m = 1
        for d in a_dims[:-2]:
            m *= d
        m *= a_dims[-2]
        k_a = a_dims[-1]
    else:
        k_a = a_dims[1]

    if len(b_dims) >= 3:
        k_b = b_dims[-2]
        n = b_dims[-1]
    else:
        k_b = b_dims[0]
        n = b_dims[1]

    if k_a != k_b:
        return 0
    return matmul_flops(m, n, k_a)


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def load_trace(path: str) -> List[dict]:
    """Load a torch profiler trace file (.json or .json.gz).

    Args:
        path: Path to the trace file.

    Returns:
        List of trace events.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Trace can be either a dict with "traceEvents" key or a plain list
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected trace format in {path}")


# ---------------------------------------------------------------------------
# Scope resolution: map cpu_op → enclosing python_function scope
# ---------------------------------------------------------------------------
def build_scope_map(events: List[dict]) -> Dict[Tuple[int, int], List[dict]]:
    """Build a mapping from (pid, tid) to python_function events sorted by ts.

    Args:
        events: All trace events.

    Returns:
        Dict mapping (pid, tid) → list of python_function events sorted by ts.
    """
    scope_map: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for ev in events:
        if ev.get("cat") == "python_function":
            key = (ev.get("pid", 0), ev.get("tid", 0))
            scope_map[key].append(ev)
    # Sort by timestamp
    for key in scope_map:
        scope_map[key].sort(key=lambda e: e.get("ts", 0))
    return scope_map


def resolve_scope(ev: dict, scope_map: Dict[Tuple[int, int], List[dict]]) -> str:
    """Find the innermost python_function scope that contains the given event.

    An event is "contained" by a python_function if it starts within the
    function's [ts, ts+dur) interval on the same (pid, tid).

    Args:
        ev: The cpu_op event.
        scope_map: Pre-built scope map from build_scope_map().

    Returns:
        The innermost scope name, or "" if no scope found.
    """
    pid = ev.get("pid", 0)
    tid = ev.get("tid", 0)
    key = (pid, tid)
    ev_ts = ev.get("ts", 0)

    candidates = scope_map.get(key, [])
    best_scope = ""
    best_dur = float("inf")

    for scope_ev in candidates:
        scope_ts = scope_ev.get("ts", 0)
        scope_dur = scope_ev.get("dur", 0)
        if scope_dur <= 0:
            continue
        # Event must start within the scope interval
        if scope_ts <= ev_ts < scope_ts + scope_dur:
            # Prefer the innermost (shortest duration) scope
            if scope_dur < best_dur:
                best_dur = scope_dur
                best_scope = scope_ev.get("name", "")

    return best_scope


def parse_nn_module_scope(scope_name: str) -> str:
    """Parse a python_function name into a cleaner scope path.

    Handles two formats:
    1. "nn.Module: Qwen2MoeForCausalLM_model_layers_0_self_attn_q_proj"
       → "Qwen2MoeForCausalLM.model.layers.0.self_attn.q_proj"
    2. "python/sglang/srt/models/deepseek_v2.py(245): forward"
       → "deepseek_v2.forward"

    Args:
        scope_name: Raw python_function name.

    Returns:
        Cleaned scope path.
    """
    if not scope_name:
        return ""

    # Format 1: nn.Module prefix
    if scope_name.startswith("nn.Module:"):
        module_path = scope_name[len("nn.Module:") :].strip()

        # Strategy: split the module path into the model-name prefix and the
        # hierarchical suffix.  The model name (e.g. "Qwen2MoeForCausalLM")
        # uses underscores that should NOT be converted to dots, while the
        # hierarchical part after "model" uses underscores as dot separators.

        # Step 1: Find the "model" boundary.
        # Common patterns:
        #   <ModelName>_model_layers_N_<submodule>
        #   <ModelName>_model_embed_tokens
        #   <ModelName>_model_lm_head
        model_match = re.match(r"^(.+?)(_model)(.*)", module_path)
        if model_match:
            model_name = model_match.group(1)
            rest = model_match.group(3)  # starts with _
            # Strategy: protect compound names by replacing their internal
            # underscores with a sentinel, then replace path-separator
            # underscores with dots, then restore the sentinels.
            sentinel = "\x00"
            compound_names = [
                "self_attn",
                "post_attention_layernorm",
                "input_layernorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "kv_compress",
                "shared_expert",
                "routed_expert",
                "embed_tokens",
                "lm_head",
                "kv_proj",
                "qkv_proj",
                "attn_score",
                "attn_v",
                "csa_attn",
                "hca_attn",
                "kv_decompress",
                "q_lora_down",
                "q_lora_up",
                "o_lora_down",
                "o_lora_up",
            ]
            # Step 1: protect compound names (longer first to avoid partial matches)
            for name in sorted(compound_names, key=len, reverse=True):
                rest = rest.replace(name, name.replace("_", sentinel))
            # Step 2: replace layer index
            rest = re.sub(r"_layers_(\d+)", r".layers.\1", rest)
            # Step 3: replace remaining underscores (path separators) with dots
            rest = rest.replace("_", ".")
            # Step 4: restore compound names
            rest = rest.replace(sentinel, "_")
            return f"{model_name}.model{rest}"

        # Fallback: replace known hierarchical separators
        result = module_path
        result = re.sub(r"_layers_(\d+)", r".layers.\1", result)
        for pattern, replacement in [
            (r"_self_attn_", ".self_attn."),
            (r"_mlp_", ".mlp."),
            (r"_embed_tokens", ".embed_tokens"),
            (r"_lm_head", ".lm_head"),
            (r"_model$", ".model"),
            (r"_q_proj$", ".q_proj"),
            (r"_k_proj$", ".k_proj"),
            (r"_v_proj$", ".v_proj"),
            (r"_o_proj$", ".o_proj"),
            (r"_gate_proj$", ".gate_proj"),
            (r"_up_proj$", ".up_proj"),
            (r"_down_proj$", ".down_proj"),
            (r"_shared_expert$", ".shared_expert"),
            (r"_experts$", ".experts"),
            (r"_router$", ".router"),
            (r"_gate$", ".gate"),
            (r"_norm$", ".norm"),
            (r"_input_layernorm$", ".input_layernorm"),
            (r"_post_attention_layernorm$", ".post_attention_layernorm"),
        ]:
            result = re.sub(pattern, replacement, result)
        return result

    # Format 2: python/file.py(line): function
    m = re.match(r"python/.*?([^/]+)\.py\(\d+\):\s*(.*)", scope_name)
    if m:
        filename = m.group(1)
        func = m.group(2).strip()
        return f"{filename}.{func}"

    return scope_name


# ---------------------------------------------------------------------------
# CUDA Graph detection
# ---------------------------------------------------------------------------
def detect_cuda_graph_events(events: List[dict]) -> List[dict]:
    """Identify CUDA graph capture/replay events.

    Args:
        events: All trace events.

    Returns:
        List of CUDA graph-related events.
    """
    cuda_graph_events = []
    for ev in events:
        name = ev.get("name", "").lower()
        cat = ev.get("cat", "")
        if "cuda_graph" in name or "cudagraph" in name:
            cuda_graph_events.append(ev)
        elif cat == "cuda_graph" or cat == "CUDAGraph":
            cuda_graph_events.append(ev)
    return cuda_graph_events


# ---------------------------------------------------------------------------
# Stage detection from annotations
# ---------------------------------------------------------------------------
def detect_stage(ev: dict) -> str:
    """Detect the serving stage (extend/decode) from event annotations.

    Looks for stage info in the event args or in enclosing scope names.

    Args:
        ev: A trace event.

    Returns:
        "extend", "decode", or "" if unknown.
    """
    args = ev.get("args", {})
    # Check args for stage annotation
    stage = args.get("stage", "")
    if stage:
        return stage.lower()

    # Check scope name for stage hints
    scope = args.get("scope", "")
    if "extend" in scope.lower() or "prefill" in scope.lower():
        return "extend"
    if "decode" in scope.lower():
        return "decode"

    return ""


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_compute_flow(
    events: List[dict],
    min_flops: int = 0,
    layer_limit: int = 0,
    tp: int = 1,
) -> List[dict]:
    """Extract compute operators from trace events.

    Args:
        events: All trace events from the profiler.
        min_flops: Minimum FLOPs threshold to include an operator.
        layer_limit: If > 0, only include ops from the first N layers.
        tp: Tensor parallelism degree (for FLOPs scaling in compare mode).

    Returns:
        List of operator dicts with keys:
            op, scope, input_dims, output_dims, flops, category,
            layer_idx, ts_us, stage, cuda_graph_capture
    """
    # Build scope map for python_function events
    scope_map = build_scope_map(events)

    # Detect CUDA graph events
    cuda_graph_events = detect_cuda_graph_events(events)
    has_cuda_graph = len(cuda_graph_events) > 0

    # Determine if CUDA graph capture phase
    capture_ranges = []
    for cg_ev in cuda_graph_events:
        name = cg_ev.get("name", "").lower()
        if "capture" in name or "record" in name:
            cg_ts = cg_ev.get("ts", 0)
            cg_dur = cg_ev.get("dur", 0)
            capture_ranges.append((cg_ts, cg_ts + cg_dur))

    # Filter and process cpu_op events
    results = []
    for ev in events:
        cat = ev.get("cat", "")
        name = ev.get("name", "")

        if cat != "cpu_op":
            continue
        if name not in COMPUTE_OPS:
            continue

        # Extract input dimensions
        args = ev.get("args", {})
        input_dims = args.get("Input Dims", [])
        input_types = args.get("Input type", [])
        output_dims = args.get("Output Dims", [])

        # Resolve scope
        raw_scope = resolve_scope(ev, scope_map)
        scope = parse_nn_module_scope(raw_scope)

        # Determine category
        # We'll refine is_moe_model later in compare mode; default to False
        category = scope_to_category(scope, is_moe_model=False)

        # If scope is empty, try to infer category from op name
        if not scope and name == "aten::embedding":
            category = "embed"

        # Extract layer index
        layer_idx = extract_layer_idx(scope)

        # Compute FLOPs
        flops = compute_op_flops(name, input_dims, output_dims)

        # Detect stage
        stage = detect_stage(ev)

        # Check if within CUDA graph capture
        in_capture = False
        ev_ts = ev.get("ts", 0)
        for cap_start, cap_end in capture_ranges:
            if cap_start <= ev_ts < cap_end:
                in_capture = True
                break

        # Build result
        op_record = {
            "op": name,
            "scope": scope,
            "input_dims": input_dims,
            "output_dims": output_dims,
            "flops": flops,
            "category": category,
            "layer_idx": layer_idx,
            "ts_us": ev_ts,
            "dur_us": ev.get("dur", 0),
            "stage": stage,
            "cuda_graph_capture": in_capture if has_cuda_graph else False,
        }

        results.append(op_record)

    # Sort by timestamp
    results.sort(key=lambda r: r["ts_us"])

    # Apply min_flops filter
    if min_flops > 0:
        results = [r for r in results if r["flops"] >= min_flops or r["flops"] == 0]

    # Apply layer_limit
    if layer_limit > 0:
        valid_layers = set(range(layer_limit)) | {-1}
        results = [r for r in results if r["layer_idx"] in valid_layers]

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_jsonl(ops: List[dict]) -> str:
    """Format extracted operators as JSONL (one JSON object per line)."""
    lines = []
    for op in ops:
        lines.append(json.dumps(op, ensure_ascii=False))
    return "\n".join(lines)


def format_text(ops: List[dict]) -> str:
    """Format extracted operators as a human-readable table."""
    if not ops:
        return "(no compute operators found)"

    lines = []
    # Header
    lines.append(
        f"{'Op':36s} {'Scope':50s} {'Cat':>8s} {'Layer':>5s} "
        f"{'FLOPs':>16s} {'Dur(μs)':>9s} {'Input Dims':30s} {'Stage':8s} {'CG':3s}"
    )
    lines.append("-" * 170)

    for op in ops:
        scope_short = op["scope"]
        if len(scope_short) > 48:
            scope_short = "..." + scope_short[-45:]
        input_dims_str = str(op["input_dims"]) if op["input_dims"] else "—"
        if len(input_dims_str) > 28:
            input_dims_str = input_dims_str[:25] + "..."
        flops_str = fmt_flops(op["flops"]) if op["flops"] > 0 else "—"
        dur_str = f"{op['dur_us']:.1f}" if op.get("dur_us", 0) > 0 else "—"
        cg_mark = "✓" if op.get("cuda_graph_capture") else ""
        stage = op.get("stage", "")

        lines.append(
            f"{op['op']:36s} {scope_short:50s} {op['category']:>8s} "
            f"{op['layer_idx']:>5d} {flops_str:>16s} {dur_str:>9s} {input_dims_str:30s} "
            f"{stage:8s} {cg_mark:3s}"
        )

    # Summary
    lines.append("")
    lines.append(_format_summary(ops))

    return "\n".join(lines)


def format_summary(ops: List[dict]) -> str:
    """Format a summary of category-level FLOPs aggregation."""
    return _format_summary(ops)


def _format_summary(ops: List[dict]) -> str:
    """Internal: build summary text from ops list."""
    lines = []
    lines.append("=" * 70)
    lines.append("  COMPUTE FLOW SUMMARY")
    lines.append("=" * 70)

    # Total ops
    lines.append(f"  Total compute ops: {len(ops)}")

    # Category breakdown
    cat_flops: Dict[str, int] = defaultdict(int)
    cat_count: Dict[str, int] = defaultdict(int)
    layer_flops: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for op in ops:
        cat_flops[op["category"]] += op["flops"]
        cat_count[op["category"]] += 1
        layer_flops[op["layer_idx"]][op["category"]] += op["flops"]

    total_flops = sum(cat_flops.values())

    lines.append(f"  Total FLOPs: {fmt_flops(total_flops)}")
    lines.append("")
    lines.append(f"  {'Category':12s} {'Count':>8s} {'FLOPs':>16s} {'Pct':>8s}")
    lines.append(f"  {'-' * 48}")
    for cat in sorted(cat_flops.keys(), key=lambda c: -cat_flops[c]):
        pct = cat_flops[cat] / total_flops * 100 if total_flops > 0 else 0
        lines.append(
            f"  {cat:12s} {cat_count[cat]:>8d} {fmt_flops(cat_flops[cat]):>16s} {pct:>7.1f}%"
        )

    # Per-layer breakdown (sample a few layers)
    if layer_flops:
        lines.append("")
        lines.append("  Per-layer FLOPs (sample):")
        sorted_layers = sorted(layer_flops.keys())
        sample_layers = (
            sorted_layers[:5] + sorted_layers[-2:]
            if len(sorted_layers) > 7
            else sorted_layers
        )
        # Remove duplicates while preserving order
        seen = set()
        unique_sample = []
        for l in sample_layers:
            if l not in seen:
                seen.add(l)
                unique_sample.append(l)

        for lidx in unique_sample:
            l_flops = sum(layer_flops[lidx].values())
            cat_detail = ", ".join(
                f"{cat}={fmt_flops(f)}"
                for cat, f in sorted(layer_flops[lidx].items(), key=lambda x: -x[1])
            )
            label = f"Layer {lidx}" if lidx >= 0 else "Global"
            lines.append(f"    {label:12s}: {fmt_flops(l_flops):>16s}  ({cat_detail})")

    # CUDA graph info
    cg_ops = [op for op in ops if op.get("cuda_graph_capture")]
    if cg_ops:
        lines.append("")
        lines.append(f"  CUDA Graph capture ops: {len(cg_ops)}")
        lines.append(
            f"  Note: These ops appear only during graph capture; "
            f"replay executes on GPU without CPU traces."
        )

    # Stage breakdown
    stages: Dict[str, int] = defaultdict(int)
    for op in ops:
        stage = op.get("stage", "")
        if stage:
            stages[stage] += 1
    if stages:
        lines.append("")
        lines.append("  Stage breakdown:")
        for stage, count in sorted(stages.items()):
            lines.append(f"    {stage}: {count} ops")

    # Missing Input Dims
    missing_dims = [op for op in ops if not op["input_dims"] and op["flops"] == 0]
    if missing_dims:
        lines.append("")
        lines.append(
            f"  Warning: {len(missing_dims)} ops have missing Input Dims "
            f"(FLOPs could not be computed)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compare mode: trace vs static template
# ---------------------------------------------------------------------------
def compare_with_template(
    ops: List[dict],
    model_key: str,
    batch_size: int,
    seq_len: int,
    tp: int,
    ep: int,
) -> str:
    """Compare trace-extracted operators with the static template.

    Args:
        ops: Extracted operators from trace.
        model_key: Model name key for model-config-index.json.
        batch_size: Batch size for template.
        seq_len: Sequence length for template.
        tp: Tensor parallelism.
        ep: Expert parallelism.

    Returns:
        Formatted comparison table string.
    """
    if not _HAS_SIMULATOR:
        return (
            "Error: Cannot compare — model_compute_simulator.py not importable.\n"
            "Ensure it is in the same directory as this script."
        )

    # Load config and build template
    config_index = load_json(CONFIG_INDEX)
    cfg = resolve_model(model_key, config_index)
    if cfg is None:
        return f"Error: Model '{model_key}' not found in config index."

    # Re-classify trace categories with MoE awareness
    is_moe = cfg.get("moe", False)
    for op in ops:
        op["category"] = scope_to_category(op["scope"], is_moe_model=is_moe)

    # Build template ops
    template_ops = build_layer_ops(cfg, batch_size, seq_len, tp, ep)

    # Aggregate trace FLOPs by category and layer
    trace_cat_flops: Dict[str, int] = defaultdict(int)
    trace_layer_flops: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    trace_op_names: Dict[str, set] = defaultdict(set)

    for op in ops:
        cat = op["category"]
        flops = op["flops"]
        lidx = op["layer_idx"]

        # Scale FLOPs by tp for comparison with template
        # (trace shows post-TP dimensions)
        if cat in ("attention", "norm", "embed", "mhc") and tp > 1:
            flops_scaled = flops * tp
        elif cat == "moe" and "routed" in op.get("scope", "").lower() and ep > 1:
            flops_scaled = flops * ep
        else:
            flops_scaled = flops

        trace_cat_flops[cat] += flops_scaled
        trace_layer_flops[lidx][cat] += flops_scaled
        trace_op_names[cat].add(op["op"])

    # Aggregate template FLOPs by category
    tmpl_cat_flops: Dict[str, int] = defaultdict(int)
    tmpl_op_names: Dict[str, set] = defaultdict(set)
    for top in template_ops:
        tmpl_cat_flops[top.category] += top.flops
        tmpl_op_names[top.category].add(top.name)

    # Build comparison output
    lines = []
    lines.append("=" * 80)
    lines.append(f"  COMPARISON: Trace vs Template (model={model_key})")
    lines.append(f"  Config: B={batch_size} S={seq_len} TP={tp} EP={ep}")
    lines.append("=" * 80)

    # Category-level comparison
    all_cats = sorted(set(list(trace_cat_flops.keys()) + list(tmpl_cat_flops.keys())))

    lines.append("")
    lines.append(
        f"  {'Category':12s} {'Trace FLOPs':>18s} {'Template FLOPs':>18s} {'Diff':>12s} {'Pct':>8s}"
    )
    lines.append(f"  {'-' * 72}")

    for cat in all_cats:
        trace_f = trace_cat_flops.get(cat, 0)
        tmpl_f = tmpl_cat_flops.get(cat, 0)
        diff = trace_f - tmpl_f
        if tmpl_f > 0:
            pct = (trace_f / tmpl_f - 1) * 100
        elif trace_f > 0:
            pct = float("inf")
        else:
            pct = 0.0

        trace_str = fmt_flops(trace_f) if trace_f > 0 else "—"
        tmpl_str = fmt_flops(tmpl_f) if tmpl_f > 0 else "—"
        diff_str = f"{diff:+d}" if diff != 0 else "0"
        pct_str = f"{pct:+.1f}%" if abs(pct) != float("inf") else "+∞"

        lines.append(
            f"  {cat:12s} {trace_str:>18s} {tmpl_str:>18s} {diff_str:>12s} {pct_str:>8s}"
        )

    # Missing categories
    trace_only_cats = set(trace_cat_flops.keys()) - set(tmpl_cat_flops.keys())
    tmpl_only_cats = set(tmpl_cat_flops.keys()) - set(trace_cat_flops.keys())

    if trace_only_cats:
        lines.append("")
        lines.append(
            f"  Categories in trace but NOT in template: {', '.join(sorted(trace_only_cats))}"
        )
    if tmpl_only_cats:
        lines.append("")
        lines.append(
            f"  Categories in template but NOT in trace: {', '.join(sorted(tmpl_only_cats))}"
        )

    # Per-layer comparison (first layer detail)
    lines.append("")
    lines.append("  Per-layer comparison (Layer 0):")
    if 0 in trace_layer_flops:
        l0_trace = trace_layer_flops[0]
        l0_tmpl: Dict[str, int] = defaultdict(int)
        for top in template_ops:
            l0_tmpl[top.category] += top.flops

        l0_cats = sorted(set(list(l0_trace.keys()) + list(l0_tmpl.keys())))
        lines.append(
            f"    {'Category':12s} {'Trace':>16s} {'Template':>16s} {'Diff%':>10s}"
        )
        lines.append(f"    {'-' * 56}")
        for cat in l0_cats:
            tf = l0_trace.get(cat, 0)
            ttf = l0_tmpl.get(cat, 0)
            if ttf > 0:
                diff_pct = (tf / ttf - 1) * 100
                diff_str = f"{diff_pct:+.1f}%"
            elif tf > 0:
                diff_str = "+∞"
            else:
                diff_str = "—"
            lines.append(
                f"    {cat:12s} {fmt_flops(tf):>16s} {fmt_flops(ttf):>16s} {diff_str:>10s}"
            )
    else:
        lines.append("    (No Layer 0 ops found in trace)")

    # Template ops not seen in trace
    lines.append("")
    lines.append("  Template ops vs trace ops:")
    for cat in sorted(tmpl_op_names.keys()):
        tmpl_names = tmpl_op_names[cat]
        trace_names = trace_op_names.get(cat, set())
        missing = tmpl_names - trace_names
        extra = trace_names - tmpl_names
        if missing:
            lines.append(f"    [{cat}] Template-only ops: {', '.join(sorted(missing))}")
        if extra:
            lines.append(f"    [{cat}] Trace-only ops: {', '.join(sorted(extra))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract compute flow from torch profiler trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  %(prog)s --input trace.json.gz

  # Human-readable table
  %(prog)s --input trace.json --format text

  # Compare with model template
  %(prog)s --input trace.json.gz --compare deepseek-v4-flash \\
      --batch-size 1 --seq-len 1 --tp 8 --ep 8
""",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to torch profiler trace file (.json or .json.gz)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "text", "summary"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--compare",
        default=None,
        metavar="MODEL_KEY",
        help="Compare with static template for MODEL_KEY (from model-config-index.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for template comparison (default: 1)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Sequence length for template comparison (default: 1)",
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallelism degree (default: 1)"
    )
    parser.add_argument(
        "--ep", type=int, default=1, help="Expert parallelism degree (default: 1)"
    )
    parser.add_argument(
        "--min-flops",
        type=int,
        default=0,
        help="Filter ops with FLOPs below this threshold (default: 0, no filter)",
    )
    parser.add_argument(
        "--layer-limit",
        type=int,
        default=0,
        help="Only output ops from the first N layers (default: 0, all layers)",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load trace
    print(f"Loading trace from {args.input} ...", file=sys.stderr)
    events = load_trace(args.input)
    print(f"Loaded {len(events)} events", file=sys.stderr)

    # Extract compute flow
    ops = extract_compute_flow(
        events,
        min_flops=args.min_flops,
        layer_limit=args.layer_limit,
        tp=args.tp,
    )
    print(f"Extracted {len(ops)} compute ops", file=sys.stderr)

    # Compare mode
    if args.compare:
        comparison = compare_with_template(
            ops,
            args.compare,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            tp=args.tp,
            ep=args.ep,
        )
        print(comparison)
        return

    # Output
    if args.format == "jsonl":
        print(format_jsonl(ops))
    elif args.format == "text":
        print(format_text(ops))
    elif args.format == "summary":
        print(format_summary(ops))


if __name__ == "__main__":
    main()
