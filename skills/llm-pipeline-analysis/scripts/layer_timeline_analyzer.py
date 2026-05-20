#!/usr/bin/env python3
"""Layer-level timeline analyzer for LLM torch profiler traces.

Parses a Chrome-trace-format .json.gz file, identifies per-layer boundaries
using model-specific anchor kernels, and produces:
  1. Per-layer wall-clock + sum-duration breakdown for a chosen forward pass
  2. Forward-pass level summary (cold-start vs steady-state comparison)
  3. Layer cluster statistics by compress_ratio

The anchor kernel and layer structure are determined by a ModelProfile,
auto-inferred from config.json or selected via --profile.

Usage:
  python3 layer_timeline_analyzer.py \
    --trace /path/to/TP-0.trace.json.gz \
    --config /path/to/config.json \
    --fwd-pass 5 \
    --show-all-passes
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict

from model_profiles import ModelProfile, get_profile, infer_profile, normalize_compress_ratios


# ── trace loading ──────────────────────────────────────────────────────────

def load_trace(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return events


# ── layer boundary detection ──────────────────────────────────────────────

def find_layer_boundaries(gpu_kernels, anchor_kernel: str):
    """Return sorted indices of anchor kernels in the trace."""
    return [i for i, e in enumerate(gpu_kernels) if anchor_kernel in e.get("name", "")]


def find_anchor_kernel(gpu_kernels, profile: ModelProfile) -> str:
    """Determine the anchor kernel substring to use.

    If profile.anchor_kernel is set, use it directly.
    Otherwise, scan the trace for common anchor candidates and pick the
    most frequent one.
    """
    if profile.anchor_kernel:
        return profile.anchor_kernel

    # Common anchor candidates across model families
    candidates = [
        "mhc_post_tilelang",      # DeepSeek-V4
        "flash_fwd_mla_combine",  # DeepSeek-V3
        "AllReduce",              # generic fallback (1 per layer)
    ]
    for c in candidates:
        count = sum(1 for e in gpu_kernels if c in e.get("name", ""))
        if count >= 4:  # at least a couple of layers
            return c

    raise ValueError(
        "Cannot auto-detect layer anchor kernel from trace. "
        "Please specify --profile or --anchor-kernel."
    )


def detect_num_layers(anchor_indices, gpu_kernels, blocks_per_layer: int,
                       default_num_layers: int = 1):
    """Heuristic: find the first forward pass boundary.

    Scan blocks 0..N and detect where the layer pattern repeats.
    Each layer spans ``blocks_per_layer`` anchor blocks.
    """
    # Use the first ~200 blocks to detect the period
    n = min(200, len(anchor_indices) - 1)
    if n < 4:
        return default_num_layers

    # Compute block durations
    durs = []
    for i in range(n):
        d = sum(gpu_kernels[j].get("dur", 0)
                for j in range(anchor_indices[i], anchor_indices[i + 1]))
        durs.append(d)

    # Find the first strong pattern repeat (correlation with offset)
    best_period = None
    best_score = 0
    for period in range(blocks_per_layer, min(120, n)):
        if period * 2 > n:
            break
        score = 0
        for i in range(min(period, n - period)):
            if durs[i] > 0 and durs[i + period] > 0:
                ratio = min(durs[i], durs[i + period]) / max(durs[i], durs[i + period])
                score += ratio
        if score > best_score:
            best_score = score
            best_period = period

    if best_period and best_score > best_period * 0.7:
        return best_period // blocks_per_layer
    return default_num_layers


# ── per-layer info extraction ─────────────────────────────────────────────

def classify_kernel(name, profile: ModelProfile):
    """Classify a kernel name using the profile's category rules."""
    for _label, key, rule in profile.category_rules:
        if rule(name):
            return key
    return "other"


def get_layer_info(gpu_kernels, anchor_indices, fwd_pass, layer_id, num_layers,
                    profile: ModelProfile):
    """Extract detailed per-layer kernel breakdown.

    Each transformer layer spans ``profile.blocks_per_layer`` consecutive
    anchor blocks.  For example, with blocks_per_layer=2 the two halves
    might be "attn" and "ffn".
    """
    bpl = profile.blocks_per_layer
    blocks_per_pass = num_layers * bpl
    base = fwd_pass * blocks_per_pass

    # Build block indices for this layer
    layer_blocks = [base + layer_id * bpl + i for i in range(bpl)]
    next_layer_start = base + (layer_id + 1) * bpl

    if next_layer_start >= len(anchor_indices):
        return None

    info = defaultdict(float)
    info["kernels"] = 0

    for bi in layer_blocks:
        s = anchor_indices[bi]
        e = anchor_indices[bi + 1]
        for j in range(s, e):
            name = gpu_kernels[j].get("name", "")
            dur = gpu_kernels[j].get("dur", 0)
            cat = classify_kernel(name, profile)
            info[cat] += dur
            info["total"] += dur
            info["kernels"] += 1

    # Wall-clock time
    first_k = gpu_kernels[anchor_indices[layer_blocks[0]]]
    last_k = gpu_kernels[anchor_indices[next_layer_start] - 1]
    info["wall_start_us"] = first_k.get("ts", 0)
    info["wall_end_us"] = last_k.get("ts", 0) + last_k.get("dur", 0)
    info["wall_us"] = info["wall_end_us"] - info["wall_start_us"]

    # Count AllReduce kernels
    info["ar_count"] = sum(
        1 for bi in layer_blocks
        for j in range(anchor_indices[bi], anchor_indices[bi + 1])
        if "AllReduce" in gpu_kernels[j].get("name", "")
    )

    return dict(info)


# ── model config helpers ──────────────────────────────────────────────────

def load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def get_compress_ratios(config):
    return normalize_compress_ratios(config)


def layer_type_label(layer_id, compress_ratios, num_layers, num_hash_layers):
    cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else -1

    # Hash layers are at the end
    hash_start = num_layers - num_hash_layers if num_hash_layers else num_layers
    is_hash = layer_id >= hash_start

    if layer_id == 0:
        return "FIRST", cr
    elif is_hash:
        return "HASH", cr
    elif layer_id == num_layers - 1:
        return "FINAL", cr
    elif cr == 0:
        return "FULL_ATTN", cr
    elif cr == 4:
        return "C4_LIGHT", cr
    elif cr == 128:
        return "C128_HEAVY", cr
    else:
        return f"CR{cr}", cr


def prefix_total(info, prefix):
    return sum(v for k, v in info.items() if k == prefix or k.startswith(prefix + "_"))


# ── main output routines ──────────────────────────────────────────────────

def print_all_passes_summary(gpu_kernels, anchor_indices, num_layers, compress_ratios,
                              trace_start, num_hash_layers, profile: ModelProfile):
    """Print one-line-per-forward-pass summary."""
    bpl = profile.blocks_per_layer
    blocks_per_pass = num_layers * bpl
    n_passes = (len(anchor_indices) - 1) // blocks_per_pass

    print(f"\n{'Fwd#':>5s}  {'Start(s)':>9s}  {'End(s)':>8s}  {'Duration(ms)':>12s}  "
          f"{'Avg Layer(ms)':>13s}  {'First Layer':>11s}  {'Notes'}")
    print("-" * 80)

    for p in range(min(n_passes, 30)):
        base = p * blocks_per_pass
        if base + blocks_per_pass >= len(anchor_indices):
            break

        first_k = gpu_kernels[anchor_indices[base]]
        last_k = gpu_kernels[anchor_indices[base + blocks_per_pass] - 1]

        start_s = (first_k.get("ts", 0) - trace_start) / 1e6
        end_s = (last_k.get("ts", 0) + last_k.get("dur", 0) - trace_start) / 1e6
        duration_ms = (end_s - start_s) * 1000
        avg_layer = duration_ms / num_layers

        # Sample layer 0
        info0 = get_layer_info(gpu_kernels, anchor_indices, p, 0, num_layers, profile)
        l0_ms = info0["wall_us"] / 1000 if info0 else 0

        notes = "cold-start" if p == 0 else ""
        if p > 0 and info0:
            info0_prev = get_layer_info(gpu_kernels, anchor_indices, p - 1, 0, num_layers, profile)
            if info0_prev:
                growth = info0["wall_us"] / info0_prev["wall_us"] if info0_prev["wall_us"] > 0 else 0
                if abs(growth - 1.0) < 0.05 and p > 2:
                    notes = "steady-state"
                elif growth > 1.5:
                    notes = f"growing {growth:.1f}x"

        print(f"  {p:>3d}  {start_s:>8.3f}s  {end_s:>7.3f}s  {duration_ms:>11.1f}  "
              f"{avg_layer:>12.2f}  {l0_ms:>10.2f}  {notes}")


def print_layer_detail(gpu_kernels, anchor_indices, fwd_pass, num_layers,
                        compress_ratios, trace_start, num_hash_layers,
                        profile: ModelProfile):
    """Print per-layer detail for a chosen forward pass."""
    print(f"\n{'=':>100s}")
    print(f"  Forward Pass #{fwd_pass} — Per-Layer Detail")
    print(f"{'=':>100s}")
    print(f"\n{'L':>3s} {'c_r':>4s} {'Type':<12s} {'Wall(ms)':>9s} {'SumDur(ms)':>10s} "
          f"{'MLA':>7s} {'MoE':>7s} {'GEMM':>7s} {'NCCL':>6s} {'MHC':>6s} "
          f"{'Hadam':>6s} {'AR#':>4s} {'K#':>4s}")
    print("-" * 100)

    for layer_id in range(num_layers):
        info = get_layer_info(gpu_kernels, anchor_indices, fwd_pass, layer_id, num_layers, profile)
        if info is None:
            break

        ltype, cr = layer_type_label(layer_id, compress_ratios, num_layers, num_hash_layers)
        gemm = info.get("gemm_fp8", 0) + info.get("gemm_bf16", 0) + info.get("gemm_f32", 0)
        mhc = prefix_total(info, "mhc")

        print(f"  {layer_id:>2d}  {cr:>3d} {ltype:<12s} {info['wall_us']/1000:>8.2f} "
              f"{info['total']/1000:>9.2f} {info.get('mla',0)/1000:>6.1f} "
              f"{info.get('moe',0)/1000:>6.1f} {gemm/1000:>6.1f} "
              f"{info.get('allreduce',0)/1000:>5.1f} {mhc/1000:>5.1f} "
              f"{info.get('hadamard',0)/1000:>5.1f} {info.get('ar_count',0):>3d} "
              f"{info.get('kernels',0):>3d}")


def print_cluster_stats(gpu_kernels, anchor_indices, fwd_pass, num_layers,
                         compress_ratios, num_hash_layers, profile: ModelProfile):
    """Print aggregate statistics by layer cluster."""
    clusters = defaultdict(list)
    for layer_id in range(num_layers):
        info = get_layer_info(gpu_kernels, anchor_indices, fwd_pass, layer_id, num_layers, profile)
        if info is None:
            break
        ltype, _ = layer_type_label(layer_id, compress_ratios, num_layers, num_hash_layers)
        clusters[ltype].append(info)

    print(f"\n{'Cluster':<16s} {'#':>3s} {'Avg Wall(ms)':>12s} {'Avg Sum(ms)':>11s} "
          f"{'MLA%':>6s} {'MoE%':>6s} {'GEMM%':>6s} {'NCCL%':>6s} {'MHC%':>6s} {'Hadam%':>7s}")
    print("-" * 85)

    for name in ["FIRST", "FULL_ATTN", "C4_LIGHT", "C128_HEAVY", "HASH", "FINAL"]:
        if name not in clusters:
            continue
        infos = clusters[name]
        n = len(infos)
        total_wall = sum(i["wall_us"] for i in infos)
        total_sum = sum(i["total"] for i in infos)
        avg_wall = total_wall / n / 1000
        avg_sum = total_sum / n / 1000

        def pct(key):
            return sum(i.get(key, 0) for i in infos) / total_sum * 100 if total_sum else 0

        def pct_prefix(prefix):
            return sum(prefix_total(i, prefix) for i in infos) / total_sum * 100 if total_sum else 0

        gemm_sum = sum(i.get("gemm_fp8", 0) + i.get("gemm_bf16", 0) + i.get("gemm_f32", 0) for i in infos)
        gemm_pct = gemm_sum / total_sum * 100 if total_sum else 0

        print(f"  {name:<14s} {n:>3d} {avg_wall:>11.2f} {avg_sum:>10.2f} "
              f"{pct('mla'):>5.1f}% {pct('moe'):>5.1f}% {gemm_pct:>5.1f}% "
              f"{pct('allreduce'):>5.1f}% {pct_prefix('mhc'):>5.1f}% {pct('hadamard'):>6.1f}%")


# ── entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LLM layer-level timeline analyzer")
    ap.add_argument("--trace", required=True, help="Path to .trace.json(.gz)")
    ap.add_argument("--config", default=None, help="Path to model config.json for compress_ratios")
    ap.add_argument("--profile", default=None,
                     help="Model profile name (dsv4_csa_hca, dsv3_mla, generic). "
                          "Auto-inferred from config if not specified.")
    ap.add_argument("--anchor-kernel", default=None,
                     help="Override anchor kernel substring for layer boundary detection")
    ap.add_argument("--fwd-pass", type=int, default=None,
                     help="Forward pass index for detailed layer breakdown (0-based)")
    ap.add_argument("--show-all-passes", action="store_true",
                     help="Print one-line-per-forward-pass summary table")
    ap.add_argument("--num-layers", type=int, default=None,
                     help="Override number of layers (default: auto-detect)")
    args = ap.parse_args()

    events = load_trace(args.trace)
    gpu = sorted([e for e in events if e.get("cat") == "kernel"], key=lambda e: e.get("ts", 0))

    config = load_config(args.config)

    # Resolve profile
    if args.profile:
        profile = get_profile(args.profile)
    else:
        profile = infer_profile(config)

    # Determine anchor kernel
    anchor_kernel = args.anchor_kernel or find_anchor_kernel(gpu, profile)
    anchor_indices = find_layer_boundaries(gpu, anchor_kernel)

    compress_ratios = get_compress_ratios(config)
    num_hash_layers = config.get("num_hash_layers", 0)
    trace_start = min(e.get("ts", float("inf")) for e in events if e.get("ts") is not None)

    if args.num_layers:
        num_layers = args.num_layers
    else:
        num_layers = config.get("num_hidden_layers", None) or detect_num_layers(
            anchor_indices, gpu, profile.blocks_per_layer, profile.default_num_layers)

    bpl = profile.blocks_per_layer
    blocks_per_pass = num_layers * bpl
    n_passes = (len(anchor_indices) - 1) // blocks_per_pass

    print(f"Trace: {args.trace}")
    print(f"Profile: {profile.name}, anchor: {anchor_kernel}, blocks/layer: {bpl}")
    print(f"GPU kernels: {len(gpu)}, anchor blocks: {len(anchor_indices)}")
    print(f"Detected: {num_layers} layers, {blocks_per_pass} blocks/pass, {n_passes} forward passes")
    print(f"compress_ratios (first 10): {compress_ratios[:10]}...")
    print(f"num_hash_layers: {num_hash_layers}")

    if args.show_all_passes:
        print_all_passes_summary(gpu, anchor_indices, num_layers, compress_ratios,
                                  trace_start, num_hash_layers, profile)

    if args.fwd_pass is not None:
        if args.fwd_pass >= n_passes:
            print(f"ERROR: fwd_pass {args.fwd_pass} exceeds available {n_passes} passes", file=sys.stderr)
            sys.exit(1)
        print_layer_detail(gpu, anchor_indices, args.fwd_pass, num_layers,
                            compress_ratios, trace_start, num_hash_layers, profile)
        print_cluster_stats(gpu, anchor_indices, args.fwd_pass, num_layers,
                             compress_ratios, num_hash_layers, profile)
    elif not args.show_all_passes:
        # Default: show summary + first steady-state pass
        print_all_passes_summary(gpu, anchor_indices, num_layers, compress_ratios,
                                  trace_start, num_hash_layers, profile)
        # Find first steady-state pass (heuristic: first pass where layer 0 > 8ms)
        for p in range(1, n_passes):
            info = get_layer_info(gpu, anchor_indices, p, 0, num_layers, profile)
            if info and info["wall_us"] > 8000:
                print(f"\nAuto-selected fwd_pass #{p} as first steady-state pass")
                print_layer_detail(gpu, anchor_indices, p, num_layers,
                                    compress_ratios, trace_start, num_hash_layers, profile)
                print_cluster_stats(gpu, anchor_indices, p, num_layers,
                                     compress_ratios, num_hash_layers, profile)
                break


if __name__ == "__main__":
    main()
