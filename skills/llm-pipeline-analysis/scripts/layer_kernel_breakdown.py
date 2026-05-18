#!/usr/bin/env python3
"""Per-layer kernel breakdown for LLM torch profiler traces.

For a specific (forward_pass, layer_id) pair, dumps every GPU kernel in
execution order with timing, category, and cumulative stats.

Usage:
  python3 layer_kernel_breakdown.py \
    --trace /path/to/TP-0.trace.json.gz \
    --config /path/to/config.json \
    --fwd-pass 5 --layer 3

  # Compute flow format (bridge to model-compute-simulation)
  python3 layer_kernel_breakdown.py \
    --trace /path/to/TP-0.trace.json.gz \
    --config /path/to/config.json \
    --fwd-pass 5 --layer 3 --format compute-flow

  # JSON export for model-compute-simulation
  python3 layer_kernel_breakdown.py \
    --trace /path/to/TP-0.trace.json.gz \
    --config /path/to/config.json \
    --fwd-pass 5 --layer 3 --format json

  # Compare two layers side-by-side
  python3 layer_kernel_breakdown.py \
    --trace /path/to/TP-0.trace.json.gz \
    --config /path/to/config.json \
    --fwd-pass 5 --layer 2 --compare-layer 3
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict

from model_profiles import ModelProfile, get_profile, infer_profile


# ── trace loading ──────────────────────────────────────────────────────────

def load_trace(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return events


# ── kernel classification ──────────────────────────────────────────────────

# CATEGORY_RULES and SIMPLIFY_RULES are now provided by ModelProfile.
# The functions below accept a profile argument.


def classify_kernel(name, profile: ModelProfile):
    """Return (display_label, machine_key) tuple using profile's rules."""
    for label, key, rule in profile.category_rules:
        if rule(name):
            return label, key
    return "  Other", "other"


def classify_kernel_label(name, profile: ModelProfile):
    """Return display label only (backward compat for text output)."""
    return classify_kernel(name, profile)[0]


def classify_kernel_key(name, profile: ModelProfile):
    """Return machine-readable key only (for JSON output)."""
    return classify_kernel(name, profile)[1]


def simplify_name(name, profile: ModelProfile):
    for pattern, repl in profile.simplify_rules:
        name = name.replace(pattern, repl)
    # Truncate template params beyond 60 chars
    if len(name) > 80:
        name = name[:77] + "..."
    return name


# ── layer kernel extraction ────────────────────────────────────────────────

def get_layer_kernels(gpu_kernels, anchor_indices, fwd_pass, layer_id, num_layers,
                      profile: ModelProfile):
    """Return ordered list of kernel dicts for a specific layer."""
    bpl = profile.blocks_per_layer
    blocks_per_pass = num_layers * bpl
    base = fwd_pass * blocks_per_pass

    layer_blocks = [base + layer_id * bpl + i for i in range(bpl)]
    next_start = base + (layer_id + 1) * bpl

    if next_start >= len(anchor_indices):
        return []

    result = []
    for bi, half_label in zip(layer_blocks, profile.half_labels):
        s = anchor_indices[bi]
        e = anchor_indices[bi + 1]
        for j in range(s, e + 1):
            k = gpu_kernels[j]
            args = k.get("args", {})
            result.append({
                "idx": j,
                "half": half_label,
                "name": k.get("name", ""),
                "ts": k.get("ts", 0),
                "dur": k.get("dur", 0),
                "input_dims": args.get("Input Dims", None),
                "output_dims": args.get("Output Dims", None),
            })
    return result


# ── config helpers ─────────────────────────────────────────────────────────

def load_config(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


def get_compress_ratios(config):
    cr = config.get("compress_ratios", [])
    return cr[:config.get("num_hidden_layers", 43)] if cr else []


# ── output ─────────────────────────────────────────────────────────────────

def print_layer_breakdown(kernels, layer_id, compress_ratios, num_hash_layers,
                           num_layers, profile: ModelProfile, label=None):
    total = sum(k["dur"] for k in kernels)
    cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else -1

    header = label or f"Layer {layer_id} (c_ratio={cr})"
    print(f"\n{'='*95}")
    print(f"  {header}  —  {total:.0f}us ({total/1000:.2f}ms), {len(kernels)} kernels")
    print(f"{'='*95}")

    # Grouped summary
    groups = defaultdict(lambda: {"dur": 0, "count": 0})
    for k in kernels:
        cat = classify_kernel_label(k["name"], profile)
        groups[cat]["dur"] += k["dur"]
        groups[cat]["count"] += 1

    print(f"\n  {'Category':<30s} {'dur(us)':>9s} {'%':>6s} {'count':>6s}")
    print(f"  {'-'*55}")
    for cat, info in sorted(groups.items(), key=lambda x: x[1]["dur"], reverse=True):
        if info["dur"] < 0.5:
            continue
        pct = info["dur"] / total * 100
        bar = "▓" * int(pct / 2)
        print(f"  {cat:<30s} {info['dur']:>8.1f} {pct:>5.1f}% {info['count']:>5d}  {bar}")

    # Detailed kernel list
    print(f"\n  {'#':>4s} {'Half':>4s} {'Simplified Name':<70s} {'dur(us)':>8s} {'%':>5s}")
    print(f"  {'-'*95}")
    for i, k in enumerate(kernels):
        name = simplify_name(k["name"], profile)
        pct = k["dur"] / total * 100
        print(f"  {i:>4d} {k['half']:>4s} {name:<70s} {k['dur']:>7.1f} {pct:>4.1f}%")

    # Unique kernel diff (compared to baseline)
    return set(simplify_name(k["name"], profile) for k in kernels)


def print_diff(kernels_a, kernels_b, label_a, label_b, profile: ModelProfile):
    names_a = set(simplify_name(k["name"], profile) for k in kernels_a)
    names_b = set(simplify_name(k["name"], profile) for k in kernels_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    print(f"\n  Kernel diff: {label_a} vs {label_b}")
    print(f"  Common: {len(names_a & names_b)} types")
    print(f"  Only in {label_a}: {len(only_a)} types")
    for n in only_a:
        print(f"    + {n}")
    print(f"  Only in {label_b}: {len(only_b)} types")
    for n in only_b:
        print(f"    + {n}")


# ── model architecture summary ───────────────────────────────────────────

def _layer_type_label(cr, layer_id, num_layers, num_hash_layers):
    """Return human-readable layer type label."""
    if layer_id == 0:
        return "FIRST"
    if layer_id == num_layers - 1:
        return "FINAL"
    if layer_id >= num_layers - num_hash_layers:
        return "HASH"
    if cr == 0:
        return "FULL_ATTN"
    if cr == 4:
        return "C4_LIGHT"
    if cr == 128:
        return "C128_HEAVY"
    return f"CR{cr}"


def print_model_architecture(config, compress_ratios, num_hash_layers, num_layers):
    """Print model architecture summary from config.json."""
    print("\n" + "=" * 80)
    print("  Model Architecture Summary")
    print("=" * 80)

    # Basic
    print(f"  Layers: {num_layers}")
    h = config.get("hidden_size", "?")
    n_heads = config.get("num_attention_heads", "?")
    n_kv = config.get("num_key_value_heads", "?")
    hd = config.get("head_dim", "?")
    print(f"  Hidden: {h}, Heads: {n_heads} (KV: {n_kv}), Head dim: {hd}")

    # Attention type
    atype = config.get("attention_type", "unknown")
    extras = []
    q_lr = config.get("q_lora_rank")
    o_lr = config.get("o_lora_rank")
    if q_lr:
        extras.append(f"Q LoRA: {q_lr}")
    if o_lr:
        extras.append(f"O LoRA: {o_lr}")
    extra_str = f", {', '.join(extras)}" if extras else ""
    print(f"  Attention: {atype}{extra_str}")

    # MoE
    if config.get("moe"):
        ne = config.get("num_experts", "?")
        topk = config.get("num_experts_per_tok", config.get("num_experts_per_tok", "?"))
        nse = config.get("num_shared_experts", 0)
        r_int = config.get("routed_expert_intermediate_size", "?")
        s_int = config.get("shared_expert_intermediate_size", "?")
        print(f"  MoE: {ne} experts, top-{topk}, {nse} shared"
              f" (routed_int={r_int}, shared_int={s_int})")
    else:
        int_size = config.get("intermediate_size", "?")
        print(f"  FFN: intermediate_size={int_size}")

    # MHC
    if config.get("mhc"):
        mhc_dim = config.get("mhc_bottleneck_dim", "?")
        print(f"  MHC: enabled (bottleneck_dim={mhc_dim})")

    # NSA
    idx_heads = config.get("index_n_heads")
    if idx_heads:
        idx_dim = config.get("index_head_dim", "?")
        idx_topk = config.get("index_topk", "?")
        rope_dim = config.get("qk_rope_head_dim", "?")
        sw = config.get("sliding_window", "?")
        print(f"  NSA indexer: {idx_heads} heads x {idx_dim} dim, topk={idx_topk}")
        print(f"  NSA rope: qk_rope_head_dim={rope_dim}, sliding_window={sw}")

    # compress_ratios distribution
    if compress_ratios:
        from collections import Counter
        type_counts = Counter()
        for i, cr in enumerate(compress_ratios):
            lbl = _layer_type_label(cr, i, num_layers, num_hash_layers)
            type_counts[lbl] += 1
        dist = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in sorted(type_counts.items()))
        print(f"  Layer distribution: {dist}")

    print("=" * 80)


# ── compute-flow output ────────────────────────────────────────────────────

def print_compute_flow(kernels, layer_id, compress_ratios, num_hash_layers,
                       num_layers, profile: ModelProfile, config=None, trace_start_ts=0):
    """Print compute flow table with model architecture summary.

    This is the bridge format for model-compute-simulation.
    Columns: # | Half | Category | Simplified Name | dur(us) | % | ts_rel(ms) | Input Dims
    No category-level summary — every kernel is shown individually.
    """
    # Model architecture summary
    if config:
        print_model_architecture(config, compress_ratios, num_hash_layers, num_layers)

    total = sum(k["dur"] for k in kernels)
    cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else -1
    type_label = _layer_type_label(cr, layer_id, num_layers, num_hash_layers)
    cr_label = f" (compress_ratio={cr})" if cr >= 0 else ""

    # Layer time offset relative to trace start
    layer_start_ts = kernels[0]["ts"] if kernels else 0
    layer_offset_ms = (layer_start_ts - trace_start_ts) / 1000.0

    print(f"\n{'=' * 120}")
    print(f"  Compute Flow: Layer {layer_id} [{type_label}]{cr_label}"
          f"  —  {total:.0f}us ({total/1000:.2f}ms), {len(kernels)} kernels"
          f"  | offset={layer_offset_ms:.3f}ms")
    print(f"{'=' * 120}")

    # Per-kernel table with Category, ts_rel, Input Dims columns
    print(f"\n  {'#':>3s} {'Half':>4s} {'Category':<16s} {'Simplified Name':<48s}"
          f" {'dur(us)':>8s} {'%':>5s} {'ts_rel(ms)':>11s} {'Input Dims':<20s}")
    print(f"  {'-' * 118}")
    for i, k in enumerate(kernels):
        _, cat_key = classify_kernel(k["name"], profile)
        name = simplify_name(k["name"], profile)
        pct = k["dur"] / total * 100
        ts_rel_ms = (k["ts"] - trace_start_ts) / 1000.0
        input_dims = k.get("input_dims") or ""
        input_dims_str = str(input_dims) if input_dims else "—"
        # Truncate name if too long
        if len(name) > 48:
            name = name[:45] + "..."
        if len(input_dims_str) > 20:
            input_dims_str = input_dims_str[:17] + "..."
        print(f"  {i:>3d} {k['half']:>4s} {cat_key:<16s} {name:<48s}"
              f" {k['dur']:>7.1f} {pct:>4.1f}% {ts_rel_ms:>10.3f} {input_dims_str:<20s}")

    return set(simplify_name(k["name"], profile) for k in kernels)


# ── JSON output ──────────────────────────────────────────────────────────

def format_json_output(kernels, layer_id, fwd_pass, compress_ratios, num_layers,
                       profile: ModelProfile):
    """Format per-kernel detail as JSON for downstream consumption.

    Returns a JSON string with:
    - kernels: per-kernel list with name, simplified_name, dur_us, category (machine key), half, ts
    - category_summary: aggregated {key: {dur_us, count}}
    - metadata: layer_id, fwd_pass, compress_ratio, wall_us, total_dur_us, num_kernels, num_layers
    """
    total_dur = sum(k["dur"] for k in kernels)
    cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else -1
    wall_start = kernels[0]["ts"] if kernels else 0
    wall_end = (kernels[-1]["ts"] + kernels[-1]["dur"]) if kernels else 0

    # Per-kernel detail
    kernel_list = []
    cat_summary = defaultdict(lambda: {"dur_us": 0, "count": 0})
    for k in kernels:
        cat_key = classify_kernel_key(k["name"], profile)
        cat_summary[cat_key]["dur_us"] += k["dur"]
        cat_summary[cat_key]["count"] += 1
        kernel_list.append({
            "name": k["name"],
            "simplified_name": simplify_name(k["name"], profile),
            "dur_us": k["dur"],
            "category": cat_key,
            "half": k["half"],
            "ts": k["ts"],
            "input_dims": k.get("input_dims"),
            "output_dims": k.get("output_dims"),
        })

    result = {
        "metadata": {
            "layer_id": layer_id,
            "fwd_pass": fwd_pass,
            "compress_ratio": cr,
            "num_layers": num_layers,
            "wall_us": wall_end - wall_start,
            "total_dur_us": total_dur,
            "num_kernels": len(kernels),
        },
        "category_summary": dict(cat_summary),
        "kernels": kernel_list,
    }
    return json.dumps(result, indent=2)


# ── entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Per-layer kernel breakdown")
    ap.add_argument("--trace", required=True, help="Path to .trace.json(.gz)")
    ap.add_argument("--config", default=None, help="Path to model config.json")
    ap.add_argument("--profile", default=None,
                     help="Model profile name (dsv4_csa_hca, dsv3_mla, generic). "
                          "Auto-inferred from config if not specified.")
    ap.add_argument("--anchor-kernel", default=None,
                     help="Override anchor kernel substring for layer boundary detection")
    ap.add_argument("--fwd-pass", type=int, required=True, help="Forward pass index")
    ap.add_argument("--layer", type=int, required=True, help="Layer ID (0-based)")
    ap.add_argument("--compare-layer", type=int, default=None,
                     help="Second layer for side-by-side comparison")
    ap.add_argument("--num-layers", type=int, default=None,
                     help="Override number of layers")
    ap.add_argument("--format", choices=["text", "json", "compute-flow"], default="text",
                     help="Output format: text (human-readable), compute-flow (bridge format for model-compute-simulation), or json (machine-readable per-kernel detail)")
    args = ap.parse_args()

    events = load_trace(args.trace)
    gpu = sorted([e for e in events if e.get("cat") == "kernel"], key=lambda e: e.get("ts", 0))
    trace_start_ts = min((e.get("ts", float("inf")) for e in events if e.get("ts") is not None), default=0)

    config = load_config(args.config)

    # Resolve profile
    if args.profile:
        profile = get_profile(args.profile)
    else:
        profile = infer_profile(config)

    # Determine anchor kernel
    anchor_kernel = args.anchor_kernel
    if not anchor_kernel:
        if profile.anchor_kernel:
            anchor_kernel = profile.anchor_kernel
        else:
            # Scan trace for common anchors
            for candidate in ["mhc_post_tilelang", "flash_fwd_mla_combine", "AllReduce"]:
                if sum(1 for e in gpu if candidate in e.get("name", "")) >= 4:
                    anchor_kernel = candidate
                    break
            if not anchor_kernel:
                print("ERROR: cannot auto-detect anchor kernel. Use --profile or --anchor-kernel.",
                      file=sys.stderr)
                sys.exit(1)

    anchor_indices = [i for i, e in enumerate(gpu) if anchor_kernel in e.get("name", "")]

    compress_ratios = get_compress_ratios(config)
    num_hash_layers = config.get("num_hash_layers", 0)
    num_layers = args.num_layers or config.get("num_hidden_layers", profile.default_num_layers)

    cr = compress_ratios[args.layer] if args.layer < len(compress_ratios) else -1

    kernels = get_layer_kernels(gpu, anchor_indices, args.fwd_pass, args.layer, num_layers, profile)
    if not kernels:
        print(f"ERROR: no kernels found for fwd_pass={args.fwd_pass}, layer={args.layer}",
              file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(format_json_output(kernels, args.layer, args.fwd_pass,
                                 compress_ratios, num_layers, profile))
    elif args.format == "compute-flow":
        print_compute_flow(kernels, args.layer, compress_ratios,
                           num_hash_layers, num_layers, profile, config=config,
                           trace_start_ts=trace_start_ts)
    else:
        names_a = print_layer_breakdown(kernels, args.layer, compress_ratios,
                                          num_hash_layers, num_layers, profile)

    if args.compare_layer is not None:
        kernels_b = get_layer_kernels(gpu, anchor_indices, args.fwd_pass,
                                        args.compare_layer, num_layers, profile)
        if not kernels_b:
            print(f"WARNING: no kernels for compare-layer {args.compare_layer}", file=sys.stderr)
        else:
            cr_b = compress_ratios[args.compare_layer] if args.compare_layer < len(compress_ratios) else -1
            names_b = print_layer_breakdown(
                kernels_b, args.compare_layer, compress_ratios,
                num_hash_layers, num_layers, profile,
                label=f"Layer {args.compare_layer} (c_ratio={cr_b})")
            print_diff(kernels, kernels_b,
                       f"Layer {args.layer} (c={cr})",
                       f"Layer {args.compare_layer} (c={cr_b})",
                       profile)


if __name__ == "__main__":
    main()
