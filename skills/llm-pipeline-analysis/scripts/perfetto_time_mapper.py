#!/usr/bin/env python3
"""Perfetto UI time mapper for LLM torch profiler traces.

Converts absolute trace timestamps to Perfetto UI relative time,
and prints time ranges for key structures (forward passes, layers).

Usage:
  python3 perfetto_time_mapper.py \
    --trace /path/to/TP-0.trace.json.gz \
    --config /path/to/config.json \
    --fwd-pass 5 \
    --layers 2,3,38,42
"""

import argparse
import gzip
import json
import sys

from model_profiles import get_profile, infer_profile, normalize_compress_ratios


def load_trace(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return events


def main():
    ap = argparse.ArgumentParser(description="Perfetto UI time mapper")
    ap.add_argument("--trace", required=True, help="Path to .trace.json(.gz)")
    ap.add_argument("--config", default=None, help="Path to model config.json")
    ap.add_argument(
        "--profile",
        default=None,
        help="Model profile name (dsv4_csa_hca, dsv3_mla, generic). "
        "Auto-inferred from config if not specified.",
    )
    ap.add_argument(
        "--anchor-kernel",
        default=None,
        help="Override anchor kernel substring for layer boundary detection",
    )
    ap.add_argument(
        "--fwd-pass",
        type=int,
        default=None,
        help="Forward pass index to show layer detail",
    )
    ap.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer IDs to highlight (e.g. 2,3,38,42)",
    )
    ap.add_argument(
        "--num-layers", type=int, default=None, help="Override number of layers"
    )
    args = ap.parse_args()

    events = load_trace(args.trace)
    gpu = sorted(
        [e for e in events if e.get("cat") == "kernel"], key=lambda e: e.get("ts", 0)
    )

    all_ts = [e.get("ts", float("inf")) for e in events if e.get("ts") is not None]
    trace_start = min(all_ts)
    trace_end = max(
        e.get("ts", 0) + e.get("dur", 0)
        for e in events
        if e.get("ts") is not None and e.get("dur") is not None
    )
    trace_dur_s = (trace_end - trace_start) / 1e6

    # Resolve profile
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

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
            for candidate in [
                "mhc_post_tilelang",
                "flash_fwd_mla_combine",
                "AllReduce",
            ]:
                if sum(1 for e in gpu if candidate in e.get("name", "")) >= 4:
                    anchor_kernel = candidate
                    break
            if not anchor_kernel:
                print(
                    "ERROR: cannot auto-detect anchor kernel. Use --profile or --anchor-kernel.",
                    file=sys.stderr,
                )
                sys.exit(1)

    anchor_indices = [
        i for i, e in enumerate(gpu) if anchor_kernel in e.get("name", "")
    ]
    compress_ratios = normalize_compress_ratios(config, args.num_layers)
    num_hash_layers = config.get("num_hash_layers", 0)
    num_layers = args.num_layers or config.get(
        "num_hidden_layers", profile.default_num_layers
    )
    bpl = profile.blocks_per_layer
    blocks_per_pass = num_layers * bpl
    n_passes = (len(anchor_indices) - 1) // blocks_per_pass

    print(f"Trace: {args.trace}")
    print(f"Profile: {profile.name}, anchor: {anchor_kernel}")
    print(f"Perfetto 时间范围: 0.000s ~ {trace_dur_s:.3f}s")
    print(f"GPU kernels: {len(gpu)}, anchor blocks: {len(anchor_indices)}")
    print(f"Forward passes: {n_passes}")

    # All forward passes
    print(
        f"\n{'Fwd#':>5s}  {'Perfetto Start':>15s}  {'Perfetto End':>13s}  {'Duration':>10s}"
    )
    print("-" * 50)
    for p in range(n_passes):
        base = p * blocks_per_pass
        if base + blocks_per_pass >= len(anchor_indices):
            break
        first_k = gpu[anchor_indices[base]]
        last_k = gpu[anchor_indices[base + blocks_per_pass] - 1]
        start_s = (first_k.get("ts", 0) - trace_start) / 1e6
        end_s = (last_k.get("ts", 0) + last_k.get("dur", 0) - trace_start) / 1e6
        dur_ms = (end_s - start_s) * 1000
        print(f"  {p:>3d}  {start_s:>13.3f}s  {end_s:>11.3f}s  {dur_ms:>8.1f}ms")

    # Per-layer detail for chosen forward pass
    if args.fwd_pass is not None:
        p = args.fwd_pass
        base = p * blocks_per_pass
        if base + blocks_per_pass >= len(anchor_indices):
            print(f"ERROR: fwd_pass {p} out of range", file=sys.stderr)
            sys.exit(1)

        highlight = set()
        if args.layers:
            highlight = {int(x) for x in args.layers.split(",")}

        print(f"\nForward Pass #{p} — Perfetto 相对时间:")
        print(
            f"{'L':>3s} {'c_r':>4s} {'Perfetto Start':>15s} {'Perfetto End':>13s} "
            f"{'Wall(ms)':>9s} {'SumDur(ms)':>10s} {'Note'}"
        )
        print("-" * 75)

        for layer_id in range(num_layers):
            b1 = base + layer_id * bpl
            b2_end = base + (layer_id + 1) * bpl
            if b2_end >= len(anchor_indices):
                break

            first_k = gpu[anchor_indices[b1]]
            last_k = gpu[anchor_indices[b2_end] - 1]

            start_s = (first_k.get("ts", 0) - trace_start) / 1e6
            end_s = (last_k.get("ts", 0) + last_k.get("dur", 0) - trace_start) / 1e6
            wall_ms = (end_s - start_s) * 1000

            sum_dur = sum(
                gpu[j].get("dur", 0)
                for j in range(anchor_indices[b1], anchor_indices[b2_end])
            )
            sum_ms = sum_dur / 1000

            cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else -1

            note = ""
            if layer_id in highlight:
                note = "◀"
            hash_start = num_layers - num_hash_layers if num_hash_layers else num_layers
            if layer_id == 0:
                note += " FIRST"
            elif layer_id >= hash_start:
                note += " HASH"
            elif layer_id == num_layers - 1:
                note += " FINAL"

            print(
                f"  {layer_id:>2d}  {cr:>3d}  {start_s:>13.3f}s  {end_s:>11.3f}s  "
                f"{wall_ms:>8.2f}  {sum_ms:>9.2f}  {note}"
            )

    print(f"\n提示: 在 Perfetto UI 中拖到对应时间范围即可查看")


if __name__ == "__main__":
    main()
