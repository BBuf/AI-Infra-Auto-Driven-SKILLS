"""Append layer guide slices to Chrome JSON without changing recorded events."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

CATEGORY = "layer_guide"
NOTICE = (
    "Anchor-to-anchor navigation guides, not full layer spans or exclusive "
    "kernel ownership. The last layer ends at a verified terminal kernel, "
    "or is a point marker without a terminal. Chrome timestamps are us."
    " Only guide endpoints are rounded to the nearest us to avoid floating-point overlap."
)


def read_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def event_list(document):
    events = document if isinstance(document, list) else document.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Expected a Chrome event array or an object with traceEvents")
    return events


def finite_time(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def annotate(
    document,
    *,
    anchor_regex,
    num_layers,
    anchor_offset,
    passes,
    phase,
    evidence,
    pid=None,
    device=None,
    first_layer=0,
    end_anchor_regex=None,
):
    if num_layers < 1 or passes < 1 or anchor_offset < 0 or first_layer < 0:
        raise ValueError(
            "Layer/pass counts must be positive; offsets must be nonnegative"
        )
    if not evidence.strip() or not phase.strip():
        raise ValueError(
            "Supply a phase and evidence establishing the first layer/anchor"
        )
    events = event_list(document)
    if any(e.get("cat") == CATEGORY for e in events):
        raise ValueError(
            "Trace already has a layer guide; start from the original trace"
        )
    pattern = re.compile(anchor_regex)
    kernels = [
        e
        for e in events
        if e.get("ph") == "X" and "kernel" in e.get("cat", "").split(",")
    ]
    anchors = [e for e in kernels if pattern.search(e.get("name", ""))]
    if pid is not None:
        anchors = [e for e in anchors if str(e.get("pid")) == str(pid)]
    if device is not None:
        anchors = [
            e for e in anchors if str(e.get("args", {}).get("device")) == str(device)
        ]
    pids = {e.get("pid") for e in anchors}
    if len(pids) != 1 or None in pids:
        raise ValueError(
            f"Expected one anchor PID, found {sorted(map(str, pids))}; use --pid"
        )
    selected_pid = next(iter(pids))
    kernels = [e for e in kernels if e.get("pid") == selected_pid]
    devices = {e.get("args", {}).get("device") for e in anchors}
    if len(devices) > 1:
        raise ValueError(
            "Matching anchors span multiple devices in one PID; use --device"
        )
    selected_device = next(iter(devices))
    if selected_device is not None:
        kernels = [
            e for e in kernels if e.get("args", {}).get("device") == selected_device
        ]
    if any(
        not finite_time(e.get("ts")) or not finite_time(e.get("dur")) or e["dur"] < 0
        for e in kernels
    ):
        raise ValueError("Selected GPU contains invalid kernel timestamps/durations")
    anchors.sort(key=lambda e: e["ts"])
    count = num_layers * passes
    selected = anchors[anchor_offset : anchor_offset + count]
    if len(selected) != count:
        raise ValueError(
            f"Need {count} anchors from offset {anchor_offset}; found {len(selected)}"
        )
    if any(a["ts"] >= b["ts"] for a, b in zip(selected, selected[1:])):
        raise ValueError("Anchors must have distinct increasing timestamps")
    terminal_pattern = re.compile(end_anchor_regex) if end_anchor_regex else None
    terminals = [
        e for e in kernels if terminal_pattern and terminal_pattern.search(e["name"])
    ]
    existing_tids = {str(e.get("tid")) for e in events if e.get("pid") == selected_pid}
    guide_tid = 1_000_000
    while str(guide_tid) in existing_tids:
        guide_tid += 1
    track_name = f"Layer guide ({phase}) — anchor intervals"
    added = [
        {
            "ph": "M",
            "name": "thread_name",
            "pid": selected_pid,
            "tid": guide_tid,
            "args": {"name": track_name},
        },
        {
            "ph": "M",
            "name": "thread_sort_index",
            "pid": selected_pid,
            "tid": guide_tid,
            "args": {"sort_index": -1000},
        },
    ]
    origin = min(e["ts"] for e in events if finite_time(e.get("ts")))
    rows = []
    for index, anchor in enumerate(selected):
        pass_id, layer_index = divmod(index, num_layers)
        layer_id = first_layer + layer_index
        global_index = anchor_offset + index
        next_anchor = (
            anchors[global_index + 1] if global_index + 1 < len(anchors) else None
        )
        upper = next_anchor["ts"] if next_anchor else math.inf
        terminal = None
        if terminal_pattern:
            matches = [e for e in terminals if anchor["ts"] <= e["ts"] < upper]
            if len(matches) != 1:
                raise ValueError(
                    f"Pass {pass_id} L{layer_id}: expected one terminal, found {len(matches)}"
                )
            terminal = matches[0]
            if terminal["ts"] + terminal["dur"] > upper:
                raise ValueError(
                    f"Pass {pass_id} L{layer_id}: terminal overlaps next anchor"
                )
        if layer_index < num_layers - 1:
            end, boundary = selected[index + 1]["ts"], "next_layer_anchor"
        elif terminal:
            end, boundary = terminal["ts"] + terminal["dur"], "terminal_kernel_end"
        else:
            end, boundary = anchor["ts"], "anchor_point_only"
        # Large epoch-like timestamps lose sub-us precision during subtraction.
        # Integer-us auxiliary endpoints avoid spurious 1 ns overlap in Perfetto.
        # Original events and the exact anchor values in the report stay intact.
        guide_start, guide_end = round(anchor["ts"]), round(end)
        row = {
            "pass_index": pass_id,
            "layer_id": layer_id,
            "anchor_index": global_index,
            "anchor_name": anchor["name"],
            "anchor_tid": anchor.get("tid"),
            "start_us": anchor["ts"],
            "end_us": end,
            "guide_start_us": guide_start,
            "guide_end_us": guide_end,
            "start_relative_ms": (anchor["ts"] - origin) / 1000,
            "end_relative_ms": (end - origin) / 1000,
            "boundary_kind": boundary,
        }
        rows.append(row)
        added.append(
            {
                "ph": "X",
                "cat": CATEGORY,
                "name": f"L{layer_id}",
                "pid": selected_pid,
                "tid": guide_tid,
                "ts": guide_start,
                "dur": guide_end - guide_start,
                "args": {**row, "phase": phase, "evidence": evidence, "notice": NOTICE},
            }
        )
    output_events = events + added
    output = (
        output_events
        if isinstance(document, list)
        else {**document, "traceEvents": output_events}
    )
    report = {
        "schema_version": 1,
        "notice": NOTICE,
        "phase": phase,
        "evidence": evidence,
        "pid": selected_pid,
        "device": selected_device,
        "guide_tid": guide_tid,
        "track_name": track_name,
        "anchor_regex": anchor_regex,
        "end_anchor_regex": end_anchor_regex,
        "anchor_offset": anchor_offset,
        "matching_anchor_count": len(anchors),
        "num_layers": num_layers,
        "first_layer": first_layer,
        "passes": passes,
        "original_event_count": len(events),
        "added_event_count": len(added),
        "trace_origin_us": origin,
        "layers": rows,
    }
    return output, report


def write_trace(path, document):
    # Exclusive creation also prevents replacement through existing symlinks.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-regex", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument(
        "--anchor-offset",
        type=int,
        required=True,
        help="Index of a verified first layer in sorted anchors",
    )
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--pid")
    parser.add_argument(
        "--device", help="GPU device from kernel args.device, when needed"
    )
    parser.add_argument("--end-anchor-regex")
    args = parser.parse_args()
    report_path = Path(str(args.output) + ".layers.json")
    try:
        if (
            args.trace.resolve() == args.output.resolve()
            or args.output.exists()
            or report_path.exists()
        ):
            raise ValueError(
                "Choose a new output path; source/output/report files are never overwritten"
            )
        output, report = annotate(
            read_trace(args.trace),
            anchor_regex=args.anchor_regex,
            num_layers=args.num_layers,
            anchor_offset=args.anchor_offset,
            passes=args.passes,
            phase=args.phase,
            evidence=args.evidence,
            pid=args.pid,
            device=args.device,
            first_layer=args.first_layer,
            end_anchor_regex=args.end_anchor_regex,
        )
        with args.trace.open("rb") as source:
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        report["source_sha256"] = digest.hexdigest()
        report["source_file"] = args.trace.name
        write_trace(args.output, output)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Annotated trace: {args.output}")
    print(f"Mapping report: {report_path}")
    print(
        f"Added {args.num_layers * args.passes} labels; original {report['original_event_count']} events retained."
    )
    print(NOTICE)


if __name__ == "__main__":
    main()
