"""Pack GPU activity into <=10 synthetic display lanes, preserving real timing."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import heapq
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from add_layer_track import event_list, finite_time, read_trace, write_trace

ACTIVITY = {"kernel", "gpu_memcpy", "gpu_memset"}
PROVENANCE = "_compact_gpu_track"
NOTICE = (
    "Synthetic GPU display lanes, NOT CUDA streams or runtime stream reduction. "
    "Kernel names, ts/dur, original args.stream and CPU events are unchanged. "
    "Original pid/tid and source event index are in each relocated event's "
    "args._compact_gpu_track. Use the original trace for stream scheduling analysis."
)


def bounds_ns(event):
    """Conservative import bounds; never round or modify recorded timestamps."""
    ts, dur = event.get("ts"), event.get("dur")
    if not finite_time(ts) or not finite_time(dur) or dur < 0:
        raise ValueError("GPU activity has invalid timestamps/durations")
    # Account for both decimal parsing and binary float conversion to ns. Avoid
    # Perfetto overflow lanes from two sub-ns-rounded, nearly adjacent X events.
    start = Decimal(str(ts)) * 1000
    duration = Decimal(str(dur)) * 1000
    return (
        math.floor(min(start, ts * 1000)),
        math.ceil(max(start, ts * 1000)) + math.ceil(max(duration, dur * 1000)),
    )


def compact(document, *, max_lanes=10, pid=None):
    if not 1 <= max_lanes <= 10:
        raise ValueError("max_lanes must be between 1 and 10")
    events = event_list(document)
    if any(PROVENANCE in e.get("args", {}) for e in events):
        raise ValueError("Trace is already compacted; start from the source trace")
    groups = defaultdict(list)
    for index, event in enumerate(events):
        if (
            event.get("ph") == "X"
            and ACTIVITY.intersection(event.get("cat", "").split(","))
            and (pid is None or str(event.get("pid")) == str(pid))
        ):
            if event.get("pid") is None or event.get("tid") is None:
                raise ValueError("GPU activity requires pid and tid")
            group = (event["pid"], event.get("args", {}).get("device"), "activity")
            groups[group].append((index, event))
    if not groups:
        raise ValueError("No GPU activity matches the selection")
    occupied = defaultdict(set)
    for event in events:
        occupied[event.get("pid")].add(str(event.get("tid")))
    gpu_pids = {key[0] for key in groups}
    for index, event in enumerate(events):
        if (
            event.get("ph") == "X"
            and event.get("cat") == "gpu_user_annotation"
            and event.get("pid") in gpu_pids
        ):
            groups[(event["pid"], None, "scope")].append((index, event))
    replacements = {}
    lane_metadata = []
    group_reports = []
    by_original_track = defaultdict(list)
    for (gpu_pid, device, kind), items in groups.items():
        display_pid = gpu_pid
        if kind == "scope":
            display_pid = 2_000_000
            while str(display_pid) in {str(p) for p in occupied}:
                display_pid += 1
            occupied[display_pid] = set()
            lane_metadata.append(
                dict(
                    ph="M",
                    name="process_name",
                    pid=display_pid,
                    args={"name": f"GPU scopes (original PID {gpu_pid})"},
                )
            )
        active = []
        lane_tids = []
        next_tid = 2_000_000
        for index, event in sorted(items, key=lambda item: (item[1]["ts"], item[0])):
            start, end = bounds_ns(event)
            if active and active[0][0] <= start:
                _, lane = heapq.heappop(active)
            else:
                lane = len(lane_tids)
                if lane >= max_lanes:
                    raise ValueError(
                        f"PID {gpu_pid} device {device} needs more than {max_lanes} "
                        "lanes for overlapping activity; cannot compact without "
                        "hiding overlap. Select a smaller time range or keep streams."
                    )
                while str(next_tid) in occupied[display_pid]:
                    next_tid += 1
                lane_tids.append(next_tid)
                occupied[display_pid].add(str(next_tid))
            tid = lane_tids[lane]
            heapq.heappush(active, (end, lane))
            replacements[index] = (display_pid, tid)
            if kind == "activity":
                by_original_track[(event["pid"], event["tid"])].append(
                    (event["ts"], event["ts"] + event["dur"], index, tid)
                )
        for lane, tid in enumerate(lane_tids):
            label = (
                f"GPU lane {lane + 1} (synthetic; device {device})"
                if kind == "activity"
                else f"GPU scope lane {lane + 1} (synthetic)"
            )
            lane_metadata.extend(
                [
                    dict(
                        ph="M",
                        name="thread_name",
                        pid=display_pid,
                        tid=tid,
                        args={"name": label},
                    ),
                    dict(
                        ph="M",
                        name="thread_sort_index",
                        pid=display_pid,
                        tid=tid,
                        args={"sort_index": lane},
                    ),
                ]
            )
        group_reports.append(
            dict(
                pid=gpu_pid,
                display_pid=display_pid,
                device=device,
                kind=kind,
                original_streams=len({e["tid"] for _, e in items}),
                original_kernel_streams=len(
                    {
                        e["tid"]
                        for _, e in items
                        if "kernel" in e.get("cat", "").split(",")
                    }
                ),
                kernel_count=sum(
                    "kernel" in e.get("cat", "").split(",") for _, e in items
                ),
                activity_count=len(items),
                lanes=len(lane_tids),
                lane_tids=lane_tids,
            )
        )

    starts = {}
    for key, intervals in by_original_track.items():
        intervals.sort()
        starts[key] = [item[0] for item in intervals]
    relocated_tracks = {(events[i]["pid"], events[i]["tid"]) for i in replacements}
    retired = []
    flows = 0
    for index, event in enumerate(events):
        key = (event.get("pid"), event.get("tid"))
        if key not in relocated_tracks or index in replacements:
            continue
        if event.get("ph") == "M" and event.get("name") in (
            "thread_name",
            "thread_sort_index",
        ):
            retired.append(dict(source_event_index=index, event=event))
        elif event.get("ph") in ("s", "t", "f"):
            intervals = by_original_track.get(key, [])
            ts = event.get("ts")
            if not finite_time(ts):
                raise ValueError("GPU flow has invalid timestamp")
            pos = bisect.bisect_right(starts.get(key, []), ts)
            # Prefer the exact kernel start (Torch ac2g), then a unique containing
            # interval. Ambiguous bindings must not silently point at another op.
            matches = [item for item in intervals[:pos] if item[0] == ts]
            if not matches:
                matches = [item for item in intervals[:pos] if item[0] <= ts <= item[1]]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot uniquely bind GPU flow at source event {index}"
                )
            replacements[index] = (event["pid"], matches[0][3])
            flows += 1
        else:
            raise ValueError(
                f"Unsupported event {event.get('ph')}/{event.get('cat')} on a GPU "
                "activity track; use a GPU-only capture or explicitly handle it first"
            )
    retired_indices = {item["source_event_index"] for item in retired}
    output_events = []
    for index, event in enumerate(events):
        if index in retired_indices:
            continue
        if index in replacements:
            event = {
                **event,
                "pid": replacements[index][0],
                "tid": replacements[index][1],
                "args": {
                    **event.get("args", {}),
                    PROVENANCE: dict(
                        pid=event["pid"],
                        tid=event["tid"],
                        source_event_index=index,
                        had_args="args" in event,
                    ),
                },
            }
        output_events.append(event)
    output_events.extend(lane_metadata)
    result = (
        output_events
        if isinstance(document, list)
        else {**document, "traceEvents": output_events}
    )
    report = dict(
        schema_version=1,
        notice=NOTICE,
        max_lanes=max_lanes,
        original_event_count=len(events),
        output_event_count=len(output_events),
        relocated_events=len(replacements),
        relocated_flows=flows,
        new_metadata_count=len(lane_metadata),
        groups=group_reports,
        retired_track_metadata=retired,
    )
    return result, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-gpu-lanes", type=int, default=10)
    p.add_argument(
        "--pid", help="Only compact this GPU PID; keep other processes intact"
    )
    a = p.parse_args()
    report_path = Path(str(a.output) + ".compact.json")
    try:
        if (
            a.trace.resolve() == a.output.resolve()
            or a.output.exists()
            or report_path.exists()
        ):
            raise ValueError(
                "Choose a new output path; never overwrite source/output/report"
            )
        result, report = compact(
            read_trace(a.trace), max_lanes=a.max_gpu_lanes, pid=a.pid
        )
        report["source_file"] = a.trace.name
        report["source_sha256"] = hashlib.sha256(a.trace.read_bytes()).hexdigest()
        write_trace(a.output, result)
        with report_path.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        p.exit(2, f"error: {exc}\n")
    print(a.output)
    print(json.dumps(report["groups"]))
    print(NOTICE)


if __name__ == "__main__":
    main()
