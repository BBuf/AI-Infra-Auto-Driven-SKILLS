from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills/torch-profiler-layer-track/scripts"
)
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location(
        "compact_gpu_tracks", SCRIPTS / "compact_gpu_tracks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def kernel(ts, tid, *, pid=0, dur=5, device=0):
    return dict(
        ph="X",
        cat="kernel",
        name="kernel",
        pid=pid,
        tid=tid,
        ts=ts,
        dur=dur,
        args={"stream": tid, "device": device, "correlation": tid},
    )


def restore(result, report):
    """Independent exact reconstruction, including absent args and metadata order."""
    out = copy.deepcopy(module.event_list(result))
    out = out[: -report["new_metadata_count"]]
    for item in sorted(
        report["retired_track_metadata"], key=lambda x: x["source_event_index"]
    ):
        out.insert(item["source_event_index"], item["event"])
    for index, event in enumerate(out):
        info = event.get("args", {}).pop(module.PROVENANCE, None)
        if info:
            assert info["source_event_index"] == index
            event["pid"], event["tid"] = info["pid"], info["tid"]
            if not info["had_args"]:
                del event["args"]
    return out


def test_many_streams_pack_without_modifying_timing_or_cpu_and_keep_flows():
    events = [dict(ph="X", cat="cpu_op", name="CPU", pid=30, tid=1, ts=0, dur=5000)]
    for i in range(200):
        events += [
            dict(
                ph="M", name="thread_name", pid=0, tid=i, args={"name": f"stream {i}"}
            ),
            kernel(i * 2, i),
            dict(ph="s", cat="ac2g", pid=30, tid=1, ts=i * 2, id=i),
            dict(ph="f", cat="ac2g", pid=0, tid=i, ts=i * 2, id=i, bp="e"),
        ]
    events.append(
        dict(ph="X", cat="layer_guide", name="L0", pid=0, tid=1000000, ts=0, dur=400)
    )
    original = dict(traceEvents=events, custom={"keep": [1, 2]})
    before = copy.deepcopy(original)
    output, report = module.compact(original)
    assert original == before
    assert restore(output, report) == events
    assert output["custom"] == original["custom"]
    assert report["groups"][0]["lanes"] == 3
    assert report["groups"][0]["original_kernel_streams"] == 200
    assert report["relocated_flows"] == 200
    assert events[0] in output["traceEvents"] and events[-1] in output["traceEvents"]
    for event in output["traceEvents"]:
        if event.get("ph") == "f":
            target = next(
                e
                for e in output["traceEvents"]
                if e.get("cat") == "kernel" and e["ts"] == event["ts"]
            )
            assert event["tid"] == target["tid"]


def test_overlapping_same_stream_and_touching_intervals_are_flat():
    events = [kernel(0, 10), kernel(2, 10), kernel(5, 11), kernel(7, 11)]
    result, report = module.compact(events)
    assert report["groups"][0]["lanes"] == 2
    assert restore(result, report) == events
    assert result[0]["tid"] != result[1]["tid"]
    assert result[0]["tid"] == result[2]["tid"]


def test_rank_device_selection_does_not_merge_devices_or_change_other_rank():
    events = [kernel(0, 10), kernel(0, 11, device=1), kernel(0, 12, pid=2)]
    output, report = module.compact(events, pid="0")
    assert len(report["groups"]) == 2
    assert output[0]["tid"] != output[1]["tid"]
    assert output[2] == events[2]
    assert restore(output, report) == events


def test_gpu_scopes_preserved_in_separate_auxiliary_process():
    events = [
        kernel(1, 10),
        dict(
            ph="X", cat="gpu_user_annotation", name="step", ts=0, dur=10, pid=0, tid=10
        ),
    ]
    result, report = module.compact(events)
    assert result[0]["pid"] == 0
    assert result[1]["pid"] != 0
    assert restore(result, report) == events
    assert [g["kind"] for g in report["groups"]] == ["activity", "scope"]


def test_refuses_to_hide_more_than_ten_concurrent_activities():
    events = [kernel(0, i) for i in range(11)]
    before = copy.deepcopy(events)
    with pytest.raises(ValueError, match="more than 10"):
        module.compact(events)
    assert before == events


@pytest.mark.parametrize("max_lanes", [0, 11, -1])
def test_invalid_limit(max_lanes):
    with pytest.raises(ValueError, match="between 1 and 10"):
        module.compact([kernel(0, 1)], max_lanes=max_lanes)


def test_rejects_ambiguous_flow_and_unsupported_track_events():
    events = [kernel(0, 1), kernel(0, 1), dict(ph="f", pid=0, tid=1, ts=0)]
    with pytest.raises(ValueError, match="uniquely bind"):
        module.compact(events)
    events = [kernel(0, 1), dict(ph="X", cat="unknown", pid=0, tid=1, ts=1, dur=1)]
    with pytest.raises(ValueError, match="Unsupported"):
        module.compact(events)


@pytest.mark.parametrize("ts,dur", [(float("nan"), 1), (1, -1), (1, float("inf"))])
def test_rejects_invalid_intervals(ts, dur):
    with pytest.raises(ValueError, match="invalid"):
        module.compact([kernel(ts, 1, dur=dur)])


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_cli_preserves_source_and_refuses_overwrite_or_recompaction(tmp_path, suffix):
    source, output = tmp_path / ("in" + suffix), tmp_path / ("out" + suffix)
    events = [kernel(0, 3), kernel(10, 4)]
    module.write_trace(source, events)
    command = [
        sys.executable,
        str(SCRIPTS / "compact_gpu_tracks.py"),
        "--trace",
        str(source),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    report = json.loads(Path(str(output) + ".compact.json").read_text())
    assert restore(module.read_trace(output), report) == events
    assert module.read_trace(source) == events
    assert len(report["source_sha256"]) == 64
    original_bytes = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert output.read_bytes() == original_bytes
    with pytest.raises(ValueError, match="already compacted"):
        module.compact(module.read_trace(output))


def test_annotation_cli_combines_flat_layers_and_compact_lanes(tmp_path):
    source, output = tmp_path / "source.json", tmp_path / "out.json"
    module.write_trace(source, [kernel(100 + i * 10, i) for i in range(3)])
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "add_layer_track.py"),
            "--trace",
            str(source),
            "--output",
            str(output),
            "--anchor-regex",
            "^kernel$",
            "--num-layers",
            "3",
            "--anchor-offset",
            "0",
            "--phase",
            "test",
            "--evidence",
            "verified fixture",
            "--max-gpu-lanes",
            "10",
        ],
        check=True,
        capture_output=True,
    )
    report = json.loads(Path(str(output) + ".layers.json").read_text())
    assert report["compaction"]["groups"][0]["lanes"] == 1
    guides = [e for e in module.read_trace(output) if e.get("cat") == "layer_guide"]
    assert [e["name"] for e in guides] == ["L0", "L1", "L2"]
    assert all(e["pid"] == 0 for e in guides)
