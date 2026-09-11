from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills/torch-profiler-layer-track/scripts"
)


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


track = load_module("add_layer_track")
viewer = load_module("serve_local_trace")


def kernel(name, ts, pid=7, tid=10, dur=2):
    return dict(ph="X", cat="kernel", name=name, ts=ts, dur=dur, pid=pid, tid=tid)


def fixture_trace():
    # Two passes, changing streams, draft work in the gap, CPU/flow metadata.
    events = [
        dict(ph="M", name="thread_name", pid=7, tid=1000000, args={"name": "occupied"}),
        dict(ph="X", cat="cpu_op", name="CPU scope", ts=90, dur=400, pid=42, tid=1),
        dict(ph="s", cat="forward_backward", name="flow", ts=95, pid=42, tid=1, id=3),
    ]
    for start in (100, 400):
        for layer in range(3):
            events += [
                kernel("anchor", start + layer * 20, tid=10 + layer),
                kernel("terminal", start + layer * 20 + 10, tid=12 - layer),
            ]
    events.append(kernel("draft", 250, dur=30))
    return {
        "schemaVersion": 1,
        "displayTimeUnit": "ms",
        "custom": {"keep": True},
        "traceEvents": list(reversed(events)),
    }


def annotate(document, **overrides):
    args = dict(
        anchor_regex="^anchor$",
        num_layers=3,
        anchor_offset=0,
        passes=2,
        phase="target-verify",
        evidence="fixture source: once per layer; first L0 verified",
        end_anchor_regex="^terminal$",
    )
    return track.annotate(document, **(args | overrides))


def test_preserves_events_and_metadata_and_does_not_label_draft_gap():
    original = fixture_trace()
    before = copy.deepcopy(original)
    result, report = annotate(original)
    count = len(before["traceEvents"])
    assert original == before
    assert result["traceEvents"][:count] == before["traceEvents"]
    assert {k: v for k, v in result.items() if k != "traceEvents"} == {
        k: v for k, v in before.items() if k != "traceEvents"
    }
    markers = [e for e in result["traceEvents"] if e.get("cat") == "layer_guide"]
    assert [(e["name"], e["ts"], e["dur"]) for e in markers] == [
        ("L0", 100, 20),
        ("L1", 120, 20),
        ("L2", 140, 12),
        ("L0", 400, 20),
        ("L1", 420, 20),
        ("L2", 440, 12),
    ]
    assert report["guide_tid"] == 1000001
    assert report["layers"][0]["start_relative_ms"] == 0.01
    assert [r["anchor_tid"] for r in report["layers"]] == [10, 11, 12] * 2


def test_point_last_layer_and_known_partition_offset():
    result, report = annotate(
        fixture_trace()["traceEvents"],
        anchor_offset=3,
        passes=1,
        first_layer=20,
        end_anchor_regex=None,
    )
    assert isinstance(result, list)
    assert [r["layer_id"] for r in report["layers"]] == [20, 21, 22]
    assert report["layers"][-1]["start_us"] == report["layers"][-1]["end_us"] == 440
    assert report["layers"][-1]["boundary_kind"] == "anchor_point_only"


def test_large_fractional_timestamps_do_not_overlap_on_guide():
    original = fixture_trace()
    for event in original["traceEvents"]:
        if "ts" in event:
            event["ts"] += 6148459900000.927
    output, _ = annotate(original)
    markers = [e for e in output["traceEvents"] if e.get("cat") == "layer_guide"]
    assert all(isinstance(e["ts"], int) and isinstance(e["dur"], int) for e in markers)
    assert all(a["ts"] + a["dur"] <= b["ts"] for a, b in zip(markers, markers[1:]))
    assert (
        output["traceEvents"][: len(original["traceEvents"])] == original["traceEvents"]
    )


def test_requires_rank_selection_and_ignores_other_rank():
    original = fixture_trace()
    original["traceEvents"].append(kernel("anchor", 100, pid=8))
    with pytest.raises(ValueError, match="one anchor PID"):
        annotate(original)
    result, report = annotate(original, pid="7")
    assert report["pid"] == 7
    assert kernel("anchor", 100, pid=8) in result["traceEvents"]


def test_multiple_devices_in_one_pid_need_selection():
    original = fixture_trace()
    kernels = [e for e in original["traceEvents"] if e.get("cat") == "kernel"]
    for e in kernels:
        e["args"] = {"device": 0}
    other_device = copy.deepcopy(kernels)
    for e in other_device:
        e["args"]["device"] = 1
    original["traceEvents"] += other_device
    with pytest.raises(ValueError, match="multiple devices"):
        annotate(original)
    output, report = annotate(original, device="0")
    assert report["device"] == 0
    assert report["matching_anchor_count"] == 6
    assert (
        output["traceEvents"][: len(original["traceEvents"])] == original["traceEvents"]
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"passes": 3}, "Need 9 anchors"),
        ({"anchor_regex": "^missing$"}, "one anchor PID"),
        ({"end_anchor_regex": "^missing$"}, "expected one terminal"),
        ({"anchor_offset": -1}, "offsets"),
        ({"evidence": ""}, "evidence"),
    ],
)
def test_rejects_unverifiable_selections(overrides, message):
    with pytest.raises(ValueError, match=message):
        annotate(fixture_trace(), **overrides)


def test_rejects_duplicate_anchors_and_ambiguous_terminals():
    original = fixture_trace()
    original["traceEvents"].append(kernel("anchor", 100))
    with pytest.raises(ValueError, match="distinct"):
        annotate(original)
    original = fixture_trace()
    original["traceEvents"].append(kernel("terminal", 111))
    with pytest.raises(ValueError, match="expected one terminal"):
        annotate(original)
    original = fixture_trace()
    original["traceEvents"].append(kernel("terminal", 390, dur=20))
    with pytest.raises(ValueError, match="expected one terminal"):
        annotate(original)
    result, _ = annotate(fixture_trace())
    with pytest.raises(ValueError, match="already has"):
        annotate(result)


def test_rejects_terminal_crossing_next_layer():
    original = fixture_trace()
    for e in original["traceEvents"]:
        if e.get("name") == "terminal" and e["ts"] == 110:
            e["dur"] = 20
    with pytest.raises(ValueError, match="overlaps"):
        annotate(original)


@pytest.mark.parametrize("compressed", [False, True])
def test_cli_round_trip_and_refuses_overwrite(tmp_path, compressed):
    source = tmp_path / ("source.json.gz" if compressed else "source.json")
    output = tmp_path / ("output.json.gz" if compressed else "output.json")
    original = fixture_trace()
    track.write_trace(source, original)
    command = [
        sys.executable,
        str(SCRIPTS / "add_layer_track.py"),
        "--trace",
        str(source),
        "--output",
        str(output),
        "--anchor-regex",
        "^anchor$",
        "--num-layers",
        "3",
        "--anchor-offset",
        "0",
        "--passes",
        "2",
        "--phase",
        "target-verify",
        "--evidence",
        "verified fixture",
        "--end-anchor-regex",
        "^terminal$",
    ]
    subprocess.run(command, check=True, capture_output=True)
    assert (
        track.read_trace(output)["traceEvents"][: len(original["traceEvents"])]
        == original["traceEvents"]
    )
    assert track.read_trace(source) == original
    report = json.loads(Path(str(output) + ".layers.json").read_text())
    assert len(report["source_sha256"]) == 64
    assert report["added_event_count"] == 8
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before
    with pytest.raises(FileExistsError):
        track.write_trace(source, {})


def test_local_server_only_exposes_viewer_and_exact_trace(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_bytes(b'{"traceEvents":[]}')
    server = ThreadingHTTPServer(("127.0.0.1", 0), viewer.make_handler(trace))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert urlopen(base + "/trace").read() == trace.read_bytes()
        assert b"PING" in urlopen(base).read()
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../secret")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    page = viewer.viewer_html("</script><script>alert(1)</script>")
    assert "</script><script>alert(1)</script>" not in page
