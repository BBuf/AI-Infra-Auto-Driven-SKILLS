"""Internal overlap helpers for triage-only torch-profiler analysis."""

from __future__ import annotations

import argparse
import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import triage_kernel_helpers as kernel_helpers
from profile_common import (
    coerce_optional_int,
    contains_any_keyword,
    discover_trace_targets,
    extract_trace_events,
    has_stream_marker,
    is_annotation_event,
    is_complete_duration_event,
    is_non_kernel_trace_category,
    is_trace_metadata_name,
    load_server_args,
    load_trace_json,
    looks_like_python_scope_name,
    normalize_repo_relative_path,
    normalize_text,
)
from profile_common import run_profiler as shared_run_profiler
from profile_common import (
    select_heaviest_pid,
)

SOURCE_MAP_SAMPLE_LIMIT_PER_NAME = 16

COMMUNICATION_STRONG_KEYWORDS = (
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "nccl",
    "cross_device_reduce",
    "deepep",
    "a2a",
    "alltoall",
    "allreduce_fusion",
    "mooncake",
)

COMMUNICATION_WEAK_KEYWORDS = (
    "broadcast",
    "dispatch",
    "combine",
)

MEMORY_STRONG_KEYWORDS = (
    "memcpy",
    "memset",
    "dma",
    "prefetch",
)

MEMORY_WEAK_KEYWORDS = (
    "fill",
    "copy",
)

ELEMENTWISE_KEYWORDS = (
    "sigmoid",
    "silu",
    "gelu",
    "relu",
    "softmax",
    "layernorm",
    "rmsnorm",
    "norm",
    "rotary",
    "rope",
    "topk",
    "gate",
    "bias",
    "_cast",
    "index",
    "gather",
    "scatter",
    "masked",
    "elementwise",
    "activation",
)

COMPUTE_KEYWORDS = (
    "cublas",
    "cudnn",
    "cutlass",
    "triton",
    "gemm",
    "gemv",
    "matmul",
    "grouped_mm",
    "flash",
    "attention",
    "fmha",
    "marlin",
    "fused_moe",
    "moe_kernel",
    "groupgemm",
    "mma",
    "wgmma",
    "conv",
    "bmm",
    "mm_kernel",
)

CATEGORY_CHARS = {
    "compute": "#",
    "communication": "=",
    "elementwise": "~",
    "memory": "+",
    "other": "*",
}

LOW_SIGNAL_FUNCTION_TOKENS = (
    "__torch_function__",
    "__torch_dispatch__",
    "__call__",
    "_call_impl",
    "_wrapped_call_impl",
)

LOW_SIGNAL_PATH_TOKENS = (
    "model_executor/parameter.py(",
    "model_executor/parameter.py:",
    "model_executor/cuda_graph_runner.py(",
    "model_executor/cuda_graph_runner.py:",
    "compilation/cuda_graph.py(",
    "compilation/cuda_graph.py:",
    "pyexecutor/cuda_graph_runner.py(",
    "pyexecutor/cuda_graph_runner.py:",
    "pyexecutor/py_executor.py(",
    "pyexecutor/py_executor.py:",
    "_torch/utils.py(",
    "_torch/utils.py:",
    "torch/fx/graph_module.py(",
    "torch/fx/graph_module.py:",
)

CATEGORY_PRIORITY = {
    "compute": 4,
    "communication": 3,
    "memory": 2,
    "elementwise": 1,
    "other": 0,
}

PYTHON_SCOPE_IGNORE_PREFIXES = (
    "threading.py(",
    "selectors.py(",
    "contextlib.py(",
    "queue.py(",
    "logging/",
    "logging/__init__.py(",
    "socket.py(",
    "asyncio/",
    "concurrent/futures/",
    "tqdm/",
    "uvicorn/",
    "fastapi/",
    "starlette/",
    "http/",
    "torch/_ops.py(",
    "torch/nn/modules/module.py(",
    "torch/utils/_contextlib.py(",
    "torch/autograd/",
    "torch/_tensor.py(",
    "torch/distributed/",
    "torch/_dynamo/",
    "torch/_inductor/",
)
KERNEL_NAME_HINTS = (
    COMMUNICATION_STRONG_KEYWORDS
    + COMMUNICATION_WEAK_KEYWORDS
    + MEMORY_STRONG_KEYWORDS
    + MEMORY_WEAK_KEYWORDS
    + COMPUTE_KEYWORDS
)


@dataclass
class KernelEvent:
    idx: int
    name: str
    canonical_name: str
    category: str
    pid: str
    tid: str
    stream: str
    ts: float
    dur: float
    end: float
    stage: str = "all"
    external_id: Optional[int] = None
    correlation: Optional[int] = None
    hidden_us: float = 0.0
    exclusive_us: float = 0.0
    hidden_by_compute_us: float = 0.0
    overlap_with: Counter = field(default_factory=Counter)


@dataclass
class AggregateStats:
    name: str
    category: str
    count: int = 0
    total_us: float = 0.0
    hidden_us: float = 0.0
    exclusive_us: float = 0.0
    hidden_by_compute_us: float = 0.0
    overlap_with: Counter = field(default_factory=Counter)
    representative_idx: Optional[int] = None
    representative_score: float = -1.0

    @property
    def hidden_ratio(self) -> float:
        return self.hidden_us / self.total_us if self.total_us else 0.0

    @property
    def exclusive_ratio(self) -> float:
        return self.exclusive_us / self.total_us if self.total_us else 0.0


@dataclass
class PythonScope:
    name: str
    normalized_name: str
    pid: str
    tid: str
    ts: float
    dur: float
    end: float
    is_meaningful: bool = False
    is_fallback: bool = False


@dataclass
class CPUOpContext:
    external_id: int
    cpu_op_name: str
    pid: str
    tid: str
    ts: float
    dur: float
    end: float
    scope_chain: Tuple[str, ...]


@dataclass
class KernelSourceStats:
    name: str
    total_count: int = 0
    mapped_count: int = 0
    scope_counter: Counter = field(default_factory=Counter)
    chain_counter: Counter = field(default_factory=Counter)
    launch_op_counter: Counter = field(default_factory=Counter)
    site_share_counter: Counter = field(default_factory=Counter)

    @property
    def mapping_ratio(self) -> float:
        return self.mapped_count / self.total_count if self.total_count else 0.0

    @property
    def best_scope(self) -> Optional[str]:
        return self.scope_counter.most_common(1)[0][0] if self.scope_counter else None

    @property
    def best_chain(self) -> Optional[str]:
        return self.chain_counter.most_common(1)[0][0] if self.chain_counter else None

    @property
    def best_launch_op(self) -> Optional[str]:
        return (
            self.launch_op_counter.most_common(1)[0][0]
            if self.launch_op_counter
            else None
        )


@dataclass
class TraceBundle:
    label: str
    trace_path: Path
    server_args: Optional[dict]
    raw_events: Sequence[dict]
    events: List[KernelEvent]
    pid: Optional[str]
    overlap_stats: Optional[Dict[str, float]] = None


@dataclass
class ActionRow:
    priority: str
    verdict: str
    kernel: str
    category: str
    total_us: float
    share_pct: float
    exclusive_ratio: float
    hidden_ratio: float
    python_scope: str
    launch_op: str
    mapping_ratio: float
    dependency_signal: str
    prev_neighbor: str
    next_neighbor: str
    recommendation: str
    suggestion: str
    representative_idx: Optional[int]


def short_name(name: str, max_len: int = 80) -> str:
    name = normalize_text(name)
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def canonicalize_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", name)
    if name.startswith("void ") and name.endswith(")"):
        depth = 0
        split_idx: Optional[int] = None
        for idx in range(len(name) - 1, -1, -1):
            char = name[idx]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
                if depth == 0:
                    split_idx = idx
                    break
        if split_idx is not None:
            name = name[:split_idx]
    return name


def canonicalize_python_scope_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", name)
    match = re.match(r"(?P<path>.+?)\((?P<line>\d+)\): (?P<func>.+)$", name)
    if match:
        path = normalize_repo_relative_path(match.group("path"))
        name = f"{path}({match.group('line')}): {match.group('func')}"
    return name


def canonicalize_cpu_op_name(name: str) -> str:
    return short_name(normalize_text(name), max_len=100)


def classify_kernel(name: str) -> str:
    # This script only needs broad overlap buckets, so keep the precedence small
    # and deterministic: memory/communication first, then compute/elementwise.
    lowered = name.lower()
    looks_compute_like = contains_any_keyword(lowered, COMPUTE_KEYWORDS)
    if contains_any_keyword(lowered, MEMORY_STRONG_KEYWORDS):
        return "memory"
    if contains_any_keyword(lowered, COMMUNICATION_STRONG_KEYWORDS):
        return "communication"
    if contains_any_keyword(lowered, COMPUTE_KEYWORDS):
        return "compute"
    if contains_any_keyword(lowered, ELEMENTWISE_KEYWORDS):
        return "elementwise"
    if contains_any_keyword(lowered, MEMORY_WEAK_KEYWORDS) and not looks_compute_like:
        return "memory"
    if (
        contains_any_keyword(lowered, COMMUNICATION_WEAK_KEYWORDS)
        and not looks_compute_like
    ):
        return "communication"
    if lowered.startswith("void "):
        return "other"
    return "other"


def is_kernel_event(event: dict) -> bool:
    # The overlap helpers prefer a slightly broader kernel detector than the
    # kernel-attribution helpers, but still reject annotations and Python
    # frames up front so the later overlap math only sees real GPU work.
    if not is_complete_duration_event(event):
        return False
    name = normalize_text(event.get("name", ""))
    if is_trace_metadata_name(name):
        return False
    cat = normalize_text(event.get("cat", "")).lower()
    args = event.get("args", {}) or {}
    if is_non_kernel_trace_category(cat):
        return False
    if is_annotation_event(name, cat):
        return False
    if "kernel" in cat or cat.startswith("gpu_"):
        return True
    lowered = name.lower()
    if looks_like_python_scope_name(name):
        return False
    if has_stream_marker(args) and (
        lowered.startswith("void ")
        or lowered.startswith("ampere_")
        or lowered.startswith("sm80_")
        or lowered.startswith("sm90_")
        or contains_any_keyword(lowered, KERNEL_NAME_HINTS)
    ):
        return True
    return False


def is_meaningful_python_scope(name: str) -> bool:
    normalized = canonicalize_python_scope_name(name)
    if not normalized:
        return False
    if normalized.startswith("<built-in method"):
        return False
    if normalized.startswith("nn.Module:"):
        return False
    if any(normalized.startswith(prefix) for prefix in PYTHON_SCOPE_IGNORE_PREFIXES):
        return False
    if normalized.startswith("python/sglang/"):
        return True
    if normalized.startswith("sglang/"):
        return True
    if normalized.startswith("vllm/"):
        return True
    if normalized.startswith("tensorrt_llm/"):
        return True
    if normalized.startswith("sgl_kernel/"):
        return True
    return ".py(" in normalized


def is_fallback_python_scope(name: str) -> bool:
    normalized = canonicalize_python_scope_name(name)
    if (
        not normalized
        or normalized.startswith("<built-in method")
        or normalized.startswith("nn.Module:")
    ):
        return False
    if normalized.startswith("threading.py("):
        return False
    return ".py(" in normalized or normalized.startswith("python/")


def extract_thread_names(events: Sequence[dict]) -> Dict[Tuple[str, str], str]:
    mapping: Dict[Tuple[str, str], str] = {}
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "thread_name":
            continue
        pid = str(event.get("pid"))
        tid = str(event.get("tid"))
        thread_name = str((event.get("args") or {}).get("name", ""))
        if thread_name:
            mapping[(pid, tid)] = thread_name
    return mapping


def build_correlation_external_lookup(raw_events: Sequence[dict]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for event in raw_events:
        args = event.get("args", {}) or {}
        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if correlation is not None and external_id is not None:
            lookup[correlation] = external_id
    return lookup


def extract_kernel_events(
    trace: dict, pid_substring: Optional[str]
) -> Tuple[List[KernelEvent], Optional[str]]:
    # We first build a clean kernel list from the chosen TP rank, then later
    # overlap analysis can stay focused on stream timing instead of trace noise.
    raw_events = extract_trace_events(trace)
    thread_names = extract_thread_names(raw_events)
    correlation_external = build_correlation_external_lookup(raw_events)
    (
        annotations_by_external_id,
        gpu_stage_annotations,
        cpu_stage_annotations,
    ) = kernel_helpers.build_stage_annotations(raw_events)
    chosen_pid = select_heaviest_pid(
        raw_events,
        is_kernel_event,
        pid_substring=pid_substring,
        preferred_substrings=(() if pid_substring else ("TP00",)),
    )
    kernel_events: List[KernelEvent] = []
    if chosen_pid is None:
        return kernel_events, None

    idx = 0
    for event in raw_events:
        if not is_kernel_event(event):
            continue
        pid = str(event.get("pid"))
        if pid != chosen_pid:
            continue
        tid = str(event.get("tid"))
        args = event.get("args", {}) or {}
        stream = (
            args.get("stream")
            or args.get("cuda_stream")
            or thread_names.get((pid, tid))
            or f"tid={tid}"
        )
        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if external_id is None and correlation is not None:
            external_id = correlation_external.get(correlation)
        name = str(event["name"])
        dur = float(event["dur"])
        ts = float(event["ts"])
        kernel_events.append(
            KernelEvent(
                idx=idx,
                name=name,
                canonical_name=canonicalize_name(name),
                category=classify_kernel(name),
                stage=kernel_helpers.resolve_kernel_stage(
                    kernel_ts=ts,
                    external_id=external_id,
                    annotations_by_external_id=annotations_by_external_id,
                    gpu_annotations=gpu_stage_annotations,
                    cpu_annotations=cpu_stage_annotations,
                ),
                pid=pid,
                tid=tid,
                stream=str(stream),
                ts=ts,
                dur=dur,
                end=ts + dur,
                external_id=external_id,
                correlation=correlation,
            )
        )
        idx += 1
    return kernel_events, chosen_pid


def group_events_by_stage(
    events: Sequence[KernelEvent], default_stage: str
) -> Dict[str, List[KernelEvent]]:
    grouped: Dict[str, List[KernelEvent]] = defaultdict(list)
    for event in events:
        stage = default_stage if default_stage != "all" else (event.stage or "all")
        grouped[stage].append(event)
    return dict(grouped)


def dominant_overlap_name(
    event: KernelEvent, active_events: Iterable[KernelEvent]
) -> Optional[str]:
    candidates = [
        other
        for other in active_events
        if other.idx != event.idx and other.stream != event.stream
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda other: (CATEGORY_PRIORITY.get(other.category, 0), other.dur),
        reverse=True,
    )
    return candidates[0].canonical_name


def analyze_overlap(events: Sequence[KernelEvent]) -> Dict[str, float]:
    # Sweep line over kernel start/end points. For each active time slice we
    # decide whether a kernel was exposed on the critical path or hidden by work
    # on other streams.
    points: List[Tuple[float, int, int]] = []
    event_map = {event.idx: event for event in events}
    for event in events:
        points.append((event.ts, 1, event.idx))
        points.append((event.end, 0, event.idx))
    points.sort(key=lambda item: (item[0], item[1]))

    total_busy = 0.0
    total_overlap = 0.0
    max_concurrent = 0
    active: Dict[int, KernelEvent] = {}
    prev_time: Optional[float] = None

    for time_point, is_start, event_idx in points:
        if prev_time is not None and time_point > prev_time and active:
            segment = time_point - prev_time
            active_events = list(active.values())
            distinct_streams = {event.stream for event in active_events}
            total_busy += segment
            max_concurrent = max(max_concurrent, len(distinct_streams))
            if len(distinct_streams) >= 2:
                total_overlap += segment
            for event in active_events:
                overlapping_events = [
                    other
                    for other in active_events
                    if other.idx != event.idx and other.stream != event.stream
                ]
                if overlapping_events:
                    event.hidden_us += segment
                    if any(other.category == "compute" for other in overlapping_events):
                        event.hidden_by_compute_us += segment
                    overlap_name = dominant_overlap_name(event, active_events)
                    if overlap_name:
                        event.overlap_with[overlap_name] += segment
                else:
                    event.exclusive_us += segment

        if is_start == 0:
            active.pop(event_idx, None)
        else:
            active[event_idx] = event_map[event_idx]
        prev_time = time_point

    return {
        "total_busy_us": total_busy,
        "total_overlap_us": total_overlap,
        "max_concurrent_streams": float(max_concurrent),
    }


def aggregate_events(
    events: Sequence[KernelEvent],
) -> Dict[Tuple[str, str], AggregateStats]:
    aggregates: Dict[Tuple[str, str], AggregateStats] = {}
    for event in events:
        key = (event.canonical_name, event.category)
        if key not in aggregates:
            aggregates[key] = AggregateStats(
                name=event.canonical_name, category=event.category
            )
        stats = aggregates[key]
        stats.count += 1
        stats.total_us += event.dur
        stats.hidden_us += event.hidden_us
        stats.exclusive_us += event.exclusive_us
        stats.hidden_by_compute_us += event.hidden_by_compute_us
        stats.overlap_with.update(event.overlap_with)
        score = event.hidden_us + event.exclusive_us
        if score > stats.representative_score:
            stats.representative_score = score
            stats.representative_idx = event.idx
    return aggregates


def top_hidden_low_roi(
    aggregates: Dict[Tuple[str, str], AggregateStats],
) -> List[AggregateStats]:
    candidates = [
        stats
        for stats in aggregates.values()
        if stats.category in {"elementwise", "memory"}
        and stats.total_us >= 5.0
        and stats.hidden_ratio >= 0.65
    ]
    candidates.sort(
        key=lambda stats: (
            stats.hidden_us
            * (1.0 + stats.hidden_by_compute_us / max(stats.hidden_us, 1.0)),
            stats.hidden_ratio,
        ),
        reverse=True,
    )
    return candidates[:5]


def top_overlap_opportunities(
    aggregates: Dict[Tuple[str, str], AggregateStats],
) -> List[AggregateStats]:
    category_weight = {
        "communication": 1.3,
        "memory": 1.15,
        "elementwise": 1.0,
        "compute": 0.35,
        "other": 0.8,
    }
    candidates = [
        stats
        for stats in aggregates.values()
        if stats.total_us >= 5.0 and stats.exclusive_ratio >= 0.45
    ]
    primary = [stats for stats in candidates if stats.category != "compute"]
    fallback = [stats for stats in candidates if stats.category == "compute"]
    primary.sort(
        key=lambda stats: stats.exclusive_us * category_weight.get(stats.category, 1.0),
        reverse=True,
    )
    fallback.sort(
        key=lambda stats: stats.exclusive_us * category_weight.get(stats.category, 1.0),
        reverse=True,
    )
    return (primary + fallback)[:5]


def choose_window_events(
    events: Sequence[KernelEvent],
    representative_idx: int,
    window_us: Optional[float],
) -> Tuple[float, float, List[KernelEvent]]:
    center = next(event for event in events if event.idx == representative_idx)
    span = window_us if window_us is not None else max(40.0, center.dur * 6.0)
    start = max(0.0, center.ts - span * 0.35)
    end = center.end + span * 0.65
    window_events = [
        event for event in events if event.end >= start and event.ts <= end
    ]
    return start, end, window_events


def render_ascii_timeline(
    events: Sequence[KernelEvent],
    representative_idx: int,
    window_us: Optional[float],
    width: int,
) -> str:
    start, end, window_events = choose_window_events(
        events, representative_idx, window_us
    )
    if not window_events:
        return "No events found in the selected window."

    streams = sorted(
        {event.stream for event in window_events}, key=lambda item: (len(item), item)
    )
    symbol_map: Dict[int, str] = {}
    legend_events = sorted(window_events, key=lambda event: event.dur, reverse=True)[:8]
    symbol_alphabet = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    for index, event in enumerate(legend_events):
        symbol_map[event.idx] = symbol_alphabet[index]

    label_width = max(len(stream) for stream in streams)
    lines = []
    marker_positions = [0, width // 4, width // 2, (3 * width) // 4, width - 1]
    header = [" "] * width
    for position in marker_positions:
        header[position] = "|"
    lines.append("time(us) " + "".join(header))

    time_line = [" "] * width
    markers = [
        start,
        start + (end - start) * 0.25,
        start + (end - start) * 0.5,
        start + (end - start) * 0.75,
        end,
    ]
    for position, value in zip(marker_positions, markers):
        text = f"{value:.1f}"
        begin = min(max(position - len(text) // 2, 0), max(0, width - len(text)))
        for offset, char in enumerate(text):
            time_line[begin + offset] = char
    lines.append("         " + "".join(time_line))

    for stream in streams:
        row = ["."] * width
        row_events = [event for event in window_events if event.stream == stream]
        for event in row_events:
            char = symbol_map.get(event.idx, CATEGORY_CHARS[event.category])
            left = int((event.ts - start) / max(end - start, 1.0) * (width - 1))
            right = int(
                math.ceil((event.end - start) / max(end - start, 1.0) * (width - 1))
            )
            right = max(left + 1, min(right, width - 1))
            for pos in range(max(0, left), min(width, right + 1)):
                row[pos] = char
        lines.append(f"{stream:<{label_width}} " + "".join(row))

    if legend_events:
        lines.append("legend:")
        for event in legend_events:
            symbol = symbol_map[event.idx]
            lines.append(
                f"  {symbol} [{event.category[:4]}] {short_name(event.canonical_name, 72)} ({event.dur:.1f} us)"
            )
    return "\n".join(lines)


def choose_best_scope(scope_chain: Sequence[str]) -> Optional[str]:
    ranked: List[Tuple[float, str]] = []
    for index, scope in enumerate(scope_chain):
        score = float(index)
        if scope.startswith("python/sglang/"):
            score += 50.0
        elif scope.startswith("sglang/"):
            score += 48.0
        elif scope.startswith("vllm/"):
            score += 46.0
        elif scope.startswith("tensorrt_llm/"):
            score += 44.0
        elif scope.startswith("sgl_kernel/"):
            score += 30.0
        elif ".py(" in scope:
            score += 10.0
        if "utils.py" in scope and "__call__" in scope:
            score -= 15.0
        if "scheduler_profiler_mixin.py" in scope:
            score -= 20.0
        if is_low_signal_scope(scope):
            score -= 25.0
        ranked.append((score, scope))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def is_low_signal_scope(scope: str) -> bool:
    lowered = canonicalize_python_scope_name(scope).lower()
    if not lowered:
        return False
    return any(token in lowered for token in LOW_SIGNAL_FUNCTION_TOKENS) or any(
        token in lowered for token in LOW_SIGNAL_PATH_TOKENS
    )


def scope_chain_key(scope_chain: Sequence[str]) -> Optional[str]:
    if not scope_chain:
        return None
    trimmed = list(scope_chain[-4:])
    return " -> ".join(trimmed)


def normalize_match_text(text: object) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", normalize_text(text)).lower()


def source_scope_priority(scope: Optional[str]) -> int:
    normalized = canonicalize_python_scope_name(scope or "")
    if not normalized or normalized == "unmapped":
        return 0
    penalty = 80 if is_low_signal_scope(normalized) else 0
    if normalized.startswith("python/sglang/"):
        return 300 - penalty
    if normalized.startswith("sglang/"):
        return 290 - penalty
    if normalized.startswith("vllm/"):
        return 285 - penalty
    if normalized.startswith("tensorrt_llm/"):
        return 280 - penalty
    if normalized.startswith("sgl_kernel/"):
        return 260 - penalty
    if ".py(" in normalized:
        return 120 - penalty
    return 0


def kernel_alias_token_groups(kernel_name: str) -> List[Tuple[str, ...]]:
    lowered = normalize_match_text(kernel_name)
    groups: List[Tuple[str, ...]] = []
    if "flashattnfwdcombine" in lowered:
        groups.append(
            (
                "flashattnfwdsm90",
                "flashattnvarlenfunc",
                "vllmflashattnflashattninterface",
                "vllmfa3cfwd",
            )
        )
    if "kernelmha" in lowered:
        groups.append(
            (
                "maskedmultiheadattentionkernel",
                "attentioninplace",
                "attentionbackendtrtllm",
            )
        )
    if "applybiasropeupdatekvcachev2" in lowered:
        groups.append(
            (
                "fusedqknormropekernel",
                "applyqknormrope",
                "modelingqwen3py98applyqknormrope",
            )
        )
    if lowered.startswith("memset"):
        groups.append(("memset",))
    return groups


def source_stats_lookup_text(
    kernel_name: str, stats: Optional[KernelSourceStats]
) -> str:
    parts = [kernel_name]
    if stats:
        parts.append(str(stats.best_scope or ""))
        parts.append(str(stats.best_chain or ""))
        parts.append(str(stats.best_launch_op or ""))
    return normalize_match_text(" ".join(parts))


def relaxed_source_stats_lookup(
    source_map: Dict[str, KernelSourceStats], kernel_name: str
) -> Optional[KernelSourceStats]:
    if kernel_name in source_map:
        return source_map[kernel_name]

    lowered = kernel_name.lower()
    best_key = None
    best_score = -1
    for candidate_key in source_map:
        candidate_lowered = candidate_key.lower()
        if candidate_lowered.startswith(lowered) or lowered.startswith(
            candidate_lowered
        ):
            score = min(len(candidate_lowered), len(lowered))
        elif candidate_lowered in lowered or lowered in candidate_lowered:
            score = min(len(candidate_lowered), len(lowered)) // 2
        else:
            continue
        if score > best_score:
            best_key = candidate_key
            best_score = score
    if best_key:
        return source_map.get(best_key)

    lowered_compact = normalize_match_text(kernel_name)
    if len(lowered_compact) >= 96:

        def common_prefix_len(left: str, right: str) -> int:
            count = 0
            for left_ch, right_ch in zip(left, right):
                if left_ch != right_ch:
                    break
                count += 1
            return count

        best_key = None
        best_score = -1
        for candidate_key in source_map:
            candidate_compact = normalize_match_text(candidate_key)
            if len(candidate_compact) < 96:
                continue
            prefix_len = common_prefix_len(lowered_compact, candidate_compact)
            shorter_len = min(len(lowered_compact), len(candidate_compact))
            if prefix_len < 64 or prefix_len < int(shorter_len * 0.4):
                continue
            score = prefix_len
            if lowered_compact.startswith(
                "voidcutlassdevicekernelflash"
            ) and candidate_compact.startswith("voidcutlassdevicekernelflash"):
                score += 32
            if score > best_score:
                best_key = candidate_key
                best_score = score
        if best_key:
            return source_map.get(best_key)

    alias_groups = kernel_alias_token_groups(kernel_name)
    if not alias_groups:
        return None
    best_key = None
    best_score = -1
    for candidate_key, stats in source_map.items():
        candidate_text = source_stats_lookup_text(candidate_key, stats)
        score = 0
        for group_index, group in enumerate(alias_groups):
            group_score = max(
                (len(token) for token in group if token in candidate_text),
                default=0,
            )
            if group_score:
                score += 1000 * (group_index + 1) + group_score
        if score <= 0:
            continue
        score += source_scope_priority(stats.best_scope)
        score += int(stats.mapping_ratio * 100)
        if score > best_score:
            best_key = candidate_key
            best_score = score
    return source_map.get(best_key) if best_key else None


def extract_cpu_launch_contexts(
    raw_events: Sequence[dict],
    target_external_ids: Optional[set[int]] = None,
) -> Dict[int, List[CPUOpContext]]:
    # Rebuild `External id -> CPU op -> active Python scopes` only for the
    # small set of launch ids that the source-map step will actually consume.
    # vLLM eager traces can have millions of Python frames on one thread, so
    # avoid global timeline reconstruction across unrelated threads and ids.
    cpu_ops_by_thread: Dict[Tuple[str, str], List[CPUOpContext]] = defaultdict(list)

    for event in raw_events:
        if not is_complete_duration_event(event):
            continue
        if str(event.get("cat", "")) != "cpu_op":
            continue
        args = event.get("args", {}) or {}
        external_id = coerce_optional_int(args.get("External id"))
        if external_id is None:
            continue
        if target_external_ids is not None and external_id not in target_external_ids:
            continue
        pid = str(event.get("pid"))
        tid = str(event.get("tid"))
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        cpu_ops_by_thread[(pid, tid)].append(
            CPUOpContext(
                external_id=external_id,
                cpu_op_name=str(event.get("name", "")),
                pid=pid,
                tid=tid,
                ts=ts,
                dur=dur,
                end=ts + dur,
                scope_chain=(),
            )
        )

    if not cpu_ops_by_thread:
        return {}

    scopes_by_thread: Dict[Tuple[str, str], List[PythonScope]] = defaultdict(list)
    relevant_threads = set(cpu_ops_by_thread)
    for event in raw_events:
        if not is_complete_duration_event(event):
            continue
        if str(event.get("cat", "")) != "python_function":
            continue
        pid = str(event.get("pid"))
        tid = str(event.get("tid"))
        thread_key = (pid, tid)
        if thread_key not in relevant_threads:
            continue
        normalized_name = canonicalize_python_scope_name(event.get("name", ""))
        is_meaningful = is_meaningful_python_scope(normalized_name)
        is_fallback = is_fallback_python_scope(normalized_name)
        if not is_meaningful and not is_fallback:
            continue
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        scopes_by_thread[thread_key].append(
            PythonScope(
                name=str(event.get("name", "")),
                normalized_name=normalized_name,
                pid=pid,
                tid=tid,
                ts=ts,
                dur=dur,
                end=ts + dur,
                is_meaningful=is_meaningful,
                is_fallback=is_fallback,
            )
        )

    contexts_by_external_id: Dict[int, List[CPUOpContext]] = defaultdict(list)
    for thread_key in relevant_threads:
        scopes = scopes_by_thread.get(thread_key, [])
        cpu_ops = cpu_ops_by_thread.get(thread_key, [])
        timeline = []
        for scope_idx, scope in enumerate(scopes):
            timeline.append((scope.ts, 0, scope_idx))
            timeline.append((scope.end, 2, scope_idx))
        for cpu_op_idx, cpu_op in enumerate(cpu_ops):
            timeline.append((cpu_op.ts, 1, cpu_op_idx))
        timeline.sort(key=lambda item: (item[0], item[1]))

        active_scopes: Dict[int, PythonScope] = {}
        for _, kind, payload in timeline:
            if kind == 0:
                active_scopes[payload] = scopes[payload]
            elif kind == 1:
                meaningful = [
                    scope.normalized_name
                    for scope in active_scopes.values()
                    if scope.is_meaningful
                ]
                fallback = (
                    []
                    if meaningful
                    else [
                        scope.normalized_name
                        for scope in active_scopes.values()
                        if scope.is_fallback
                    ]
                )
                chosen_chain = tuple((meaningful or fallback)[-6:])
                cpu_op = cpu_ops[payload]
                contexts_by_external_id[cpu_op.external_id].append(
                    CPUOpContext(
                        external_id=cpu_op.external_id,
                        cpu_op_name=cpu_op.cpu_op_name,
                        pid=cpu_op.pid,
                        tid=cpu_op.tid,
                        ts=cpu_op.ts,
                        dur=cpu_op.dur,
                        end=cpu_op.end,
                        scope_chain=chosen_chain,
                    )
                )
            else:
                active_scopes.pop(payload, None)
    return contexts_by_external_id


def build_correlation_external_lookup(raw_events: Sequence[dict]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for event in raw_events:
        args = event.get("args", {}) or {}
        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if correlation is not None and external_id is not None:
            lookup[correlation] = external_id
    return lookup


def is_cuda_launch_event(name: str, cat: str) -> bool:
    lowered_name = normalize_text(name).lower()
    lowered_cat = normalize_text(cat).lower()
    if lowered_cat == "cuda_runtime":
        return lowered_name in {
            "cudaLaunchKernel",
            "cudaLaunchKernelExC",
        }
    return lowered_name in {
        "cuLaunchKernel",
        "cuLaunchKernelEx",
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
    }


@dataclass
class LaunchContext:
    correlation: int
    pid: str
    tid: str
    ts: float
    dur: float
    end: float
    launch_name: str


def build_launch_contexts(
    raw_events: Sequence[dict],
) -> Dict[int, List[LaunchContext]]:
    output: Dict[int, List[LaunchContext]] = defaultdict(list)
    for event in raw_events:
        if not is_complete_duration_event(event):
            continue
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        args = event.get("args", {}) or {}
        correlation = coerce_optional_int(args.get("correlation"))
        if correlation is None or not is_cuda_launch_event(name, cat):
            continue
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        output[correlation].append(
            LaunchContext(
                correlation=correlation,
                pid=str(event.get("pid")),
                tid=str(event.get("tid")),
                ts=ts,
                dur=dur,
                end=ts + dur,
                launch_name=name,
            )
        )
    for items in output.values():
        items.sort(key=lambda item: item.ts)
    return output


def choose_launch_context(
    contexts: Sequence[LaunchContext], kernel_ts: float
) -> Optional[LaunchContext]:
    if not contexts:
        return None
    return min(contexts, key=lambda context: (abs(context.ts - kernel_ts), context.dur))


def choose_cpu_context(
    contexts: Sequence[CPUOpContext], kernel_ts: float
) -> Optional[CPUOpContext]:
    if not contexts:
        return None
    return min(contexts, key=lambda context: (abs(context.ts - kernel_ts), context.dur))


def extract_meaningful_python_scopes(raw_events: Sequence[dict]) -> List[PythonScope]:
    scopes: List[PythonScope] = []
    for event in raw_events:
        if not is_complete_duration_event(event):
            continue
        if str(event.get("cat", "")) != "python_function":
            continue
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        normalized_name = canonicalize_python_scope_name(event.get("name", ""))
        if not is_meaningful_python_scope(normalized_name):
            continue
        scopes.append(
            PythonScope(
                name=str(event.get("name", "")),
                normalized_name=normalized_name,
                pid=str(event.get("pid")),
                tid=str(event.get("tid")),
                ts=ts,
                dur=dur,
                end=ts + dur,
            )
        )
    return scopes


def choose_temporal_scope_chain(
    scopes: Sequence[PythonScope], kernel_ts: float
) -> Tuple[str, ...]:
    matches = [scope for scope in scopes if scope.ts <= kernel_ts <= scope.end]
    if not matches:
        return ()
    matches.sort(key=lambda scope: (scope.ts, -scope.dur, scope.normalized_name))
    chain = []
    seen = set()
    for scope in matches:
        if scope.normalized_name in seen:
            continue
        seen.add(scope.normalized_name)
        chain.append(scope.normalized_name)
    return tuple(chain[-6:])


def build_temporal_scope_lookup(
    scopes: Sequence[PythonScope],
    query_points: Sequence[Tuple[int, float]],
) -> Dict[int, Tuple[str, ...]]:
    if not scopes or not query_points:
        return {}

    timeline: List[Tuple[float, int, object]] = []
    for scope in scopes:
        timeline.append((scope.ts, 0, scope))
        timeline.append((scope.end, 2, scope))
    for event_idx, probe_ts in query_points:
        timeline.append((probe_ts, 1, event_idx))
    timeline.sort(key=lambda item: (item[0], item[1]))

    active_scopes: List[PythonScope] = []
    resolved: Dict[int, Tuple[str, ...]] = {}
    for _, kind, payload in timeline:
        if kind == 0:
            active_scopes.append(payload)
            continue
        if kind == 2:
            if payload in active_scopes:
                active_scopes.remove(payload)
            continue

        chain: List[str] = []
        seen: set[str] = set()
        for scope in sorted(
            active_scopes,
            key=lambda scope: (scope.ts, -scope.dur, scope.normalized_name),
        ):
            name = scope.normalized_name
            if name in seen:
                continue
            seen.add(name)
            chain.append(name)
        resolved[payload] = tuple(chain[-6:])
    return resolved


def build_temporal_scope_lookup_from_raw_events(
    raw_events: Sequence[dict],
    query_points: Sequence[Tuple[int, float]],
) -> Dict[int, Tuple[str, ...]]:
    if not query_points:
        return {}

    ordered_queries = sorted(
        ((float(query_ts), int(query_id)) for query_id, query_ts in query_points),
        key=lambda item: item[0],
    )
    query_times = [query_ts for query_ts, _ in ordered_queries]
    query_ids = [query_id for _, query_id in ordered_queries]
    first_query_ts = query_times[0]
    last_query_ts = query_times[-1]

    matches_by_query: Dict[int, List[PythonScope]] = defaultdict(list)
    for event in raw_events:
        if not is_complete_duration_event(event):
            continue
        if str(event.get("cat", "")) != "python_function":
            continue
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        end = ts + dur
        if end < first_query_ts or ts > last_query_ts:
            continue

        normalized_name = canonicalize_python_scope_name(event.get("name", ""))
        if not is_meaningful_python_scope(normalized_name):
            continue

        left = bisect_left(query_times, ts - 1e-3)
        right = bisect_right(query_times, end + 1e-3)
        if left >= right:
            continue

        scope = PythonScope(
            name=str(event.get("name", "")),
            normalized_name=normalized_name,
            pid=str(event.get("pid")),
            tid=str(event.get("tid")),
            ts=ts,
            dur=dur,
            end=end,
            is_meaningful=True,
            is_fallback=False,
        )
        for pos in range(left, right):
            matches_by_query[query_ids[pos]].append(scope)

    resolved: Dict[int, Tuple[str, ...]] = {}
    for query_id, scopes in matches_by_query.items():
        chain: List[str] = []
        seen: set[str] = set()
        for scope in sorted(
            scopes,
            key=lambda scope: (scope.ts, -scope.dur, scope.normalized_name),
        ):
            name = scope.normalized_name
            if name in seen:
                continue
            seen.add(name)
            chain.append(name)
        resolved[query_id] = tuple(chain[-6:])
    return resolved


def build_kernel_source_map(
    mapping_bundle: TraceBundle,
    kernel_map_entry_lookup=None,
    stage: str = "all",
) -> Dict[str, KernelSourceStats]:
    sampled_events = sample_source_map_events(mapping_bundle.events)
    target_external_ids = {
        event.external_id for event in sampled_events if event.external_id is not None
    }
    contexts_by_external_id = extract_cpu_launch_contexts(
        mapping_bundle.raw_events,
        target_external_ids=target_external_ids or None,
    )
    correlation_external = build_correlation_external_lookup(mapping_bundle.raw_events)
    launch_contexts_by_correlation = build_launch_contexts(mapping_bundle.raw_events)
    fallback_queries = [
        (event.idx, event.ts)
        for event in sampled_events
        if event.external_id is None
        or not contexts_by_external_id.get(event.external_id)
    ]
    temporal_scope_lookup = build_temporal_scope_lookup_from_raw_events(
        mapping_bundle.raw_events,
        fallback_queries,
    )
    source_map: Dict[str, KernelSourceStats] = {}
    for event in sampled_events:
        stats = source_map.setdefault(
            event.canonical_name, KernelSourceStats(name=event.canonical_name)
        )
        stats.total_count += 1
        kernel_entry = (
            kernel_map_entry_lookup(stage, event.canonical_name)
            if kernel_map_entry_lookup is not None
            else None
        )
        cpu_context = None
        effective_external_id = event.external_id
        if effective_external_id is None and event.correlation is not None:
            effective_external_id = correlation_external.get(event.correlation)
        if effective_external_id is not None:
            cpu_context = choose_cpu_context(
                contexts_by_external_id.get(effective_external_id, []), event.ts
            )

        launch_op = None
        scope_chain: Tuple[str, ...] = ()
        if cpu_context is not None:
            launch_op = canonicalize_cpu_op_name(cpu_context.cpu_op_name)
            scope_chain = cpu_context.scope_chain
        else:
            launch_context = (
                choose_launch_context(
                    launch_contexts_by_correlation.get(event.correlation, []), event.ts
                )
                if event.correlation is not None
                else None
            )
            if launch_context is not None:
                scope_chain = build_temporal_scope_lookup_from_raw_events(
                    mapping_bundle.raw_events,
                    [(event.idx, launch_context.ts)],
                ).get(event.idx, ())
                if scope_chain:
                    launch_op = canonicalize_cpu_op_name(launch_context.launch_name)
            if not scope_chain:
                scope_chain = temporal_scope_lookup.get(event.idx, ())
            if scope_chain:
                launch_op = "time-window fallback"

        if not scope_chain:
            if kernel_entry:
                best_location = str(kernel_entry.get("best_location") or "").strip()
                if best_location and best_location != "unresolved":
                    stats.mapped_count += 1
                    stats.scope_counter[best_location] += 1
                    stats.site_share_counter[best_location] += 1
                    for site in kernel_entry.get("sites") or []:
                        display_location = str(
                            site.get("display_location") or site.get("location") or ""
                        ).strip()
                        if display_location and display_location != "unresolved":
                            launches = int(site.get("launches") or 0)
                            stats.site_share_counter[display_location] += max(
                                1, launches
                            )
                            if launches > 0:
                                stats.scope_counter[display_location] += launches
                        top_cpu_op = site.get("top_cpu_op")
                        if top_cpu_op:
                            launches = int(site.get("launches") or 0)
                            stats.launch_op_counter[str(top_cpu_op)] += max(1, launches)
            continue

        stats.mapped_count += 1
        best_scope = choose_best_scope(scope_chain)
        if best_scope:
            stats.scope_counter[best_scope] += 1
            stats.site_share_counter[best_scope] += 1
        chain = scope_chain_key(scope_chain)
        if chain:
            stats.chain_counter[chain] += 1
        if launch_op:
            stats.launch_op_counter[launch_op] += 1
    return source_map


def merge_source_map_from_kernel_payload(
    source_map: Dict[str, KernelSourceStats],
    stage_payload: Optional[dict],
) -> Dict[str, KernelSourceStats]:
    if not stage_payload:
        return source_map

    for kernel_name, entry in (stage_payload.get("kernels") or {}).items():
        sites = entry.get("sites") or []
        best_location = str(entry.get("best_location") or "").strip()
        if not sites and (not best_location or best_location == "unresolved"):
            continue

        stats = source_map.setdefault(kernel_name, KernelSourceStats(name=kernel_name))
        if sites:
            for site in sites:
                location = str(site.get("location") or best_location or "").strip()
                launches = max(1, int(site.get("launches") or 0))
                stats.total_count += launches
                if location and location != "unresolved":
                    stats.mapped_count += launches
                    stats.scope_counter[location] += launches
                    stats.site_share_counter[location] += launches
                top_cpu_op = str(site.get("top_cpu_op") or "").strip()
                if top_cpu_op:
                    stats.launch_op_counter[top_cpu_op] += launches
                stack = str(site.get("stack") or "").strip()
                if stack:
                    stats.chain_counter[stack] += launches
            continue

        stats.total_count += 1
        stats.mapped_count += 1
        stats.scope_counter[best_location] += 1
        stats.site_share_counter[best_location] += 1
    return source_map


def sample_source_map_events(
    events: Sequence[KernelEvent],
    per_name_limit: int = SOURCE_MAP_SAMPLE_LIMIT_PER_NAME,
) -> List[KernelEvent]:
    if per_name_limit <= 0:
        return list(events)

    grouped: Dict[str, List[KernelEvent]] = defaultdict(list)
    for event in events:
        grouped[event.canonical_name].append(event)

    sampled: List[KernelEvent] = []
    for kernel_name in sorted(grouped):
        items = grouped[kernel_name]
        if len(items) <= per_name_limit:
            sampled.extend(items)
            continue
        for sample_idx in range(per_name_limit):
            pos = round(sample_idx * (len(items) - 1) / (per_name_limit - 1))
            sampled.append(items[pos])
    sampled.sort(key=lambda event: (event.ts, event.idx))
    return sampled


def format_overlap_counter(counter: Counter, limit: int = 2) -> str:
    if not counter:
        return "n/a"
    parts = []
    for name, duration in counter.most_common(limit):
        parts.append(f"{short_name(name, 48)} ({duration:.1f} us)")
    return ", ".join(parts)


def build_headroom_suggestion(stats: AggregateStats) -> str:
    if stats.category == "communication":
        return "Communication is still exposed. Check whether it can overlap with nearby compute."
    if stats.category in {"elementwise", "memory"}:
        return "This path is still exposed. Consider fusion or moving it under nearby compute."
    return "Exposed time remains. Check stream placement and nearby dependencies."


def build_hidden_suggestion(stats: AggregateStats) -> str:
    overlap = format_overlap_counter(stats.overlap_with, limit=1)
    if overlap != "n/a":
        return f"Mostly hidden under {overlap}. Tune it only if schedule or fusion changes."
    return "Mostly hidden already. Revisit it only if schedule or fusion changes."


def build_other_suggestion(stats: AggregateStats) -> str:
    if stats.exclusive_ratio >= 0.6:
        return (
            "Exposed time remains, but it was not one of the leading overlap targets."
        )
    if stats.hidden_ratio >= 0.6:
        return "Often hidden already. Revisit it if launch count or schedule changes."
    return "Mixed exposure and overlap. Inspect it after the higher-share rows above."


def parse_scope_signature(scope: str) -> Tuple[str, str]:
    if not scope or scope in {"unmapped", "n/a"}:
        return "", ""
    match = re.match(r"(.+?)\(\d+\):\s*(.+)$", scope)
    if match:
        return match.group(1), match.group(2)
    return scope, ""


def same_scope_family(left: str, right: str) -> bool:
    left_path, left_func = parse_scope_signature(left)
    right_path, right_func = parse_scope_signature(right)
    if not left_path or not right_path:
        return False
    if left_path == right_path:
        return True
    return bool(left_func and right_func and left_func == right_func)


def is_neighbor_dependency_like(
    current: KernelEvent, neighbor: Optional[KernelEvent]
) -> bool:
    if neighbor is None:
        return False
    if current.category == "communication":
        return neighbor.category in {"compute", "elementwise", "memory", "other"}
    if current.category in {"elementwise", "memory"}:
        return neighbor.category in {
            "compute",
            "communication",
            "elementwise",
            "memory",
        }
    return False


def build_stream_neighbor_index(
    events: Sequence[KernelEvent],
) -> Dict[int, Tuple[Optional[KernelEvent], Optional[KernelEvent]]]:
    by_stream: Dict[str, List[KernelEvent]] = defaultdict(list)
    for event in events:
        by_stream[event.stream].append(event)

    index: Dict[int, Tuple[Optional[KernelEvent], Optional[KernelEvent]]] = {}
    for stream_events in by_stream.values():
        stream_events.sort(key=lambda event: (event.ts, event.end, event.idx))
        for pos, event in enumerate(stream_events):
            prev_event = stream_events[pos - 1] if pos > 0 else None
            next_event = (
                stream_events[pos + 1] if pos + 1 < len(stream_events) else None
            )
            index[event.idx] = (prev_event, next_event)
    return index


def describe_neighbor(
    neighbor: Optional[KernelEvent],
    gap_us: Optional[float],
    source_map: Dict[str, KernelSourceStats],
) -> str:
    if neighbor is None:
        return "none"
    source = relaxed_source_stats_lookup(source_map, neighbor.canonical_name)
    scope = source.best_scope if source and source.best_scope else "unmapped"
    if gap_us is not None:
        gap_us = max(gap_us, 0.0)
        gap_text = f"{gap_us:.1f} us"
    else:
        gap_text = "n/a"
    return (
        f"{short_name(neighbor.canonical_name, 28)} "
        f"@ {short_name(scope, 28)} "
        f"(gap {gap_text})"
    )


def classify_dependency_signal(
    current: KernelEvent,
    source: Optional[KernelSourceStats],
    prev_event: Optional[KernelEvent],
    next_event: Optional[KernelEvent],
    source_map: Dict[str, KernelSourceStats],
) -> Tuple[str, str, str]:
    current_scope = source.best_scope if source and source.best_scope else "unmapped"
    current_launch = (
        source.best_launch_op if source and source.best_launch_op else "n/a"
    )

    prev_gap = current.ts - prev_event.end if prev_event is not None else None
    next_gap = next_event.ts - current.end if next_event is not None else None
    prev_source = (
        relaxed_source_stats_lookup(source_map, prev_event.canonical_name)
        if prev_event is not None
        else None
    )
    next_source = (
        relaxed_source_stats_lookup(source_map, next_event.canonical_name)
        if next_event is not None
        else None
    )
    prev_scope = (
        prev_source.best_scope if prev_source and prev_source.best_scope else "unmapped"
    )
    next_scope = (
        next_source.best_scope if next_source and next_source.best_scope else "unmapped"
    )
    prev_launch = (
        prev_source.best_launch_op
        if prev_source and prev_source.best_launch_op
        else "n/a"
    )
    next_launch = (
        next_source.best_launch_op
        if next_source and next_source.best_launch_op
        else "n/a"
    )

    if prev_gap is not None:
        prev_gap = max(prev_gap, 0.0)
    if next_gap is not None:
        next_gap = max(next_gap, 0.0)

    tight_gap_threshold = max(2.0, min(20.0, current.dur * 0.15))
    prev_tight = prev_gap is not None and prev_gap <= tight_gap_threshold
    next_tight = next_gap is not None and next_gap <= tight_gap_threshold

    prev_risk = prev_tight and (
        same_scope_family(current_scope, prev_scope)
        or (current_launch != "n/a" and current_launch == prev_launch)
        or is_neighbor_dependency_like(current, prev_event)
    )
    next_risk = next_tight and (
        same_scope_family(current_scope, next_scope)
        or (current_launch != "n/a" and current_launch == next_launch)
        or is_neighbor_dependency_like(current, next_event)
    )

    prev_unclear = (
        prev_tight
        and not prev_risk
        and (current_scope == "unmapped" or prev_scope == "unmapped")
    )
    next_unclear = (
        next_tight
        and not next_risk
        and (current_scope == "unmapped" or next_scope == "unmapped")
    )

    if prev_risk and next_risk:
        signal = "both-side serial risk"
    elif prev_risk:
        signal = "prev-side serial risk"
    elif next_risk:
        signal = "next-side serial risk"
    elif prev_unclear or next_unclear:
        signal = "adjacency unclear"
    else:
        signal = "serial risk low"

    prev_desc = describe_neighbor(prev_event, prev_gap, source_map)
    next_desc = describe_neighbor(next_event, next_gap, source_map)
    return signal, prev_desc, next_desc


def dependency_risk_label(signal: str) -> str:
    mapping = {
        "serial risk low": "low",
        "prev-side serial risk": "high",
        "next-side serial risk": "high",
        "both-side serial risk": "high",
        "adjacency unclear": "unclear",
    }
    return mapping.get(signal, signal)


def build_priority_and_recommendation(
    verdict: str,
    category: str,
    dependency_signal: str,
    stats: AggregateStats,
    share_pct: float,
) -> Tuple[str, str]:
    dep_label = dependency_risk_label(dependency_signal)
    if share_pct < 1.0:
        return "P5", "skip"

    if verdict == "headroom":
        if dep_label == "low":
            if category == "communication":
                return "P1", "try overlap"
            return "P1", "try fusion"
        return "P2", "check deps"

    if verdict == "low-roi-hidden":
        return "P4", "skip"

    if stats.exclusive_ratio >= 0.85 and dep_label == "low":
        return "P3", "defer"
    if stats.hidden_ratio >= 0.7:
        return "P5", "skip"
    if dep_label == "high":
        return "P4", "check deps"
    if dep_label == "unclear":
        return "P4", "inspect"
    return "P4", "defer"


def make_action_row(
    stats: AggregateStats,
    verdict: str,
    suggestion: str,
    source_map: Dict[str, KernelSourceStats],
    formal_events: Sequence[KernelEvent],
    neighbor_index: Dict[int, Tuple[Optional[KernelEvent], Optional[KernelEvent]]],
    total_busy_us: float,
) -> ActionRow:
    source = relaxed_source_stats_lookup(source_map, stats.name)
    representative_idx = stats.representative_idx
    dependency_signal = "adjacency unclear"
    prev_neighbor = "none"
    next_neighbor = "none"
    share_pct = (stats.total_us / total_busy_us * 100.0) if total_busy_us > 0 else 0.0
    if representative_idx is not None:
        current_event = next(
            (event for event in formal_events if event.idx == representative_idx), None
        )
        if current_event is not None:
            prev_event, next_event = neighbor_index.get(
                representative_idx, (None, None)
            )
            dependency_signal, prev_neighbor, next_neighbor = (
                classify_dependency_signal(
                    current=current_event,
                    source=source,
                    prev_event=prev_event,
                    next_event=next_event,
                    source_map=source_map,
                )
            )
    priority, recommendation = build_priority_and_recommendation(
        verdict=verdict,
        category=stats.category,
        dependency_signal=dependency_signal,
        stats=stats,
        share_pct=share_pct,
    )

    return ActionRow(
        priority=priority,
        verdict=verdict,
        kernel=stats.name,
        category=stats.category,
        total_us=stats.total_us,
        share_pct=share_pct,
        exclusive_ratio=stats.exclusive_ratio,
        hidden_ratio=stats.hidden_ratio,
        python_scope=source.best_scope if source and source.best_scope else "unmapped",
        launch_op=source.best_launch_op if source and source.best_launch_op else "n/a",
        mapping_ratio=source.mapping_ratio if source else 0.0,
        dependency_signal=dependency_signal,
        prev_neighbor=prev_neighbor,
        next_neighbor=next_neighbor,
        recommendation=recommendation,
        suggestion=suggestion,
        representative_idx=representative_idx,
    )


def build_action_rows(
    aggregates: Dict[Tuple[str, str], AggregateStats],
    source_map: Dict[str, KernelSourceStats],
    formal_events: Sequence[KernelEvent],
    total_busy_us: float,
    table_limit: int,
) -> List[ActionRow]:
    rows: List[ActionRow] = []
    seen: set[str] = set()
    neighbor_index = build_stream_neighbor_index(formal_events)

    for stats in top_overlap_opportunities(aggregates):
        row = make_action_row(
            stats=stats,
            verdict="headroom",
            suggestion=build_headroom_suggestion(stats),
            source_map=source_map,
            formal_events=formal_events,
            neighbor_index=neighbor_index,
            total_busy_us=total_busy_us,
        )
        if row.priority == "P5":
            continue
        rows.append(row)
        seen.add(stats.name)

    for stats in top_hidden_low_roi(aggregates):
        if stats.name in seen:
            continue
        rows.append(
            make_action_row(
                stats=stats,
                verdict="low-roi-hidden",
                suggestion=build_hidden_suggestion(stats),
                source_map=source_map,
                formal_events=formal_events,
                neighbor_index=neighbor_index,
                total_busy_us=total_busy_us,
            )
        )
        seen.add(stats.name)

    if table_limit > 0:
        return rows[:table_limit]
    return rows


def render_action_table(rows: Sequence[ActionRow]) -> List[str]:
    lines = [
        "| Priority | Verdict | Kernel | Python scope | Formal signal | Dep risk | Recommendation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not rows:
        lines.append(
            "| - | No overlap rows met the reporting bar in the formal trace. | - | - | - | - | - |"
        )
        return lines
    for row in rows:
        formal_signal = (
            f"share {row.share_pct:.1f}%, "
            f"excl {row.exclusive_ratio * 100:.1f}% / "
            f"hid {row.hidden_ratio * 100:.1f}%"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row.priority,
                    row.verdict,
                    row.kernel,
                    row.python_scope,
                    f"{row.total_us:.1f} us, {formal_signal}",
                    dependency_risk_label(row.dependency_signal),
                    row.recommendation,
                ]
            )
            + " |"
        )
    return lines


def trace_summary_line(bundle: TraceBundle) -> str:
    events = bundle.events
    streams = sorted({event.stream for event in events})
    if bundle.overlap_stats is None:
        return f"{bundle.label}: {len(events)} kernel events, {len(streams)} streams"
    overlap_ratio = (
        bundle.overlap_stats["total_overlap_us"] / bundle.overlap_stats["total_busy_us"]
        if bundle.overlap_stats["total_busy_us"]
        else 0.0
    )
    return (
        f"{bundle.label}: {len(events)} kernel events, {len(streams)} streams, "
        f"busy={bundle.overlap_stats['total_busy_us']:.1f} us, "
        f"2+ stream overlap={bundle.overlap_stats['total_overlap_us']:.1f} us "
        f"({overlap_ratio * 100:.1f}%), "
        f"peak_concurrency={int(bundle.overlap_stats['max_concurrent_streams'])}"
    )


def launch_summary(server_args: Optional[dict]) -> Optional[str]:
    if not server_args:
        return None
    model_path = server_args.get("model_path") or server_args.get("model")
    shape_bits = []
    if model_path:
        shape_bits.append(f"model={model_path}")
    for key in ("tp_size", "dp_size", "pp_size", "ep_size", "enable_dp_attention"):
        if key in server_args:
            shape_bits.append(f"{key}={server_args[key]}")
    return ", ".join(shape_bits) if shape_bits else None


def build_report(
    mapping_bundle: TraceBundle,
    formal_bundle: TraceBundle,
    source_map: Dict[str, KernelSourceStats],
    aggregates: Dict[Tuple[str, str], AggregateStats],
    rows: Sequence[ActionRow],
    window_us: Optional[float],
    timeline_count: int,
    width: int,
    table_only: bool,
) -> str:
    lines: List[str] = []
    lines.append(f"Mapping Trace: {mapping_bundle.trace_path}")
    mapping_launch = launch_summary(mapping_bundle.server_args)
    if mapping_launch:
        lines.append(f"Mapping Launch: {mapping_launch}")
    if mapping_bundle.pid:
        lines.append(f"Mapping PID slice: {mapping_bundle.pid}")
    lines.append(trace_summary_line(mapping_bundle))

    lines.append("")
    lines.append(f"Formal Trace: {formal_bundle.trace_path}")
    formal_launch = launch_summary(formal_bundle.server_args)
    if formal_launch:
        lines.append(f"Formal Launch: {formal_launch}")
    if formal_bundle.pid:
        lines.append(f"Formal PID slice: {formal_bundle.pid}")
    lines.append(trace_summary_line(formal_bundle))

    mapped_kernels = sum(1 for stats in source_map.values() if stats.mapped_count > 0)
    table_mapped = sum(1 for row in rows if row.python_scope != "unmapped")
    lines.append("")
    lines.append(
        "Source Map Coverage: "
        f"{mapped_kernels}/{len(source_map)} mapping-trace kernels found a Python scope, "
        f"{table_mapped}/{len(rows)} table rows were mapped back to code."
    )

    lines.append("")
    lines.append("Action Table")
    lines.extend(render_action_table(rows))

    if not table_only:
        detail_lookup = {row.kernel: row for row in rows}
        focus_rows = list(rows)
        lines.append("")
        lines.append("Source Context")
        if not focus_rows:
            lines.append("  No source-mapped rows to expand.")
        else:
            for index, row in enumerate(focus_rows, start=1):
                stats = relaxed_source_stats_lookup(source_map, row.kernel)
                lines.append(
                    f"  {index}. {short_name(row.kernel, 88)} [{row.priority}, {row.verdict}, {row.category}] "
                    f"mapping={row.mapping_ratio * 100:.1f}%"
                )
                lines.append(f"     time share: {row.share_pct:.1f}%")
                lines.append(f"     python scope: {row.python_scope}")
                lines.append(f"     launch op: {row.launch_op}")
                lines.append(
                    f"     dependency signal: {dependency_risk_label(row.dependency_signal)}"
                )
                lines.append(f"     prev neighbor: {row.prev_neighbor}")
                lines.append(f"     next neighbor: {row.next_neighbor}")
                lines.append(f"     recommendation: {row.recommendation}")
                if stats and stats.best_chain:
                    lines.append(
                        f"     call chain: {short_name(stats.best_chain, 132)}"
                    )
                lines.append(f"     note: {row.suggestion}")

        timeline_targets: List[int] = []
        for row in rows:
            if (
                row.representative_idx is not None
                and row.representative_idx not in timeline_targets
            ):
                timeline_targets.append(row.representative_idx)
        timeline_targets = timeline_targets[:timeline_count]

        if timeline_targets:
            lines.append("")
            lines.append("ASCII Timelines")
            for index, representative_idx in enumerate(timeline_targets, start=1):
                event = next(
                    event
                    for event in formal_bundle.events
                    if event.idx == representative_idx
                )
                mapped_scope = (
                    detail_lookup.get(event.canonical_name).python_scope
                    if event.canonical_name in detail_lookup
                    else "unmapped"
                )
                lines.append(
                    f"  Window {index}: {short_name(event.canonical_name, 90)} "
                    f"[{event.category}] ts={event.ts:.1f} us dur={event.dur:.1f} us"
                )
                lines.append(f"  mapped scope: {short_name(mapped_scope, 120)}")
                lines.append(
                    render_ascii_timeline(
                        formal_bundle.events, representative_idx, window_us, width
                    )
                )
                lines.append("")

        lines.append("Notes")
        lines.append(
            "  - The mapping trace should be graph-off so kernel-to-code attribution stays readable."
        )
        lines.append(
            "  - The formal trace should keep the real serving optimizations enabled; the overlap read comes from this trace."
        )
        lines.append(
            "  - A mapped Python scope is a launch-site clue, not proof that the code is dependency-free to reorder."
        )
    return "\n".join(lines).rstrip()


def discover_trace_file(path: Path) -> Tuple[Path, Optional[dict]]:
    traces, server_args = discover_trace_targets(path, all_traces=False)
    if not traces:
        raise FileNotFoundError(f"No trace files found under {path}")
    return traces[0], server_args or load_server_args(path)


def resolve_trace_source(
    label: str,
    input_path: Optional[str],
    url: Optional[str],
    output_dir: Optional[str],
    profile_prefix: Optional[str],
    args: argparse.Namespace,
) -> TraceBundle:
    if bool(input_path) == bool(url):
        raise ValueError(f"{label} trace requires exactly one of input path or URL.")

    if url:
        target_dir = shared_run_profiler(
            url=url,
            output_dir=output_dir,
            num_steps=args.num_steps,
            profile_by_stage=args.profile_by_stage,
            merge_profiles=args.merge_profiles,
            profile_prefix=profile_prefix,
            probe_requests=max(0, args.probe_requests),
            probe_prompt=args.probe_prompt,
            probe_max_new_tokens=args.probe_max_new_tokens,
            probe_delay=args.probe_delay,
            start_step=args.start_step,
        )
        trace_path, server_args = discover_trace_file(target_dir)
    else:
        trace_path, server_args = discover_trace_file(Path(input_path).resolve())

    trace = load_trace_json(trace_path)
    raw_events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    events, pid = extract_kernel_events(trace, args.pid_substring)
    if not events:
        raise RuntimeError(f"No GPU kernel events found in {trace_path}")
    return TraceBundle(
        label=label,
        trace_path=trace_path,
        server_args=server_args,
        raw_events=raw_events,
        events=events,
        pid=pid,
    )
