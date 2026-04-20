#!/usr/bin/env python3
"""Summarize a collected SGLang incident bundle into a compact first-pass report."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

METRIC_RE = re.compile(
    r"^(?P<name>[^{\s]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)
LABEL_RE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)="((?:[^"\\]|\\.)*)"')


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def unwrap_result(path: Path) -> Optional[Dict[str, Any]]:
    obj = load_json(path)
    if obj is None:
        return None
    if isinstance(obj, dict) and "json" in obj:
        return obj.get("json")
    return obj


def read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text()


def endpoint_ok(bundle_dir: Path, stem: str) -> bool:
    return (bundle_dir / f"{stem}.txt").exists() and not (
        bundle_dir / f"{stem}.txt.error.json"
    ).exists()


def parse_labels(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    labels = {}
    for key, value in LABEL_RE.findall(raw):
        labels[key] = bytes(value, "utf-8").decode("unicode_escape")
    return labels


def parse_metrics(metrics_text: str) -> Dict[str, list[dict[str, Any]]]:
    series: Dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        series[match.group("name")].append(
            {
                "labels": parse_labels(match.group("labels")),
                "value": float(match.group("value")),
            }
        )
    return series


def metric_sum(metrics: Dict[str, list[dict[str, Any]]], name: str) -> float:
    return sum(item["value"] for item in metrics.get(name, []))


def safe_div(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "n/a"
    return f"{value:.{digits}f}"


def compute_stage_averages(
    metrics: Dict[str, list[dict[str, Any]]], sum_name: str, count_name: str
) -> Dict[str, float]:
    grouped_sum: Dict[str, float] = defaultdict(float)
    grouped_count: Dict[str, float] = defaultdict(float)
    for item in metrics.get(sum_name, []):
        stage = item["labels"].get("stage", "")
        rank = item["labels"].get("tp_rank", "")
        grouped_sum[f"{stage}|{rank}"] += item["value"]
    for item in metrics.get(count_name, []):
        stage = item["labels"].get("stage", "")
        rank = item["labels"].get("tp_rank", "")
        grouped_count[f"{stage}|{rank}"] += item["value"]

    result: Dict[str, float] = {}
    for key, total_sum in grouped_sum.items():
        stage, _rank = key.split("|", 1)
        avg = safe_div(total_sum, grouped_count.get(key))
        if avg is None:
            continue
        result[stage] = max(result.get(stage, 0.0), avg)
    return result


def add_signal(signals: list[str], text: str) -> None:
    if text not in signals:
        signals.append(text)


def build_summary(bundle_dir: Path) -> Dict[str, Any]:
    metadata = load_json(bundle_dir / "metadata.json") or {}
    model_info = unwrap_result(bundle_dir / "model_info.json") or {}
    server_info = unwrap_result(bundle_dir / "server_info.json") or {}
    loads_info = unwrap_result(bundle_dir / "loads_all.json") or {}
    metrics_text = read_text(bundle_dir / "metrics.txt") or ""
    metrics = parse_metrics(metrics_text)

    aggregate = loads_info.get("aggregate") or {}
    loads = loads_info.get("loads") or []
    load0 = loads[0] if loads else {}
    internal_states = server_info.get("internal_states") or []
    runtime_state = internal_states[0] if internal_states else {}
    memory_usage = runtime_state.get("memory_usage") or load0.get("memory") or {}

    ttft_avg = safe_div(
        metric_sum(metrics, "sglang:time_to_first_token_seconds_sum"),
        metric_sum(metrics, "sglang:time_to_first_token_seconds_count"),
    )
    e2e_avg = safe_div(
        metric_sum(metrics, "sglang:e2e_request_latency_seconds_sum"),
        metric_sum(metrics, "sglang:e2e_request_latency_seconds_count"),
    )
    queue_avg = safe_div(
        metric_sum(metrics, "sglang:queue_time_seconds_sum"),
        metric_sum(metrics, "sglang:queue_time_seconds_count"),
    )
    per_stage_avg = compute_stage_averages(
        metrics,
        "sglang:per_stage_req_latency_seconds_sum",
        "sglang:per_stage_req_latency_seconds_count",
    )

    summary: Dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "base_url": metadata.get("base_url"),
        "collected_at": metadata.get("collected_at"),
        "health": {
            "health_ok": endpoint_ok(bundle_dir, "health"),
            "health_generate_ok": endpoint_ok(bundle_dir, "health_generate"),
        },
        "model": {
            "model_path": model_info.get("model_path") or server_info.get("model_path"),
            "served_model_name": server_info.get("served_model_name"),
            "weight_version": model_info.get("weight_version")
            or server_info.get("weight_version"),
            "model_type": model_info.get("model_type"),
            "is_generation": model_info.get("is_generation"),
        },
        "topology": {
            "tp_size": server_info.get("tp_size"),
            "dp_size": server_info.get("dp_size"),
            "pp_size": server_info.get("pp_size"),
            "ep_size": server_info.get("ep_size"),
            "disaggregation_mode": server_info.get("disaggregation_mode"),
            "attention_backend": server_info.get("attention_backend"),
            "sampling_backend": server_info.get("sampling_backend"),
            "schedule_policy": server_info.get("schedule_policy"),
            "enable_trace": server_info.get("enable_trace"),
            "enable_metrics": server_info.get("enable_metrics"),
        },
        "capacity": {
            "max_total_num_tokens": server_info.get("max_total_num_tokens"),
            "max_req_input_len": server_info.get("max_req_input_len"),
            "effective_max_running_requests_per_dp": runtime_state.get(
                "effective_max_running_requests_per_dp"
            )
            or load0.get("max_running_requests"),
            "weight_gb": memory_usage.get("weight") or memory_usage.get("weight_gb"),
            "kv_cache_gb": memory_usage.get("kvcache")
            or memory_usage.get("kv_cache_gb"),
            "graph_gb": memory_usage.get("graph") or memory_usage.get("graph_gb"),
            "token_capacity": memory_usage.get("token_capacity"),
        },
        "point_in_time_load": {
            "running_reqs": aggregate.get(
                "total_running_reqs", load0.get("num_running_reqs")
            ),
            "waiting_reqs": aggregate.get(
                "total_waiting_reqs", load0.get("num_waiting_reqs")
            ),
            "total_reqs": aggregate.get("total_reqs", load0.get("num_total_reqs")),
            "token_usage": aggregate.get("avg_token_usage", load0.get("token_usage")),
            "avg_throughput": aggregate.get(
                "avg_throughput", load0.get("gen_throughput")
            ),
            "avg_utilization": aggregate.get(
                "avg_utilization", load0.get("utilization")
            ),
            "cache_hit_rate": load0.get("cache_hit_rate"),
            "queues": load0.get("queues"),
            "disaggregation": load0.get("disaggregation"),
        },
        "metrics": {
            "request_count": metric_sum(metrics, "sglang:num_requests_total"),
            "prompt_tokens_total": metric_sum(metrics, "sglang:prompt_tokens_total"),
            "generation_tokens_total": metric_sum(
                metrics, "sglang:generation_tokens_total"
            ),
            "avg_ttft_seconds": ttft_avg,
            "avg_e2e_seconds": e2e_avg,
            "avg_queue_time_seconds": queue_avg,
            "stage_avg_seconds_max_tp_rank": per_stage_avg,
        },
        "signals": [],
    }

    signals = summary["signals"]
    health = summary["health"]
    point_in_time_load = summary["point_in_time_load"]

    if health["health_ok"] and not health["health_generate_ok"]:
        add_signal(
            signals,
            "/health is green but /health_generate failed. Suspect runtime or scheduler path, not just HTTP liveness.",
        )
    if not health["health_ok"]:
        add_signal(
            signals,
            "/health failed. Treat this as a startup, crash, or global unhealthy incident first.",
        )
    if point_in_time_load.get("waiting_reqs", 0) > 0:
        add_signal(
            signals,
            f"Point-in-time load shows queue buildup: waiting_reqs={point_in_time_load['waiting_reqs']}.",
        )
    if (
        point_in_time_load.get("token_usage") is not None
        and point_in_time_load["token_usage"] >= 0.9
    ):
        add_signal(
            signals,
            "Token usage is near saturation. KV or token-capacity pressure may explain latency.",
        )
    if (
        ttft_avg is not None
        and queue_avg is not None
        and ttft_avg > 2.0
        and queue_avg < 0.2
    ):
        add_signal(
            signals,
            f"Average TTFT is high ({fmt_float(ttft_avg)}s) while average queue time is low ({fmt_float(queue_avg)}s). Suspect compute-side prefill cost or a request-path slowdown rather than queue pressure.",
        )
    prefill_forward = per_stage_avg.get("prefill_forward")
    request_process = per_stage_avg.get("request_process")
    if (
        prefill_forward is not None
        and request_process is not None
        and prefill_forward > max(0.5, request_process * 10)
    ):
        add_signal(
            signals,
            f"Prefill forward dominates first-pass stage timing: prefill_forward≈{fmt_float(prefill_forward)}s vs request_process≈{fmt_float(request_process)}s.",
        )
    if (
        point_in_time_load.get("running_reqs", 0) == 0
        and point_in_time_load.get("waiting_reqs", 0) == 0
    ):
        add_signal(
            signals,
            "Bundle snapshot was captured while the server was effectively idle. Reproduce under live traffic or replayed workload if the incident is intermittent.",
        )

    return summary


def render_text(summary: Dict[str, Any]) -> str:
    health = summary["health"]
    model = summary["model"]
    topology = summary["topology"]
    capacity = summary["capacity"]
    load = summary["point_in_time_load"]
    metrics = summary["metrics"]
    stage_avgs = metrics["stage_avg_seconds_max_tp_rank"]

    lines = [
        f"Bundle: {summary['bundle_dir']}",
        f"Base URL: {summary.get('base_url') or 'n/a'}",
        f"Collected At: {summary.get('collected_at') or 'n/a'}",
        "",
        f"Health: /health={'ok' if health['health_ok'] else 'failed'} /health_generate={'ok' if health['health_generate_ok'] else 'failed'}",
        f"Model: {model.get('model_path') or 'n/a'} weight_version={model.get('weight_version') or 'n/a'} type={model.get('model_type') or 'n/a'}",
        "Topology: "
        f"tp={topology.get('tp_size')} dp={topology.get('dp_size')} pp={topology.get('pp_size')} ep={topology.get('ep_size')} "
        f"disagg={topology.get('disaggregation_mode')} trace={topology.get('enable_trace')} metrics={topology.get('enable_metrics')}",
        "Capacity: "
        f"max_total_tokens={capacity.get('max_total_num_tokens')} "
        f"max_running_reqs={capacity.get('effective_max_running_requests_per_dp')} "
        f"weight_gb={fmt_float(capacity.get('weight_gb'))} "
        f"kv_cache_gb={fmt_float(capacity.get('kv_cache_gb'))} "
        f"graph_gb={fmt_float(capacity.get('graph_gb'))}",
        "Point-in-time load: "
        f"running={load.get('running_reqs')} waiting={load.get('waiting_reqs')} total={load.get('total_reqs')} "
        f"token_usage={fmt_float(load.get('token_usage'))} throughput={fmt_float(load.get('avg_throughput'))} "
        f"cache_hit_rate={fmt_float(load.get('cache_hit_rate'))}",
        "Metrics: "
        f"requests={fmt_float(metrics.get('request_count'), 0)} "
        f"prompt_tokens={fmt_float(metrics.get('prompt_tokens_total'), 0)} "
        f"generation_tokens={fmt_float(metrics.get('generation_tokens_total'), 0)} "
        f"avg_ttft_s={fmt_float(metrics.get('avg_ttft_seconds'))} "
        f"avg_e2e_s={fmt_float(metrics.get('avg_e2e_seconds'))} "
        f"avg_queue_s={fmt_float(metrics.get('avg_queue_time_seconds'))}",
    ]

    if stage_avgs:
        stage_parts = [
            f"{name}={fmt_float(value)}s" for name, value in sorted(stage_avgs.items())
        ]
        lines.append("Stage Averages (max across TP ranks): " + ", ".join(stage_parts))

    queues = load.get("queues") or {}
    if queues:
        lines.append(
            "Queues: "
            + ", ".join(f"{key}={value}" for key, value in sorted(queues.items()))
        )

    disagg = load.get("disaggregation") or {}
    if disagg:
        lines.append(
            "Disaggregation: "
            + ", ".join(f"{key}={value}" for key, value in sorted(disagg.items()))
        )

    lines.append("")
    lines.append("Signals:")
    if summary["signals"]:
        lines.extend(f"- {signal}" for signal in summary["signals"])
    else:
        lines.append("- No strong heuristic signal from the first-pass bundle.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a collected SGLang incident bundle into a compact first-pass report."
    )
    parser.add_argument(
        "bundle_dir", help="Path to a directory produced by collect_incident_bundle.py"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output text path. Defaults to <bundle_dir>/SUMMARY_REPORT.txt.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output JSON path. Defaults to <bundle_dir>/SUMMARY_REPORT.json.",
    )
    parser.add_argument(
        "--stdout-json",
        action="store_true",
        help="Print the JSON summary to stdout instead of the text report.",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    if not bundle_dir.is_dir():
        raise SystemExit(
            f"bundle_dir does not exist or is not a directory: {bundle_dir}"
        )

    summary = build_summary(bundle_dir)
    out_text = render_text(summary)

    text_path = Path(args.out) if args.out else bundle_dir / "SUMMARY_REPORT.txt"
    json_path = (
        Path(args.json_out) if args.json_out else bundle_dir / "SUMMARY_REPORT.json"
    )
    text_path.write_text(out_text)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if args.stdout_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(out_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
