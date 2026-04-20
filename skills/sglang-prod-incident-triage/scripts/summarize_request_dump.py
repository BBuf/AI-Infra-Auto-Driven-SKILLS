#!/usr/bin/env python3
"""Summarize a trusted SGLang request dump or crash dump."""

from __future__ import annotations

import argparse
import glob
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def iter_files(args: argparse.Namespace) -> Sequence[Path]:
    if args.input_file:
        return [Path(args.input_file)]
    if args.input_folder:
        return [Path(p) for p in sorted(glob.glob(f"{args.input_folder}/*.pkl"))]
    raise SystemExit("Either --input-file or --input-folder must be provided.")


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        return payload
    return {"requests": payload}


def pick_text_preview(req: Any) -> str:
    candidates = [
        get_field(req, "origin_input_text"),
        get_field(req, "text"),
        get_field(req, "prompt"),
    ]
    for value in candidates:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first:
                return first
    return ""


def format_timestamp(ts: Any) -> str:
    if not isinstance(ts, (int, float)):
        return "n/a"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def summarize_request(
    record: tuple[Any, dict[str, Any], Any, Any], idx: int, preview_chars: int
) -> list[str]:
    req, output, start_time, end_time = record
    preview = pick_text_preview(req).replace("\n", " ").strip()
    if len(preview) > preview_chars:
        preview = preview[: preview_chars - 3] + "..."

    output_dict = output if isinstance(output, dict) else {}
    meta_info = get_field(output_dict, "meta_info", {}) or {}
    rid = get_field(req, "rid") or get_field(meta_info, "id")
    stream = bool(get_field(req, "stream", False))
    prompt_tokens = get_field(meta_info, "prompt_tokens")
    completion_tokens = get_field(meta_info, "completion_tokens")
    duration = (
        end_time - start_time
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float))
        else None
    )

    lines = [
        (
            f"[{idx}] rid={rid or 'n/a'} stream={stream} "
            f"prompt_tokens={prompt_tokens if prompt_tokens is not None else 'n/a'} "
            f"completion_tokens={completion_tokens if completion_tokens is not None else 'n/a'} "
            f"start={format_timestamp(start_time)} "
            f"elapsed_s={duration:.3f}"
            if duration is not None
            else f"[{idx}] rid={rid or 'n/a'} stream={stream} "
            f"prompt_tokens={prompt_tokens if prompt_tokens is not None else 'n/a'} "
            f"completion_tokens={completion_tokens if completion_tokens is not None else 'n/a'} "
            f"start={format_timestamp(start_time)} elapsed_s=n/a"
        )
    ]
    if preview:
        lines.append(f"      text={preview}")
    return lines


def summarize_file(path: Path, max_requests: int, preview_chars: int) -> str:
    payload = load_payload(path)
    requests = payload.get("requests") or []
    server_args = payload.get("server_args")
    launch_command = payload.get("launch_command")

    model_path = get_field(server_args, "model_path")
    tp_size = get_field(server_args, "tp_size")
    dp_size = get_field(server_args, "dp_size")
    pp_size = get_field(server_args, "pp_size")
    host = get_field(server_args, "host")
    port = get_field(server_args, "port")

    timestamps = [
        record[2]
        for record in requests
        if isinstance(record, tuple)
        and len(record) >= 4
        and isinstance(record[2], (int, float))
    ]
    time_span = (
        max(timestamps) - min(timestamps)
        if len(timestamps) >= 2
        else 0.0 if len(timestamps) == 1 else None
    )

    lines = [
        f"File: {path}",
        f"Requests: {len(requests)}",
        f"Model: {model_path or 'n/a'}",
        f"Topology: tp={tp_size if tp_size is not None else 'n/a'} "
        f"dp={dp_size if dp_size is not None else 'n/a'} "
        f"pp={pp_size if pp_size is not None else 'n/a'}",
        f"Endpoint: {host or 'n/a'}:{port if port is not None else 'n/a'}",
        (
            f"Time span seconds: {time_span:.3f}"
            if time_span is not None
            else "Time span seconds: n/a"
        ),
    ]
    if launch_command:
        lines.append(f"Launch command: {launch_command}")

    for idx, record in enumerate(requests[:max_requests]):
        if not isinstance(record, tuple) or len(record) < 4:
            lines.append(f"[{idx}] Unsupported record shape: {type(record)!r}")
            continue
        lines.extend(summarize_request(record, idx, preview_chars))

    if len(requests) > max_requests:
        lines.append(f"... truncated {len(requests) - max_requests} more requests")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a trusted SGLang request dump or crash dump."
    )
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--input-folder", default=None)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--preview-chars", type=int, default=160)
    args = parser.parse_args()

    files = iter_files(args)
    if not files:
        raise SystemExit("No .pkl files matched the provided input.")

    for idx, path in enumerate(files):
        if idx:
            print()
        print(
            summarize_file(
                path=path,
                max_requests=args.max_requests,
                preview_chars=args.preview_chars,
            )
        )


if __name__ == "__main__":
    main()
