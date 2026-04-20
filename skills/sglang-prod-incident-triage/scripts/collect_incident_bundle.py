#!/usr/bin/env python3
"""Collect a lightweight SGLang incident bundle from a live server.

This script intentionally prefers read-only evidence collection. It does not
mutate server state except for optional best-effort admin reads.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


def build_opener(token: Optional[str]):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.build_opener()


def request_text(
    base_url: str,
    path: str,
    token: Optional[str],
    timeout: float = 10.0,
) -> Tuple[bool, int, str]:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return False, e.code, body
    except Exception as e:  # noqa: BLE001
        return False, -1, f"{type(e).__name__}: {e}"


def request_json(
    base_url: str,
    path: str,
    token: Optional[str],
    timeout: float = 10.0,
) -> Dict:
    ok, status, body = request_text(base_url, path, token, timeout=timeout)
    result: Dict[str, object] = {
        "ok": ok,
        "status": status,
        "path": path,
    }
    if ok:
        try:
            result["json"] = json.loads(body)
        except json.JSONDecodeError:
            result["text"] = body
            result["decode_error"] = "response was not valid JSON"
    else:
        result["error"] = body
    return result


def request_plain(
    base_url: str,
    path: str,
    token: Optional[str],
    timeout: float = 10.0,
) -> Dict:
    ok, status, body = request_text(base_url, path, token, timeout=timeout)
    result: Dict[str, object] = {
        "ok": ok,
        "status": status,
        "path": path,
    }
    if ok:
        result["text"] = body
    else:
        result["error"] = body
    return result


def write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect a read-only SGLang incident bundle from a live server."
    )
    parser.add_argument(
        "--base-url", required=True, help="Base URL, e.g. http://127.0.0.1:30000"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SGLANG_BEARER_TOKEN"),
        help="Bearer token for protected endpoints. Defaults to $SGLANG_BEARER_TOKEN.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to ./incident_bundle_<timestamp>.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds.",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir or f"./incident_bundle_{timestamp}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "base_url": args.base_url,
        "collected_at": timestamp,
        "token_provided": bool(args.token),
        "timeout_seconds": args.timeout,
    }
    write_json(outdir / "metadata.json", metadata)

    json_endpoints = {
        "model_info.json": "/model_info",
        "server_info.json": "/server_info",
        "loads_all.json": "/v1/loads?include=all",
        "loads_core_queues_disagg.json": "/v1/loads?include=core,queues,disagg,spec",
        "hicache_storage_backend.json": "/hicache/storage-backend",
    }
    text_endpoints = {
        "health.txt": "/health",
        "health_generate.txt": "/health_generate",
        "metrics.txt": "/metrics",
    }

    summary_lines = []

    for filename, path in text_endpoints.items():
        result = request_plain(args.base_url, path, args.token, timeout=args.timeout)
        if result.get("ok"):
            write_text(outdir / filename, str(result.get("text", "")))
            summary_lines.append(f"{filename}: ok")
        else:
            write_json(outdir / (filename + ".error.json"), result)
            summary_lines.append(
                f"{filename}: failed status={result.get('status')} error={result.get('error')}"
            )

    for filename, path in json_endpoints.items():
        result = request_json(args.base_url, path, args.token, timeout=args.timeout)
        write_json(outdir / filename, result)
        if result.get("ok"):
            summary_lines.append(f"{filename}: ok")
        else:
            summary_lines.append(
                f"{filename}: failed status={result.get('status')} error={result.get('error')}"
            )

    notes = [
        "This bundle is read-only. It does not start profiling or change trace level.",
        "HiCache status may fail if admin_api_key is not configured or the wrong bearer token was used.",
        "loads_all.json is the best point-in-time load snapshot in this bundle.",
        "metrics.txt is raw Prometheus text intended for follow-up parsing.",
    ]

    write_text(outdir / "SUMMARY.txt", "\n".join(summary_lines + [""] + notes) + "\n")
    print(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
