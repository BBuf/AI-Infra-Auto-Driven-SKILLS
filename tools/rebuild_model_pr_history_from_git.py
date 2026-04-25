#!/usr/bin/env python3
"""Rebuild model PR history docs from framework git traces.

The script uses current framework worktrees as the source of truth:

* `git ls-files` + per-model path filters select implementation-related files.
* `git log --name-only` on those files finds merged PR numbers that actually
  modified the model surface.
* GitHub's PR and files APIs provide the final diff metadata and short patch
  snippets used in the dossier cards.

It also preserves explicitly cited PRs that already existed in the skill or
history docs, because open PRs and still-relevant review tracks may not be
reachable from current `main`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMMON_ROOT = ROOT.parent
FRAMEWORK_ROOTS = {
    "sglang": COMMON_ROOT / "_worktrees" / "sglang-pr-history",
    "vllm": COMMON_ROOT / "_worktrees" / "vllm-pr-history",
}
REPOS = {
    "sglang": "sgl-project/sglang",
    "vllm": "vllm-project/vllm",
}
HISTORY_ROOT = ROOT / "model-pr-optimization-history"
SKILL_ROOT = ROOT / "skills" / "model-optimization"
CACHE_PATH = Path(os.environ.get("MODEL_PR_HISTORY_CACHE", "/tmp/model_pr_history_git_trace_cache_v4.json"))
TODAY = date.today().isoformat()

DIFFUSION_MODELS = {"ltx23-hq", "qwen-image", "z-image-turbo"}

MODEL_TITLES = {
    "deepseek-v3-r1": "DeepSeek V3/R1",
    "deepseek-v31": "DeepSeek V3.1",
    "deepseek-v32": "DeepSeek V3.2",
    "deepseek-v4": "DeepSeek V4",
    "ernie45": "ERNIE 4.5",
    "gemma4": "Gemma 4",
    "glm-vlm-ocr": "GLM VLM/OCR",
    "glm45": "GLM-4.5",
    "glm46-glm47": "GLM-4.6/4.7",
    "glm5-glm51": "GLM-5/5.1",
    "gpt-oss": "GPT-OSS",
    "hunyuan3-preview": "Hunyuan3 Preview",
    "intern-s1": "Intern-S1",
    "internvl35": "InternVL 3.5",
    "kimi": "Kimi K2/K2.5/Linear/VL",
    "llama4": "Llama 4",
    "mimo-v2-flash": "MiMo V2 Flash",
    "minimax": "MiniMax M2 Series",
    "mistral-small-4": "Mistral Small 4",
    "mixtral-quark-int4fp8-moe": "Mixtral Quark INT4/FP8 MoE",
    "moss-vl": "MOSS-VL",
    "nemotron-super": "Nemotron Super",
    "qwen-vlm-omni-asr": "Qwen VLM/Omni/ASR",
    "qwen3-coder": "Qwen3 Coder",
    "qwen3-core": "Qwen3 Core",
    "qwen3-next": "Qwen3 Next",
    "qwen35": "Qwen3.5",
    "qwen36": "Qwen3.6",
    "step35": "Step 3.5",
}

MODEL_ORDER = [
    "deepseek-v3-r1",
    "deepseek-v31",
    "deepseek-v32",
    "deepseek-v4",
    "ernie45",
    "gemma4",
    "glm-vlm-ocr",
    "glm45",
    "glm46-glm47",
    "glm5-glm51",
    "gpt-oss",
    "hunyuan3-preview",
    "intern-s1",
    "internvl35",
    "kimi",
    "llama4",
    "mimo-v2-flash",
    "minimax",
    "mistral-small-4",
    "mixtral-quark-int4fp8-moe",
    "moss-vl",
    "nemotron-super",
    "qwen-vlm-omni-asr",
    "qwen3-coder",
    "qwen3-core",
    "qwen3-next",
    "qwen35",
    "qwen36",
    "step35",
]

# The filters intentionally start from implementation-adjacent files: model
# wrappers, configs, processors, parsers, attention/kernel hooks, and the
# model-specific docs/tests that usually ship in the same support PR.
MODEL_FILTERS: dict[str, dict[str, dict[str, list[str]]]] = {
    "sglang": {
        "deepseek-v3-r1": {
            "include": [
                "*deepseek_v2*",
                "*deepseek_common*",
                "*deepseekv3*",
                "*deepseek-v3*",
                "*deepseek-r1*",
                "*deepseek_r1*",
                "*nsa*",
            ],
            "exclude": ["*deepseek-v3.1*", "*deepseek-v3.2*", "*deepseek_v4*", "*deepseek-v4*"],
        },
        "deepseek-v31": {
            "include": [
                "*deepseek_v2*",
                "*deepseekv31*",
                "*deepseek-v3.1*",
                "*deepseek_v31*",
                "*deepseek_common*",
            ],
            "exclude": ["*deepseek-v3.2*", "*deepseek_v4*", "*deepseek-v4*"],
        },
        "deepseek-v32": {
            "include": [
                "*deepseek_v2*",
                "*deepseekv32*",
                "*deepseek-v3.2*",
                "*deepseek_v32*",
                "*encoding_dsv32*",
                "*nsa*",
                "*deepseek_common*",
            ],
            "exclude": ["*deepseek_v4*", "*deepseek-v4*"],
        },
        "deepseek-v4": {"include": ["*deepseek_v4*", "*deepseek-v4*"], "exclude": []},
        "ernie45": {"include": ["*ernie45*", "*ernie_4_5*", "*ernie-4.5*"], "exclude": []},
        "gemma4": {"include": ["*gemma4*"], "exclude": []},
        "glm-vlm-ocr": {"include": ["*glm_ocr*", "*glm-ocr*", "*glm4v*", "*glm-vlm*", "*glm_vlm*"], "exclude": []},
        "glm45": {"include": ["*glm4_moe*", "*glm-4.5*", "*glm45*"], "exclude": ["*glm47*", "*glm-4.7*", "*glm5*"]},
        "glm46-glm47": {"include": ["*glm47*", "*glm4_moe*", "*glm-4.6*", "*glm-4.7*"], "exclude": ["*glm5*"]},
        "glm5-glm51": {"include": ["*glm5*", "*glm-5*", "*glm51*", "*glm-5.1*"], "exclude": []},
        "gpt-oss": {"include": ["*gpt_oss*", "*gpt-oss*"], "exclude": []},
        "hunyuan3-preview": {
            "include": ["*hunyuan3-preview*", "*hunyuan3_preview*", "*hy3_preview*", "*hunyuan_detector*", "*hy3*"],
            "exclude": ["*hunyuan3d*", "*diffusion*", "*image_generation*"],
        },
        "intern-s1": {"include": ["*interns1*", "*intern-s1*", "*internlm_detector*"], "exclude": []},
        "internvl35": {"include": ["*internvl*", "*intern_vit*", "*internvl35*", "*internvl3.5*"], "exclude": ["*interns1*"]},
        "kimi": {"include": ["*kimi*", "*moonvit*"], "exclude": []},
        "llama4": {"include": ["*llama4*", "*mllama4*"], "exclude": []},
        "mimo-v2-flash": {"include": ["*mimo*", "*mimo_v2_flash*"], "exclude": []},
        "minimax": {"include": ["*minimax*"], "exclude": []},
        "mistral-small-4": {"include": ["*mistral*", "*ministral*"], "exclude": []},
        "mixtral-quark-int4fp8-moe": {"include": ["*mixtral*", "*quark*"], "exclude": []},
        "moss-vl": {"include": ["*moss_vl*", "*moss-vl*"], "exclude": []},
        "nemotron-super": {"include": ["*nemotron*", "*jet_nemotron*"], "exclude": []},
        "qwen-vlm-omni-asr": {
            "include": [
                "*qwen2_vl*",
                "*qwen2_5_vl*",
                "*qwen3_vl*",
                "*qwen3_omni*",
                "*qwen3_asr*",
                "*qwen2_audio*",
                "*qwen_vl*",
                "*omni*",
                "*glmasr*",
            ],
            "exclude": ["*qwen-image*", "*qwen_image*", "*diffusion*"],
        },
        "qwen3-coder": {"include": ["*qwen3_coder*", "*qwen3-coder*", "*qwen3.py"], "exclude": []},
        "qwen3-core": {"include": ["*qwen3.py", "*qwen3_moe.py", "*qwen3_moe_mtp.py", "*qwen3-deployment*", "*Qwen3.mdx"], "exclude": ["*qwen3_next*", "*qwen3_5*", "*qwen35*", "*qwen36*", "*qwen3_vl*", "*qwen3_omni*", "*qwen3_asr*", "*qwen3-coder*", "*qwen-image*", "*diffusion*"]},
        "qwen3-next": {"include": ["*qwen3_next*"], "exclude": []},
        "qwen35": {"include": ["*qwen3_5*", "*qwen35*", "*qwen3.5*", "*qwen3-5*"], "exclude": []},
        "qwen36": {"include": ["*qwen36*", "*qwen3.6*", "*qwen3-6*"], "exclude": []},
        "step35": {"include": ["*step3p5*", "*step-3.5*", "*step35*", "*step3_5*"], "exclude": []},
    },
    "vllm": {
        "deepseek-v3-r1": {
            "include": ["*deepseek_v2*", "*deepseek_mtp*", "*deepseekv3*", "*deepseek-v3*", "*deepseek-r1*", "*deepseek_r1*"],
            "exclude": ["*deepseek-v3.1*", "*deepseek-v3.2*", "*deepseek_v4*", "*deepseek-v4*", "*deepseek_ocr*"],
        },
        "deepseek-v31": {
            "include": ["*deepseek_v2*", "*deepseek_mtp*", "*deepseekv31*", "*deepseek-v3.1*", "*deepseek_v31*"],
            "exclude": ["*deepseek-v3.2*", "*deepseek_v4*", "*deepseek-v4*", "*deepseek_ocr*"],
        },
        "deepseek-v32": {
            "include": ["*deepseek_v2*", "*deepseek_mtp*", "*deepseekv32*", "*deepseek-v3.2*", "*deepseek_v32*", "*dsv32*"],
            "exclude": ["*deepseek_v4*", "*deepseek-v4*", "*deepseek_ocr*"],
        },
        "deepseek-v4": {"include": ["*deepseek_v4*", "*deepseek-v4*"], "exclude": ["*deepseek_ocr*"]},
        "ernie45": {"include": ["*ernie45*", "*ernie_4_5*", "*ernie-4.5*", "*ernie_mtp*"], "exclude": []},
        "gemma4": {"include": ["*gemma4*"], "exclude": []},
        "glm-vlm-ocr": {"include": ["*glm_ocr*", "*glm-ocr*", "*glm4v*", "*glm4_1v*", "*glm_vlm*", "*glm-vlm*"], "exclude": []},
        "glm45": {"include": ["*glm4_moe*", "*glm-4.5*", "*glm45*"], "exclude": ["*glm47*", "*glm5*"]},
        "glm46-glm47": {"include": ["*glm47*", "*glm4_moe*", "*glm-4.6*", "*glm-4.7*"], "exclude": ["*glm5*"]},
        "glm5-glm51": {"include": ["*glm5*", "*glm-5*", "*glm51*", "*glm-5.1*"], "exclude": []},
        "gpt-oss": {"include": ["*gpt_oss*", "*gpt-oss*"], "exclude": []},
        "hunyuan3-preview": {"include": ["*hy_v3*", "*hunyuan3*", "*hunyuan3-preview*"], "exclude": ["*hunyuan3d*", "*diffusion*"]},
        "intern-s1": {"include": ["*interns1*", "*intern-s1*", "*internlm2*", "*internlm*"], "exclude": ["*internvl*"]},
        "internvl35": {"include": ["*internvl*", "*intern_vit*", "*internvl35*", "*internvl3.5*"], "exclude": ["*interns1*"]},
        "kimi": {"include": ["*kimi*", "*moonvit*"], "exclude": []},
        "llama4": {"include": ["*llama4*", "*mllama4*"], "exclude": []},
        "mimo-v2-flash": {"include": ["*mimo*", "*mimo_v2_flash*"], "exclude": []},
        "minimax": {"include": ["*minimax*"], "exclude": []},
        "mistral-small-4": {"include": ["*mistral*", "*ministral*"], "exclude": []},
        "mixtral-quark-int4fp8-moe": {"include": ["*mixtral*", "*quark*"], "exclude": []},
        "moss-vl": {"include": ["*moss_vl*", "*moss-vl*"], "exclude": []},
        "nemotron-super": {"include": ["*nemotron*", "*jet_nemotron*", "*nano_nemotron*"], "exclude": []},
        "qwen-vlm-omni-asr": {
            "include": [
                "*qwen2_vl*",
                "*qwen2_5_vl*",
                "*qwen3_vl*",
                "*qwen3_omni*",
                "*qwen3_asr*",
                "*qwen2_audio*",
                "*qwen_vl*",
                "*omni*",
                "*glmasr*",
            ],
            "exclude": ["*qwen-image*", "*qwen_image*", "*diffusion*"],
        },
        "qwen3-coder": {"include": ["*qwen3_coder*", "*qwen3-coder*", "*qwen3.py"], "exclude": []},
        "qwen3-core": {"include": ["*qwen3.py", "*qwen3_moe.py", "*qwen3_moe_mtp.py", "*qwen3_dflash.py", "*qwen3-deployment*", "*Qwen3.mdx"], "exclude": ["*qwen3_next*", "*qwen3_5*", "*qwen35*", "*qwen36*", "*qwen3_vl*", "*qwen3_omni*", "*qwen3_asr*", "*qwen3-coder*", "*qwen-image*", "*diffusion*"]},
        "qwen3-next": {"include": ["*qwen3_next*"], "exclude": []},
        "qwen35": {"include": ["*qwen3_5*", "*qwen35*", "*qwen3.5*", "*qwen3-5*"], "exclude": []},
        "qwen36": {"include": ["*qwen36*", "*qwen3.6*", "*qwen3-6*"], "exclude": []},
        "step35": {"include": ["*step3p5*", "*step-3.5*", "*step35*", "*step3_5*", "*step3_text*", "*step3_vl*"], "exclude": []},
    },
}

SUBJECT_HINTS = {
    "deepseek-v3-r1": ["deepseek-v3", "deepseek v3", "deepseek-r1", "deepseek r1", "deepseekv3", "r1", "dsv3"],
    "deepseek-v31": ["deepseek-v3.1", "deepseek v3.1", "deepseek-v31", "deepseekv31", "v3.1", "v31"],
    "deepseek-v32": ["deepseek-v3.2", "deepseek v3.2", "deepseek-v32", "deepseekv32", "v3.2", "v32", "dsv32", "nsa"],
    "deepseek-v4": ["deepseek-v4", "deepseek v4", "deepseek-v4", "v4"],
    "ernie45": ["ernie45", "ernie-4.5", "ernie 4.5"],
    "gemma4": ["gemma4", "gemma-4", "gemma 4"],
    "glm-vlm-ocr": ["glm-ocr", "glm ocr", "glm4v", "glm-vlm", "glm vlm"],
    "glm45": ["glm-4.5", "glm 4.5", "glm45", "glm4-moe"],
    "glm46-glm47": ["glm-4.6", "glm 4.6", "glm-4.7", "glm 4.7", "glm46", "glm47"],
    "glm5-glm51": ["glm-5", "glm 5", "glm5", "glm-5.1", "glm 5.1", "glm51"],
    "gpt-oss": ["gpt-oss", "gpt oss", "gpt_oss"],
    "hunyuan3-preview": ["hunyuan3", "hy3", "hunyuan 3"],
    "intern-s1": ["intern-s1", "intern s1", "interns1"],
    "internvl35": ["internvl", "intern-vl", "internvl3.5", "internvl35"],
    "kimi": ["kimi", "moonvit"],
    "llama4": ["llama4", "llama-4", "llama 4", "mllama4"],
    "mimo-v2-flash": ["mimo", "mimo-v2", "mimo v2"],
    "minimax": ["minimax", "mini max"],
    "mistral-small-4": ["mistral", "ministral"],
    "mixtral-quark-int4fp8-moe": ["mixtral", "quark", "int4", "fp8", "moe"],
    "moss-vl": ["moss-vl", "moss vl", "moss_vl"],
    "nemotron-super": ["nemotron", "jet-nemotron", "nano-nemotron"],
    "qwen-vlm-omni-asr": ["qwen-vl", "qwen vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl", "omni", "asr", "glmasr", "qwen-audio"],
    "qwen3-coder": ["qwen3-coder", "qwen3 coder"],
    "qwen3-core": ["qwen3", "qwen3-moe", "qwen3 moe"],
    "qwen3-next": ["qwen3-next", "qwen3 next"],
    "qwen35": ["qwen3.5", "qwen35", "qwen3-5"],
    "qwen36": ["qwen3.6", "qwen36", "qwen3-6"],
    "step35": ["step3.5", "step-3.5", "step35", "step3p5"],
}


@dataclass
class TraceInfo:
    files: set[str] = field(default_factory=set)
    commits: set[str] = field(default_factory=set)
    subjects: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)


@dataclass
class PRBundle:
    framework: str
    repo: str
    number: int
    info: dict[str, Any] | None
    files: list[dict[str, Any]]
    trace: TraceInfo
    source_tags: set[str]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True, timeout: int | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def gh_api(path: str, paginate: bool = False) -> Any:
    cmd = ["gh", "api", path]
    if paginate:
        cmd.append("--paginate")
    out = run(cmd, timeout=90)
    if not out.strip():
        return [] if paginate else {}
    if paginate:
        decoder = json.JSONDecoder()
        pos = 0
        values: list[Any] = []
        text = out.strip()
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            value, pos = decoder.raw_decode(text, pos)
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)
        return values
    return json.loads(out)


def load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {"prs": {}}


def save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def framework_commit(framework: str) -> str:
    return run(["git", "rev-parse", "--short=10", "HEAD"], FRAMEWORK_ROOTS[framework]).strip()


def git_files(framework: str) -> list[str]:
    out = run(["git", "ls-files"], FRAMEWORK_ROOTS[framework])
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return [f for f in files if not is_global_excluded(f)]


def is_global_excluded(path: str) -> bool:
    lower = path.lower()
    excluded = [
        "/.git/",
        "node_modules/",
        "site-packages/",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".ipynb",
        ".log",
        ".jsonl",
        ".safetensors",
        "multimodal_gen/",
    ]
    if any(token in lower for token in excluded):
        return True
    if "diffusion" in lower and not ("deepseek" in lower or "qwen3" in lower):
        return True
    if any(token in lower for token in ("qwen-image", "qwen_image", "z-image", "z_image", "zimage", "ltx_video", "ltx23", "hunyuan3d")):
        return True
    return False


def matches_any(path: str, patterns: list[str]) -> bool:
    lower = path.lower()
    return any(fnmatch.fnmatch(lower, pattern.lower()) for pattern in patterns)


def selected_files(framework: str, model: str, all_files: list[str]) -> list[str]:
    filt = MODEL_FILTERS[framework][model]
    include = filt["include"]
    exclude = filt.get("exclude", [])
    files = [
        path
        for path in all_files
        if matches_any(path, include) and not matches_any(path, exclude) and not is_global_excluded(path)
    ]
    # Keep the dossier focused on files that can plausibly affect model support,
    # validation, launch, or documentation.
    wanted_roots = (
        ".github/",
        "benchmark",
        "benchmarks",
        "docs",
        "docs_new",
        "examples",
        "python/",
        "scripts/",
        "sglang/",
        "test",
        "tests",
        "vllm/",
    )
    return sorted({path for path in files if path.startswith(wanted_roots) or "/" not in path})


PR_RE = re.compile(r"(?:\(#|#)(\d{3,6})\)?")


def trace_model_prs(framework: str, files: list[str]) -> dict[int, TraceInfo]:
    if not files:
        return {}
    root = FRAMEWORK_ROOTS[framework]
    selected = set(files)
    merged: dict[int, TraceInfo] = defaultdict(TraceInfo)
    chunk_size = 80
    for start in range(0, len(files), chunk_size):
        chunk = files[start : start + chunk_size]
        out = run(
            [
                "git",
                "log",
                "--date=short",
                "--format=\x02%H\x01%ad\x01%s",
                "--name-only",
                "--",
                *chunk,
            ],
            root,
        )
        commit = commit_date = subject = ""
        touched: list[str] = []

        def flush() -> None:
            if not commit or not subject:
                return
            pr_numbers = [int(m.group(1)) for m in PR_RE.finditer(subject)]
            if not pr_numbers:
                return
            traced_files = [path for path in touched if path in selected]
            if not traced_files:
                traced_files.extend(chunk)
            for number in pr_numbers:
                info = merged[number]
                info.files.update(traced_files)
                info.commits.add(commit[:12])
                info.subjects.add(f"{commit_date} {subject}")
                info.sources.add("git log --name-only")

        for line in out.splitlines():
            if line.startswith("\x02"):
                flush()
                parts = line[1:].split("\x01", 2)
                if len(parts) == 3:
                    commit, commit_date, subject = parts
                    touched = []
                continue
            path = line.strip()
            if path:
                touched.append(path)
        flush()
    return dict(sorted(merged.items()))


def normalize_hint_text(text: str) -> str:
    text = text.lower().replace("_", "-")
    return re.sub(r"\s+", " ", text)


def filter_traces_by_subject(model: str, traces: dict[int, TraceInfo]) -> dict[int, TraceInfo]:
    hints = [normalize_hint_text(hint) for hint in SUBJECT_HINTS.get(model, [])]
    if not hints:
        return traces
    filtered: dict[int, TraceInfo] = {}
    for number, trace in traces.items():
        haystack = normalize_hint_text(" ".join(trace.subjects))
        if any(hint in haystack for hint in hints):
            filtered[number] = trace
    return filtered


def extract_existing_prs(framework: str, model: str) -> set[int]:
    repo = REPOS[framework]
    paths = [
        HISTORY_ROOT / framework / model / "README.zh.md",
        HISTORY_ROOT / framework / model / "README.en.md",
        skill_pr_history_path(framework, model),
        skill_dir(framework, model) / "SKILL.md",
    ]
    numbers: set[int] = set()
    for path in paths:
        relpath = path.relative_to(ROOT)
        text = run(["git", "show", f"main:{relpath.as_posix()}"], ROOT, check=False)
        if not text:
            continue
        for match in re.finditer(rf"https://github\.com/{re.escape(repo)}/pull/(\d+)", text):
            numbers.add(int(match.group(1)))
    return numbers


def fetch_pr_bundle(framework: str, number: int, cache: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    repo = REPOS[framework]
    key = f"{repo}#{number}"
    if key in cache["prs"]:
        entry = cache["prs"][key]
        return entry.get("info"), entry.get("files", [])
    try:
        info = gh_api(f"repos/{repo}/pulls/{number}")
        files = gh_api(f"repos/{repo}/pulls/{number}/files?per_page=100", paginate=True)
    except Exception as exc:  # keep the rebuild resilient to deleted private refs
        info = {"number": number, "html_url": f"https://github.com/{repo}/pull/{number}", "title": f"unavailable PR #{number}", "state": "unknown", "fetch_error": str(exc)}
        files = []
    cache["prs"][key] = {"info": info, "files": files}
    return info, files


def fetch_many(framework: str, numbers: set[int], traces: dict[int, TraceInfo], source_tags: dict[int, set[str]], cache: dict[str, Any]) -> list[PRBundle]:
    bundles: list[PRBundle] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {executor.submit(fetch_pr_bundle, framework, number, cache): number for number in numbers}
        for future in concurrent.futures.as_completed(future_map):
            number = future_map[future]
            info, files = future.result()
            bundles.append(
                PRBundle(
                    framework=framework,
                    repo=REPOS[framework],
                    number=number,
                    info=info,
                    files=files,
                    trace=traces.get(number, TraceInfo()),
                    source_tags=source_tags.get(number, set()),
                )
            )
    return sorted(bundles, key=sort_key)


def sort_key(bundle: PRBundle) -> tuple[str, int]:
    info = bundle.info or {}
    when = info.get("merged_at") or info.get("closed_at") or info.get("created_at") or ""
    return (when, bundle.number)


def skill_dir(framework: str, model: str) -> Path:
    if framework == "sglang" and model == "kimi":
        name = "sglang-kimi-k2-k25-optimization"
    elif framework == "sglang" and model == "minimax":
        name = "sglang-minimax-m2-series-optimization"
    else:
        name = f"{framework}-{model}-optimization"
    return SKILL_ROOT / framework / name


def skill_pr_history_path(framework: str, model: str) -> Path:
    return skill_dir(framework, model) / "references" / "pr-history.md"


def md_escape(text: str) -> str:
    text = text or ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", "\\|")
    return re.sub(r"\s+", " ", text).strip()


def clean_block(text: str) -> str:
    """Remove template indentation without changing embedded diff lines."""
    lines = text.splitlines()
    cleaned = [line[8:] if line.startswith("        ") else line for line in lines]
    return "\n".join(cleaned).strip()


def plain(text: str, limit: int = 220) -> str:
    text = re.sub(r"`{3,}.*?`{3,}", " ", text or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", lambda m: m.group(0).split("](")[0].lstrip("["), text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def body_excerpt(text: str, limit: int = 220) -> str:
    """Keep useful PR-body context and drop repository checklist boilerplate."""
    text = re.sub(r"`{3,}.*?`{3,}", " ", text or "", flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    cutoff_markers = [
        "Essential Elements of an Effective PR Description Checklist",
        "Format your code according to",
        "\n## Checklist",
        "\n### Checklist",
        "\n## TODO",
        "\n### TODO",
        "\nTODO:",
        "\n**TODO",
        "\n---\nEssential Elements",
    ]
    lower = text.lower()
    cutoffs = [lower.find(marker.lower()) for marker in cutoff_markers if lower.find(marker.lower()) >= 0]
    if cutoffs:
        text = text[: min(cutoffs)]

    useful_lines: list[str] = []
    heading_only = re.compile(
        r"^#{1,6}\s*(motivation|modifications?|accuracy tests?|speed tests?.*|"
        r"benchmark(?:ing)?(?: and profiling)?|test plan|test result|purpose|summary)\s*:?\s*$",
        re.I,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"[#>*_\-\s.]+", line):
            continue
        if heading_only.match(line):
            continue
        if re.match(r"^[-*]\s*\[[ xX]\]", line):
            continue
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        line = re.sub(r"^DRAFT:\s*", "", line, flags=re.I)
        useful_lines.append(line)

    cleaned = plain(" ".join(useful_lines), limit)
    cleaned = re.sub(r"[\U0001F000-\U0001FAFF\U00002700-\U000027BF]+", "", cleaned).strip()
    cleaned = re.sub(r"\s+(---|--|-)$", "", cleaned).strip()
    empty_templates = {
        "## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling",
        "## Purpose ## Test Plan ## Test Result",
    }
    if (
        not cleaned
        or re.fullmatch(r"[#>*_\-\s.]+", cleaned)
        or any(cleaned.lower() == template.lower() for template in empty_templates)
    ):
        return ""
    return cleaned


def file_status(file: dict[str, Any]) -> str:
    return file.get("status") or "modified"


def patch_lines(file: dict[str, Any]) -> list[str]:
    patch = file.get("patch") or ""
    return patch.splitlines()


def changed_line_count(file: dict[str, Any]) -> int:
    return len(patch_lines(file))


def is_relevant_file(bundle: PRBundle, filename: str) -> bool:
    if filename in bundle.trace.files:
        return True
    lower = filename.lower()
    for traced in bundle.trace.files:
        base = Path(traced).name.lower()
        if base and base in lower:
            return True
    return False


def top_files(bundle: PRBundle, limit: int = 8) -> list[dict[str, Any]]:
    files = [file for file in (bundle.files or []) if not is_global_excluded(file.get("filename", ""))]
    relevant = [f for f in files if is_relevant_file(bundle, f.get("filename", ""))]
    chosen = relevant or files
    chosen = sorted(
        chosen,
        key=lambda item: (0 if is_runtime_file(item.get("filename", "")) else 1, -int(item.get("changes") or 0)),
    )
    return chosen[:limit]


def is_unusable_or_disallowed_bundle(bundle: PRBundle) -> bool:
    info = bundle.info or {}
    if info.get("fetch_error") and not bundle.files:
        return True
    title = normalize_hint_text(info.get("title") or "")
    disallowed_title = ("[diffusion]" in title or "zimage" in title or "z-image" in title or "qwen-image" in title or "ltx" in title)
    if disallowed_title:
        return True
    files = bundle.files or []
    if files and all(is_global_excluded(file.get("filename", "")) for file in files):
        return True
    return False


def is_runtime_file(path: str) -> bool:
    lower = path.lower()
    return any(token in lower for token in ("/models/", "model_executor", "/layers/", "/configs/", "/processors/", "tool_parser", "reasoning", "tokenizer", "entrypoints", "multimodal", "function_call"))


def file_category(path: str) -> str:
    lower = path.lower()
    if "test/" in lower or "tests/" in lower:
        return "tests"
    if lower.startswith("docs") or "/docs" in lower or lower.startswith("examples") or "cookbook" in lower:
        return "docs"
    if is_runtime_file(path) or lower.startswith("python/") or lower.startswith("vllm/"):
        return "runtime"
    if ".github" in lower or "workflow" in lower:
        return "ci"
    return "other"


def group_files(files: list[dict[str, Any]], limit: int = 6) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for file in files:
        path = file.get("filename", "")
        grouped[file_category(path)].append(
            f"`{path}` {file_status(file)} +{file.get('additions', 0)}/-{file.get('deletions', 0)}"
        )
    return {key: values[:limit] for key, values in grouped.items()}


def symbols_from_patch(file: dict[str, Any], limit: int = 5) -> list[str]:
    symbols: list[str] = []
    for line in patch_lines(file):
        raw = line[1:].strip() if line[:1] in {"+", "-"} else line.strip()
        match = re.search(r"\b(class|def|async def)\s+([A-Za-z_][A-Za-z0-9_]*)", raw)
        if match:
            symbols.append(match.group(2))
        if len(symbols) >= limit:
            break
    return list(dict.fromkeys(symbols))


def digest_file(file: dict[str, Any]) -> str:
    path = file.get("filename", "")
    symbols = symbols_from_patch(file)
    hunk_headers = [line.strip("@ ") for line in patch_lines(file) if line.startswith("@@")]
    hunk_text = "; hunks: " + "; ".join(plain(h, 80) for h in hunk_headers[:2]) if hunk_headers else ""
    symbol_text = "; symbols: " + ", ".join(symbols[:4]) if symbols else ""
    return (
        f"`{path}` {file_status(file)} +{file.get('additions', 0)}/-{file.get('deletions', 0)} "
        f"({file.get('changes', 0)} lines){hunk_text}{symbol_text}"
    )


def diff_snippet(files: list[dict[str, Any]], max_lines: int = 18) -> str:
    snippets: list[str] = []
    seen_files: set[str] = set()
    for file in files:
        filename = file.get("filename", "")
        if filename in seen_files:
            continue
        seen_files.add(filename)
        patch = patch_lines(file)
        if not patch:
            continue
        snippets.append(f"diff -- {filename}")
        picked: list[str] = []
        for line in patch:
            if line.startswith(("+++", "---")):
                continue
            if line.startswith("@@") or (line[:1] in {"+", "-"} and line[1:].strip()):
                picked.append(line[:180].rstrip())
            if len(picked) >= 7:
                break
        snippets.extend(picked)
        if len(snippets) >= max_lines:
            break
    text = "\n".join(snippets[:max_lines])
    text = text.replace("```", "'''")
    return text or "No textual patch was returned by GitHub for the selected changed files."


def total_patch_lines(files: list[dict[str, Any]]) -> int:
    return sum(changed_line_count(file) for file in files)


def pr_state(bundle: PRBundle) -> str:
    info = bundle.info or {}
    if info.get("merged_at"):
        return "merged"
    return info.get("state") or "unknown"


def pr_when(bundle: PRBundle) -> str:
    info = bundle.info or {}
    return (info.get("merged_at") or info.get("closed_at") or info.get("created_at") or "")[:10] or "unknown"


def pr_title(bundle: PRBundle) -> str:
    return plain((bundle.info or {}).get("title") or f"PR #{bundle.number}", 160)


def pr_url(bundle: PRBundle) -> str:
    return (bundle.info or {}).get("html_url") or f"https://github.com/{bundle.repo}/pull/{bundle.number}"


def source_text_zh(bundle: PRBundle) -> str:
    parts: list[str] = []
    if bundle.trace.files:
        files = ", ".join(f"`{f}`" for f in sorted(bundle.trace.files)[:5])
        suffix = "" if len(bundle.trace.files) <= 5 else f" 等 {len(bundle.trace.files)} 个文件"
        parts.append(f"`git log --name-only -- <model-files>` 反查到 {files}{suffix}")
    if bundle.trace.commits:
        parts.append("关联提交 " + ", ".join(f"`{c}`" for c in sorted(bundle.trace.commits)[:5]))
    if "existing-doc" in bundle.source_tags:
        parts.append("保留自原 history/skill 显式引用")
    return "；".join(parts) or "保留自既有文档引用，当前实现文件未直接追到该 PR"


def source_text_en(bundle: PRBundle) -> str:
    parts: list[str] = []
    if bundle.trace.files:
        files = ", ".join(f"`{f}`" for f in sorted(bundle.trace.files)[:5])
        suffix = "" if len(bundle.trace.files) <= 5 else f" and {len(bundle.trace.files)} files"
        parts.append(f"`git log --name-only -- <model-files>` found it through {files}{suffix}")
    if bundle.trace.commits:
        parts.append("associated commits " + ", ".join(f"`{c}`" for c in sorted(bundle.trace.commits)[:5]))
    if "existing-doc" in bundle.source_tags:
        parts.append("preserved from an explicit existing history/skill citation")
    return "; ".join(parts) or "Preserved from existing docs; not directly traced from current implementation files"


def motivation_zh(bundle: PRBundle, model_title: str) -> str:
    title = pr_title(bundle)
    body = body_excerpt((bundle.info or {}).get("body") or "", 180)
    files = top_files(bundle, 4)
    names = ", ".join(f"`{f.get('filename')}`" for f in files[:3]) or "文件级 diff"
    lower = f"{title} {body}".lower()
    if any(token in lower for token in ("fix", "bug", "wrong", "error", "crash", "regression")):
        category = "缺陷修复"
    elif any(token in lower for token in ("perf", "opt", "fuse", "fp8", "fp4", "deepgemm", "deepep", "flash", "aiter", "cuda", "triton")):
        category = "性能/后端优化"
    elif any(token in lower for token in ("doc", "cookbook", "test", "ci", "benchmark")):
        category = "文档/测试/CI"
    elif any(token in lower for token in ("support", "add ", "introduce", "initial", "enable")):
        category = "模型支持/运行时入口"
    else:
        category = "模型实现调整"
    body_part = f"；PR 正文摘要: {body}" if body else "；PR 正文未提供可用摘要"
    return f"标题「{title}」；模型线: {model_title}；类别: {category}；主要 diff: {names}{body_part}。"


def implementation_zh(bundle: PRBundle) -> str:
    files = top_files(bundle, 5)
    if not files:
        return "GitHub 没有返回文件级 patch；本卡仅保留 PR 元数据，后续审计应重新打开 PR 页面确认最终 diff。"
    pieces: list[str] = []
    for file in files[:4]:
        symbols = symbols_from_patch(file)
        symbol_text = f"，涉及 `{', '.join(symbols[:3])}`" if symbols else ""
        pieces.append(f"{digest_file(file)}{symbol_text}")
    return "；".join(pieces) + "。"


def validation_zh(bundle: PRBundle) -> str:
    files = bundle.files or []
    tests = [f.get("filename", "") for f in files if file_category(f.get("filename", "")) == "tests"]
    docs = [f.get("filename", "") for f in files if file_category(f.get("filename", "")) == "docs"]
    runtime = [f.get("filename", "") for f in files if file_category(f.get("filename", "")) == "runtime"]
    if tests:
        names = ", ".join(f"`{p}`" for p in tests[:4])
        return f"diff 自带测试面 {names}；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。"
    if runtime:
        names = ", ".join(f"`{p}`" for p in runtime[:3])
        return f"runtime 路径改动集中在 {names}；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。"
    if docs:
        names = ", ".join(f"`{p}`" for p in docs[:3])
        return f"该 PR 主要落在文档/示例 {names}；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。"
    return "未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。"


def motivation_en(bundle: PRBundle, model_title: str) -> str:
    title = pr_title(bundle)
    body = body_excerpt((bundle.info or {}).get("body") or "", 180)
    files = top_files(bundle, 4)
    names = ", ".join(f"`{f.get('filename')}`" for f in files[:3]) or "the file-level diff"
    lower = f"{title} {body}".lower()
    if any(token in lower for token in ("fix", "bug", "wrong", "error", "crash", "regression")):
        category = "bug fix"
    elif any(token in lower for token in ("perf", "opt", "fuse", "fp8", "fp4", "deepgemm", "deepep", "flash", "aiter", "cuda", "triton")):
        category = "performance/backend optimization"
    elif any(token in lower for token in ("doc", "cookbook", "test", "ci", "benchmark")):
        category = "docs/tests/CI"
    elif any(token in lower for token in ("support", "add ", "introduce", "initial", "enable")):
        category = "model support/runtime entry"
    else:
        category = "model implementation change"
    body_part = f"; PR body summary: {body}" if body else "; no usable PR-body summary"
    return f"Title: \"{title}\"; model line: {model_title}; category: {category}; main diff: {names}{body_part}."


def implementation_en(bundle: PRBundle) -> str:
    files = top_files(bundle, 5)
    if not files:
        return "GitHub returned no file-level patch; this card keeps the PR metadata and should be re-opened manually before code changes depend on it."
    pieces: list[str] = []
    for file in files[:4]:
        symbols = symbols_from_patch(file)
        symbol_text = f", touching `{', '.join(symbols[:3])}`" if symbols else ""
        pieces.append(f"{digest_file(file)}{symbol_text}")
    return "; ".join(pieces) + "."


def validation_en(bundle: PRBundle) -> str:
    files = bundle.files or []
    tests = [f.get("filename", "") for f in files if file_category(f.get("filename", "")) == "tests"]
    docs = [f.get("filename", "") for f in files if file_category(f.get("filename", "")) == "docs"]
    runtime = [f.get("filename", "") for f in files if file_category(f.get("filename", "")) == "runtime"]
    if tests:
        names = ", ".join(f"`{p}`" for p in tests[:4])
        return f"The diff ships test coverage in {names}; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke."
    if runtime:
        names = ", ".join(f"`{p}`" for p in runtime[:3])
        return f"Runtime changes concentrate in {names}; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output."
    if docs:
        names = ", ".join(f"`{p}`" for p in docs[:3])
        return f"This is mostly docs/examples in {names}; validation should confirm the documented command still maps to current CLI flags and model repo names."
    return "No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks."


def card_zh(bundle: PRBundle, model_title: str) -> str:
    files = top_files(bundle, 8)
    total_files = len(bundle.files)
    additions = sum(int(f.get("additions") or 0) for f in bundle.files)
    deletions = sum(int(f.get("deletions") or 0) for f in bundle.files)
    patch_count = total_patch_lines(bundle.files)
    groups = group_files(files)
    grouped_text = "\n".join(
        f"  - {key}: " + "; ".join(values)
        for key, values in groups.items()
    ) or "  - GitHub 未返回文件级 patch。"
    digest = "\n".join(f"  - {digest_file(f)}" for f in files[:5]) or "  - GitHub 未返回文件级 patch。"
    return clean_block(
        f"""\
        ### PR #{bundle.number} - {pr_title(bundle)}

        - 链接: {pr_url(bundle)}
        - 状态/时间: {pr_state(bundle)} / {pr_when(bundle)}
        - 反查来源: {source_text_zh(bundle)}
        - 代码 diff 已读范围: GitHub Pull Request files API 返回 {total_files} 个文件，+{additions}/-{deletions}，可读 patch {patch_count} 行；本卡优先审计模型相关文件和高变更量文件。
        - 动机: {motivation_zh(bundle, model_title)}
        - 实现要点: {implementation_zh(bundle)}
        - 代码 diff 细节:
        {digest}
        - 关键代码摘录:

        ```diff
        {diff_snippet(files)}
        ```

        - 已读文件:
        {grouped_text}
        - 验证与风险: {validation_zh(bundle)}
        """
    )


def card_en(bundle: PRBundle, model_title: str) -> str:
    files = top_files(bundle, 8)
    total_files = len(bundle.files)
    additions = sum(int(f.get("additions") or 0) for f in bundle.files)
    deletions = sum(int(f.get("deletions") or 0) for f in bundle.files)
    patch_count = total_patch_lines(bundle.files)
    groups = group_files(files)
    grouped_text = "\n".join(
        f"  - {key}: " + "; ".join(values)
        for key, values in groups.items()
    ) or "  - No patch file list returned by GitHub."
    digest = "\n".join(f"  - {digest_file(f)}" for f in files[:5]) or "  - No patch file list returned by GitHub."
    return clean_block(
        f"""\
        ### PR #{bundle.number} - {pr_title(bundle)}

        - Link: {pr_url(bundle)}
        - Status/date: {pr_state(bundle)} / {pr_when(bundle)}
        - Trace source: {source_text_en(bundle)}
        - Diff scope read: GitHub Pull Request files API returned {total_files} files, +{additions}/-{deletions}, {patch_count} readable patch lines; this card prioritizes model-related and high-change files.
        - Motivation: {motivation_en(bundle, model_title)}
        - Key implementation: {implementation_en(bundle)}
        - Code diff details:
        {digest}
        - Key code excerpts:

        ```diff
        {diff_snippet(files)}
        ```

        - Reviewed files:
        {grouped_text}
        - Risk and verification: {validation_en(bundle)}
        """
    )


def coverage_table_zh(files: list[str], traces: dict[int, TraceInfo]) -> str:
    rows = ["| 文件 | git 追溯到的 PR |", "| --- | --- |"]
    for relpath in files:
        nums = [num for num, trace in traces.items() if relpath in trace.files]
        pr_text = ", ".join(f"[#{num}](https://github.com/{{repo}}/pull/{num})" for num in nums[:8])
        rows.append(f"| `{relpath}` | {pr_text or '无直接 PR 号提交'} |")
    return "\n".join(rows).replace("https://github.com/{repo}", "https://github.com/{repo}")


def timeline_zh(bundles: list[PRBundle]) -> str:
    rows = ["| 日期 | PR | 状态 | 标题 | 主要文件 |", "| --- | --- | --- | --- | --- |"]
    for bundle in bundles:
        files = ", ".join(f"`{f.get('filename')}`" for f in top_files(bundle, 3))
        rows.append(f"| {pr_when(bundle)} | [#{bundle.number}]({pr_url(bundle)}) | {pr_state(bundle)} | {md_escape(pr_title(bundle))} | {md_escape(files)} |")
    return "\n".join(rows)


def timeline_en(bundles: list[PRBundle]) -> str:
    rows = ["| Date | PR | State | Title | Main files |", "| --- | --- | --- | --- | --- |"]
    for bundle in bundles:
        files = ", ".join(f"`{f.get('filename')}`" for f in top_files(bundle, 3))
        rows.append(f"| {pr_when(bundle)} | [#{bundle.number}]({pr_url(bundle)}) | {pr_state(bundle)} | {md_escape(pr_title(bundle))} | {md_escape(files)} |")
    return "\n".join(rows)


def coverage_table(files: list[str], traces: dict[int, TraceInfo], repo: str, lang: str) -> str:
    header = "| 文件 | git 追溯到的 PR |" if lang == "zh" else "| File | Git-traced PRs |"
    rows = [header, "| --- | --- |"]
    if not files:
        empty = "当前主线没有匹配到实现文件" if lang == "zh" else "No matching implementation file on current main"
        return "\n".join(rows + [f"| - | {empty} |"])
    for relpath in files[:80]:
        nums = [num for num, trace in traces.items() if relpath in trace.files]
        pr_text = ", ".join(f"[#{num}](https://github.com/{repo}/pull/{num})" for num in nums[:12])
        if len(nums) > 12:
            pr_text += f", ... ({len(nums)} total)"
        rows.append(f"| `{relpath}` | {pr_text or ('无直接 PR 号提交' if lang == 'zh' else 'no direct PR-number commit')} |")
    if len(files) > 80:
        rows.append(f"| ... | {len(files) - 80} more files omitted from table; all were used for git tracing. |")
    return "\n".join(rows)


def no_pr_section_zh(model_title: str, framework: str, files: list[str]) -> str:
    file_note = "；".join(f"`{p}`" for p in files[:8]) if files else "当前主线没有匹配到实现文件"
    return clean_block(
        f"""\
        ## 逐 PR diff 审计卡

        ### 公开 PR 检索结论

        - 结论: 未确认到可归入 {framework} {model_title} 模型支持或优化主线的公开 PR。
        - 检索方式: 已对匹配文件执行 `git log --name-only -- <model-files>`，并检查原 history/skill 中的显式 PR 链接。
        - 覆盖文件: {file_note}
        - 后续验收: 如果该模型后续出现实现文件或 PR，必须补齐同一模板的逐 PR diff 卡。
        """
    )


def no_pr_section_en(model_title: str, framework: str, files: list[str]) -> str:
    file_note = "; ".join(f"`{p}`" for p in files[:8]) if files else "no matching implementation file on current main"
    return clean_block(
        f"""\
        ## Per-PR Diff Audit Cards

        ### Public PR search conclusion

        - Conclusion: No public PR was confirmed as part of the {framework} {model_title} model support or optimization line.
        - Search method: `git log --name-only -- <model-files>` was run on the matched files, and explicit PR links in the previous history/skill were checked.
        - Covered files: {file_note}
        - Acceptance rule: if implementation files or PRs appear later, add the same per-PR diff card format.
        """
    )


def render_history_zh(framework: str, model: str, files: list[str], traces: dict[int, TraceInfo], bundles: list[PRBundle], existing_only_count: int) -> str:
    title = MODEL_TITLES[model]
    repo = REPOS[framework]
    commit = framework_commit(framework)
    cards = "\n\n".join(card_zh(bundle, title) for bundle in bundles)
    pr_section = f"## 逐 PR diff 审计卡\n\n{cards}" if cards else no_pr_section_zh(title, framework, files)
    traced_count = sum(1 for bundle in bundles if bundle.trace.files)
    timeline = (
        timeline_zh(bundles)
        if bundles
        else "| 日期 | PR | 状态 | 标题 | 主要文件 |\n| --- | --- | --- | --- | --- |\n| - | - | - | 未发现可归档 PR | - |"
    )
    body = clean_block(
        f"""\
        # {framework} {title} 模型 PR 优化历史

        ## 文档口径

        - 重做日期: {TODAY}
        - 源码基线: `{repo}` 当前追溯 worktree commit `{commit}`
        - PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
        - 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

        ## 模型实现文件覆盖

        {coverage_table(files, traces, repo, "zh")}

        ## PR 覆盖总览

        - git 追溯 PR 数: {traced_count}
        - 原文档显式引用补充 PR 数: {existing_only_count}
        - 当前文档总 PR 数: {len(bundles)}
        - 文件追溯命令: `git log --name-only -- <model-files>`
        - diff 审计来源: GitHub Pull Request files API

        ## 时间线

        {timeline}

        {pr_section}

        ## 补漏结论

        - 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
        - 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
        """
    )
    return body.strip() + "\n"


def render_history_en(framework: str, model: str, files: list[str], traces: dict[int, TraceInfo], bundles: list[PRBundle], existing_only_count: int) -> str:
    title = MODEL_TITLES[model]
    repo = REPOS[framework]
    commit = framework_commit(framework)
    cards = "\n\n".join(card_en(bundle, title) for bundle in bundles)
    pr_section = f"## Per-PR Diff Audit Cards\n\n{cards}" if cards else no_pr_section_en(title, framework, files)
    traced_count = sum(1 for bundle in bundles if bundle.trace.files)
    timeline = (
        timeline_en(bundles)
        if bundles
        else "| Date | PR | State | Title | Main files |\n| --- | --- | --- | --- | --- |\n| - | - | - | no archived PR found | - |"
    )
    body = clean_block(
        f"""\
        # {framework} {title} Model PR Optimization History

        ## Scope

        - Rebuilt on: {TODAY}
        - Source baseline: `{repo}` trace worktree commit `{commit}`
        - PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
        - Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

        ## Implementation File Coverage

        {coverage_table(files, traces, repo, "en")}

        ## PR Coverage Summary

        - Git-traced PRs: {traced_count}
        - Extra PRs preserved from existing docs: {existing_only_count}
        - Total PRs in this document: {len(bundles)}
        - File trace command: `git log --name-only -- <model-files>`
        - Diff audit source: GitHub Pull Request files API

        ## Timeline

        {timeline}

        {pr_section}

        ## Gap-Closure Notes

        - Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
        - If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
        """
    )
    return body.strip() + "\n"


def render_reference(framework: str, model: str, files: list[str], traces: dict[int, TraceInfo], bundles: list[PRBundle], existing_only_count: int) -> str:
    title = MODEL_TITLES[model]
    repo = REPOS[framework]
    commit = framework_commit(framework)
    cards = "\n\n".join(card_en(bundle, title) for bundle in bundles)
    pr_section = f"## Per-PR Diff Audit Cards\n\n{cards}" if cards else no_pr_section_en(title, framework, files)
    timeline = (
        timeline_en(bundles)
        if bundles
        else "| Date | PR | State | Title | Main files |\n| --- | --- | --- | --- | --- |\n| - | - | - | no archived PR found | - |"
    )
    body = clean_block(
        f"""\
        # {framework} {title} PR Diff Audit Reference

        - Rebuilt on: {TODAY}
        - Source baseline: `{repo}` trace worktree commit `{commit}`
        - Collection: model implementation files were traced with `git log --name-only -- <model-files>`, filtered by model keywords in commit subjects, then every PR card was populated from the GitHub Pull Request files API.
        - Extra preserved PRs from prior docs: {existing_only_count}
        - Rule: use this evidence file before changing model-specific skill guidance; it is not only PR titles.

        ## Implementation File Coverage

        {coverage_table(files, traces, repo, "en")}

        ## Timeline

        {timeline}

        {pr_section}
        """
    )
    return body.strip() + "\n"


def update_indexes() -> None:
    model_lines = "\n".join(f"- `{model}`" for model in MODEL_ORDER)
    (HISTORY_ROOT / "sglang" / "README.md").write_text(
        f"# SGLang Model PR Optimization History\n\nCurrent model families:\n\n{model_lines}\n",
        encoding="utf-8",
    )
    (HISTORY_ROOT / "vllm" / "README.md").write_text(
        f"# vLLM Model PR Optimization History\n\nCurrent model families:\n\n{model_lines}\n",
        encoding="utf-8",
    )
    sglang_skill_lines = "\n".join(f"- `{skill_dir('sglang', model).name}`" for model in MODEL_ORDER)
    vllm_skill_lines = "\n".join(f"- `{skill_dir('vllm', model).name}`" for model in MODEL_ORDER)
    (SKILL_ROOT / "sglang" / "README.md").write_text(
        f"# SGLang Model Optimization Skills\n\nCurrent SGLang model skills:\n\n{sglang_skill_lines}\n",
        encoding="utf-8",
    )
    (SKILL_ROOT / "vllm" / "README.md").write_text(
        f"# vLLM Model Optimization Skills\n\nCurrent vLLM model skills:\n\n{vllm_skill_lines}\n",
        encoding="utf-8",
    )


def ensure_skill_entry(framework: str, model: str) -> None:
    path = skill_dir(framework, model) / "SKILL.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    additions: list[str] = []
    if "references/pr-history.md" not in text:
        additions.append("- Use `references/pr-history.md` as the audited PR dossier before changing this model path.")
    if "model-pr-diff-dossier" not in text:
        additions.append("- Follow the shared `model-pr-diff-dossier` standard for every cited PR.")
    if "not only PR titles" not in text:
        additions.append("- PR evidence must be diff-backed, not only PR titles.")
    if additions:
        text = text.rstrip() + "\n\n## PR Dossier Standard\n\n" + "\n".join(additions) + "\n"
        path.write_text(text, encoding="utf-8")


def rebuild(dry_run: bool = False) -> None:
    cache = load_cache()
    all_files = {framework: git_files(framework) for framework in REPOS}
    update_indexes()
    for framework in ("sglang", "vllm"):
        print(f"== {framework} ==")
        for model in MODEL_ORDER:
            files = selected_files(framework, model, all_files[framework])
            raw_traces = trace_model_prs(framework, files)
            traces = filter_traces_by_subject(model, raw_traces)
            existing = extract_existing_prs(framework, model)
            source_tags: dict[int, set[str]] = defaultdict(set)
            for number in traces:
                source_tags[number].add("git-trace")
            for number in existing:
                source_tags[number].add("existing-doc")
            numbers = set(traces) | existing
            existing_only_count = sum(1 for number in existing if number not in traces)
            if dry_run:
                print(
                    f"{framework}/{model}: files={len(files)} git_prs={len(traces)} "
                    f"raw_git_prs={len(raw_traces)} existing_extra={existing_only_count} total_candidates={len(numbers)}",
                    flush=True,
                )
                continue
            print(
                f"{framework}/{model}: files={len(files)} git_prs={len(traces)} "
                f"raw_git_prs={len(raw_traces)} existing_extra={existing_only_count} fetching={len(numbers)}",
                flush=True,
            )
            bundles = [
                bundle
                for bundle in fetch_many(framework, numbers, traces, source_tags, cache)
                if not is_unusable_or_disallowed_bundle(bundle)
            ]
            print(f"{framework}/{model}: fetched total={len(bundles)}", flush=True)
            model_dir = HISTORY_ROOT / framework / model
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "README.zh.md").write_text(
                render_history_zh(framework, model, files, traces, bundles, existing_only_count),
                encoding="utf-8",
            )
            (model_dir / "README.en.md").write_text(
                render_history_en(framework, model, files, traces, bundles, existing_only_count),
                encoding="utf-8",
            )
            ref = skill_pr_history_path(framework, model)
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(
                render_reference(framework, model, files, traces, bundles, existing_only_count),
                encoding="utf-8",
            )
            ensure_skill_entry(framework, model)
            save_cache(cache)
    save_cache(cache)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    missing = [str(path) for path in FRAMEWORK_ROOTS.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing framework worktrees: {', '.join(missing)}")
    rebuild(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
