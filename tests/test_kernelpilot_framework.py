from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNELPILOT_ROOT = ROOT / "kernel-pilot"

sys.path.insert(0, str(KERNELPILOT_ROOT))

from kernelpilot.config import load_taskset  # noqa: E402
from kernelpilot.knowledge import load_catalog, render_catalog_for_prompt  # noqa: E402
from kernelpilot.lineage import LineageEntry, LineageStore  # noqa: E402
from kernelpilot.prompting import build_iteration_prompt  # noqa: E402
from kernelpilot.scoring import score_against_baseline  # noqa: E402
from kernelpilot.supervisor import is_stagnated  # noqa: E402


def write_sample_taskset(tmp_path: Path) -> Path:
    payload = {
        "name": "sample-kernel-taskset",
        "base_repo": "<target-kernel-repository>",
        "base_ref": "origin/main",
        "paper_url": "https://arxiv.org/abs/2603.24517",
        "global_rules": [
            "Use a single-lineage KernelPilot loop.",
            "Start candidates from the semantic spec.",
            "Correctness failure gives score zero regardless of latency.",
        ],
        "tasks": [
            {
                "id": "add_constant_large_int32",
                "title": "Optimize large add_constant tensors",
                "baseline_id": "baseline-add-constant",
                "baseline_url": "<baseline-source>",
                "operator": "add_constant",
                "lane": "CUDA C++ JIT",
                "semantic_spec": (
                    "Add a constant to every element while preserving exact "
                    "integer arithmetic."
                ),
                "forbidden_files": ["src/kernels/add_constant.cuh"],
                "correctness": [
                    "Exact equality against x + constant over selected sizes."
                ],
                "benchmark_command": "python benchmarks/bench_add_constant.py",
                "baseline": [
                    {"case_id": "n=1048576", "baseline_latency_us": 7.456},
                    {"case_id": "n=4194304", "baseline_latency_us": 11.632},
                    {"case_id": "n=16777216", "baseline_latency_us": 49.712},
                ],
                "search_space": [
                    "vectorized int32 memory path",
                    "alignment and size threshold",
                ],
            }
        ],
    }
    taskset_path = tmp_path / "taskset.json"
    taskset_path.write_text(json.dumps(payload), encoding="utf-8")
    return taskset_path


def test_taskset_loads_generic_kernel_baseline(tmp_path: Path) -> None:
    taskset = load_taskset(write_sample_taskset(tmp_path))
    assert len(taskset.tasks) == 1
    assert taskset.tasks[0].baseline_id == "baseline-add-constant"
    for task in taskset.tasks:
        assert task.forbidden_files
        assert task.baseline
        assert task.correctness


def test_prompt_enforces_from_scratch_kernel_constraint(tmp_path: Path) -> None:
    taskset = load_taskset(write_sample_taskset(tmp_path))
    task = taskset.task_by_id("add_constant_large_int32")
    prompt = build_iteration_prompt(taskset, task, [])
    assert "Start candidates from the semantic spec" in prompt
    assert "src/kernels/add_constant.cuh" in prompt
    assert "Knowledge base K to consult before edits" in prompt
    assert "gpu-kernel-ako4all references" in prompt
    assert "No committed KernelPilot candidate exists yet" in prompt


def test_score_requires_correctness_and_full_case_coverage(tmp_path: Path) -> None:
    task = load_taskset(write_sample_taskset(tmp_path)).task_by_id(
        "add_constant_large_int32"
    )
    faster = {
        "correct": True,
        "measurements": [
            {"case_id": "n=1048576", "latency_us": 7.0},
            {"case_id": "n=4194304", "latency_us": 10.0},
            {"case_id": "n=16777216", "latency_us": 45.0},
        ],
    }
    summary = score_against_baseline(task, faster)
    assert summary.correct
    assert summary.matched_cases == 3
    assert not summary.missing_cases
    assert summary.geomean_vs_baseline > 1.05
    assert summary.clearly_better_than_baseline

    incomplete = {"correct": True, "measurements": faster["measurements"][:1]}
    incomplete_summary = score_against_baseline(task, incomplete)
    assert incomplete_summary.missing_cases
    assert not incomplete_summary.clearly_better_than_baseline

    wrong = dict(faster)
    wrong["correct"] = False
    wrong_summary = score_against_baseline(task, wrong)
    assert wrong_summary.geomean_vs_baseline == 0.0
    assert not wrong_summary.clearly_better_than_baseline

    mixed = {
        "correct": True,
        "measurements": [
            {"case_id": "n=1048576", "latency_us": 2.0},
            {"case_id": "n=4194304", "latency_us": 8.0},
            {"case_id": "n=16777216", "latency_us": 90.0},
        ],
    }
    mixed_summary = score_against_baseline(task, mixed)
    assert mixed_summary.geomean_vs_baseline > 1.05
    assert not mixed_summary.clearly_better_than_baseline


def test_lineage_store_and_stagnation_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = LineageStore(tmp, "task")
        for idx, score in enumerate([1.0, 1.01, 1.0101, 1.0102, 1.0103], start=1):
            store.append(
                LineageEntry(
                    version=f"v{idx}",
                    task_id="task",
                    candidate_ref=f"candidate-{idx}",
                    score={"geomean_vs_baseline": score},
                    notes="",
                )
            )

        entries = store.load()
        assert len(entries) == 5
        assert store.best() is not None
        assert store.best().version == "v5"
        assert is_stagnated(entries, window=3, min_relative_gain=0.02)


def test_cli_module_is_importable() -> None:
    from kernelpilot import cli

    parser = cli.build_parser()
    args = parser.parse_args(["validate", "--taskset", "<taskset.json>"])
    assert args.taskset == "<taskset.json>"


def test_kernel_source_catalog_lists_confirmed_and_rejected_sources() -> None:
    catalog = load_catalog()
    rendered = render_catalog_for_prompt(catalog, lane="CUDA C++ JIT")
    assert "FlashInfer" in rendered
    assert "FlashAttention" in rendered
    assert "hi-ops / HPC Ops" in rendered
    assert "DeepGEMM" in rendered
    assert "target kernel project worktree" in rendered
