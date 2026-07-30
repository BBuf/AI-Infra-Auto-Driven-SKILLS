from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

import sgl_engine_sglang_diffusion.runtime as runtime_module
from sgl_engine_sglang_diffusion.cli import initialize
from sgl_engine_sglang_diffusion.models import SourceLock
from sgl_engine_sglang_diffusion.process import run
from sgl_engine_sglang_diffusion.runtime import (
    CampaignRuntimeError,
    FileCampaignHooks,
    LockedSolQualityEvaluator,
    run_campaign_command,
)
from sgl_engine_sglang_diffusion.state import StateStore


pytest_plugins = ("helpers",)


FAKE_BENCHMARK = r"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-path", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--profile", action="store_true")
args, _ = parser.parse_known_args()

checkout = Path(__file__).resolve().parents[4]
output_file = Path(args.output_file)
media = Path(args.output_path)
run_dir = output_file.parent.parent
output_file.parent.mkdir(parents=True, exist_ok=True)
media.mkdir(parents=True, exist_ok=True)
optimized = (checkout / "python/sglang/kernels/agent/runtime.py").is_file()
total = 9.0 if optimized else 10.0
result = {
    "results": {
        "successful_requests": 5,
        "failed_requests": 0,
        "total_duration_seconds": total,
        "peak_memory_mb": 90.0 if optimized else 100.0,
    }
}
output_file.write_text(json.dumps(result) + "\n")
for index in range(5):
    (media / f"prompt-{index}.png").write_bytes(b"fake-image-" + bytes([index]))

if not optimized and "baseline" in run_dir.parts:
    counter = run_dir.parents[1] / "baseline-invocations.txt"
    previous = int(counter.read_text()) if counter.is_file() else 0
    counter.write_text(str(previous + 1))

if args.profile:
    trace_root = Path(os.environ["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"])
    trace_root.mkdir(parents=True, exist_ok=True)
    (trace_root / "fake.trace.json").write_text('{"traceEvents": []}\n')

if optimized:
    base = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{base}..HEAD"],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    hashes = {}
    for relative in changed:
        path = checkout / relative
        if path.is_file():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run_dir / "engagement-receipt.json").write_text(json.dumps({
        "schema_version": 1,
        "profile_id": "integrated",
        "model_match": True,
        "hardware_match": True,
        "workload_match": True,
        "techniques": {"kernel": {"engaged": True, "call_count": 5, "fallback_count": 0}},
        "source_hashes": hashes,
    }))
    (run_dir / "equivalence.json").write_text(json.dumps({
        "candidate_id": "integrated",
        "method_argument": "The registered dispatch changes launch plumbing only.",
        "baseline": {"global_steps": 4, "dit_calls": 4},
        "candidate": {"global_steps": 4, "dit_calls": 4},
        "logical_work_unchanged": True,
        "approximation": False,
        "step_skipping": False,
        "sparsity": False,
        "sub_16bit": False,
        "rank_reduction": False,
    }))
    (run_dir / "authenticity.json").write_text(json.dumps({
        "overall": "authenticity_only",
        "authentic": True,
    }))
"""


FAKE_AGENT = r"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

prompt_path = Path(sys.argv[-1])
prompt = prompt_path.read_text()

if prompt.startswith("# Independent lossless method-equivalence audit"):
    def field(name):
        match = re.search(rf"^{name}: (.+)$", prompt, re.MULTILINE)
        if match is None:
            raise RuntimeError(name)
        return match.group(1).strip()

    output_match = re.search(r"Write only (.+) as JSON with fields:", prompt)
    digest_match = re.search(r"method_argument_sha256='([0-9a-f]{64})'", prompt)
    if output_match is None or digest_match is None:
        raise RuntimeError("master output contract")
    output = Path(output_match.group(1))
    output.write_text(json.dumps({
        "accepted": True,
        "findings": [],
        "producer": "coding-agent-built-in-reasoning",
        "external_api": False,
        "technique": field("Technique"),
        "base_commit": field("Base commit"),
        "candidate_commit": field("Candidate commit"),
        "method_argument_sha256": digest_match.group(1),
    }))
    raise SystemExit(0)

worktree = Path.cwd()
delivery_match = re.search(r"Required delivery path: (.+)", prompt)
if delivery_match is None:
    raise RuntimeError("delivery path")
delivery_path = Path(delivery_match.group(1).strip())
executor_root = worktree.parent
attempt_path = executor_root / "fake-agent-attempt.txt"
attempt = int(attempt_path.read_text()) + 1 if attempt_path.is_file() else 1
attempt_path.write_text(str(attempt))

base = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=worktree,
    text=True,
    capture_output=True,
    check=True,
).stdout.strip()
base_path = executor_root / "fake-agent-base.txt"
if not base_path.is_file():
    base_path.write_text(base)
locked_base = base_path.read_text().strip()
if attempt == 1:
    shared = worktree / "python/sglang/kernels/agent"
    profile_root = shared / "diffusion/fake-model"
    profile_root.mkdir(parents=True, exist_ok=True)
    (shared / "__init__.py").write_text("")
    (shared / "registry.py").write_text(
        'OPTION = "--agent-optimization"\nMODES = ("off", "auto")\n'
    )
    (shared / "manifest.py").write_text("SCHEMA_VERSION = 1\n")
    (shared / "runtime.py").write_text(
        "def engage_agent_profile():\n    return True\n"
    )
    (shared / "receipt.py").write_text(
        "def engagement_receipt():\n    return {'engaged': True}\n"
    )
    runtime_hash = hashlib.sha256((shared / "runtime.py").read_bytes()).hexdigest()
    (profile_root / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "profile_id": "fake-kernel",
        "campaign_id": "mock-campaign",
        "model_ids": ["fake-model"],
        "sglang_base_sha": locked_base,
        "hardware": {"environment": "cpu-test", "gpu_count": 1},
        "workload": {"prompt_count": 5},
        "techniques": {"kernel": {"enabled": True}},
        "server_args": {"agent_optimization": "fake-kernel"},
        "fallback_policy": "native",
        "source_hashes": {"python/sglang/kernels/agent/runtime.py": runtime_hash},
        "integrated_delivery_sha256": "0" * 64,
        "speedup": 10.0 / 9.0,
        "derived_checkpoint": None,
    }, sort_keys=True))
    subprocess.run(["git", "add", "python/sglang/kernels/agent"], cwd=worktree, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fake Agent",
            "-c",
            "user.email=fake@example.invalid",
            "commit",
            "-m",
            "fake kernel candidate",
        ],
        cwd=worktree,
        check=True,
        capture_output=True,
    )

head = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=worktree,
    text=True,
    capture_output=True,
    check=True,
).stdout.strip()
root_commit = locked_base
changed = subprocess.run(
    ["git", "diff", "--name-only", f"{root_commit}..HEAD"],
    cwd=worktree,
    text=True,
    capture_output=True,
    check=True,
).stdout.splitlines()
source_hashes = {
    relative: hashlib.sha256((worktree / relative).read_bytes()).hexdigest()
    for relative in changed
    if (worktree / relative).is_file()
}

run_dir = worktree / "candidate-run"
(run_dir / "outputs/media").mkdir(parents=True, exist_ok=True)
raw = {
    "results": {
        "successful_requests": 5,
        "failed_requests": 0,
        "total_duration_seconds": 9.0,
        "peak_memory_mb": 90.0,
    }
}
(run_dir / "outputs/benchmark.jsonl").write_text(json.dumps(raw) + "\n")
for index in range(5):
    (run_dir / f"outputs/media/prompt-{index}.png").write_bytes(
        b"candidate-" + bytes([index])
    )
(run_dir / "PERFORMANCE.json").write_text(json.dumps({
    "schema_version": 1,
    "total_s": 9.0,
    "peak_memory_mib": 90.0,
    "timing_scope": "load_excluded_end_to_end",
}))
(run_dir / "source-hashes.json").write_text(json.dumps(source_hashes))
(run_dir / "engagement-receipt.json").write_text(json.dumps({
    "schema_version": 1,
    "profile_id": "fake-kernel",
    "model_match": True,
    "hardware_match": True,
    "workload_match": True,
    "techniques": {"kernel": {"engaged": True, "call_count": 5, "fallback_count": 0}},
    "source_hashes": source_hashes,
}))
(run_dir / "equivalence.json").write_text(json.dumps({
    "candidate_id": "fake-kernel",
    "method_argument": "Only runtime dispatch and launch plumbing change.",
    "baseline": {"global_steps": 4, "dit_calls": 4},
    "candidate": {"global_steps": 4, "dit_calls": 4},
    "logical_work_unchanged": True,
    "approximation": False,
    "step_skipping": False,
    "sparsity": False,
    "sub_16bit": False,
    "rank_reduction": False,
}))
(run_dir / "authenticity.json").write_text(json.dumps({
    "overall": "authenticity_only",
    "authentic": True,
}))
implementation = {
    "schema_version": 1,
    "candidate_id": "fake-kernel",
    "technique": "kernel",
    "kind": "patch",
    "base_commit": root_commit,
    "candidate_commit": head,
    "activation": {
        "env": {},
        "server_args": ["--agent-optimization", "fake-kernel"],
    },
    "eval_profile": {
        "prompt_count": 5,
        "timing_scope": "load_excluded_end_to_end",
    },
    "knowledge_origin": [{"source": "locked-test", "commit": root_commit}],
}
(run_dir / "implementation-manifest.json").write_text(
    json.dumps(implementation, sort_keys=True)
)
reported_total = 5.0 if attempt == 1 else 9.0
delivery = {
    "schema_version": 2,
    "status": "complete",
    "component": "kernel",
    "model_id": "fake-model",
    "baseline": {"total_s": 10.0},
    "frontier_points": [{
        "candidate_id": "fake-kernel",
        "run_dir": str(run_dir),
        "activation": {
            "env": {},
            "server_args": ["--agent-optimization", "fake-kernel"],
        },
        "implementation_manifest": implementation,
        "performance": {
            "frontier_axis": "latency",
            "baseline_total_s": 10.0,
            "candidate_total_s": reported_total,
            "speedup": 10.0 / reported_total,
        },
        "quality": {
            "mode": "not_gated",
            "lpips_max": None,
            "lpips_mean": None,
            "visual_overall": "authenticity_only",
            "visual_verdict": "authenticity.json",
            "relation": "equivalent",
        },
        "artifacts": [
            "PERFORMANCE.json",
            "outputs/benchmark.jsonl",
            "outputs/media",
            "source-hashes.json",
            "engagement-receipt.json",
            "equivalence.json",
            "authenticity.json",
            "implementation-manifest.json",
        ],
    }],
    "pareto_assessment": "Measured CPU fixture frontier.",
}
delivery_path.write_text(json.dumps(delivery, sort_keys=True))
"""


def _prepare_fake_source(repository: Path, tmp_path: Path) -> Path:
    benchmark = (
        repository
        / "python/sglang/multimodal_gen/benchmarks/bench_offline_throughput.py"
    )
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(FAKE_BENCHMARK)
    (repository / "python/sglang/kernels/ops/diffusion").mkdir(
        parents=True, exist_ok=True
    )
    (repository / "python/sglang/kernels/README.md").write_text(
        "fake unified kernels\n"
    )
    (repository / "python/sglang/kernels/jit").mkdir(parents=True)
    (repository / "python/sglang/kernels/jit/README.md").write_text("fake JIT root\n")
    (repository / "python/sglang/kernels/ops/diffusion/README.md").write_text(
        "fake diffusion operator\n"
    )
    (repository / "docs/inference").mkdir(parents=True)
    (repository / "docs/inference/optimizations.md").write_text("fake optimization\n")
    (repository / "diffusion/docs").mkdir(parents=True)
    (repository / "diffusion/docs/optimization.md").write_text("fake KDA note\n")
    (repository / "search").mkdir()
    (repository / "search/plan_eval.py").write_text(
        "raise SystemExit('quality evaluator is not used by this lossless test')\n"
    )
    run(["git", "add", "."], cwd=repository)
    run(["git", "commit", "-m", "fake SGLang runtime"], cwd=repository)
    run(["git", "branch", "-M", "main"], cwd=repository)

    agent = tmp_path / "fake_agent.py"
    agent.write_text(FAKE_AGENT)
    return agent


def _write_goal(tmp_path: Path, repository: Path, agent: Path) -> Path:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    goal = tmp_path / "goal.yaml"
    goal.write_text(
        f"""schema_version: 1
model:
  id: fake-model
hardware:
  environment: cpu-test
  gpu_count: 1
workload:
  prompts: {prompts.name}
  prompt_count: 5
  seed: 42
  height: 64
  width: 64
  frames: 1
  fps: 24
  steps: 4
  guidance: 1.0
  dtype: bfloat16
  timing_scope: load_excluded_end_to_end
goal:
  target_speedup: 1.1
  allow_quality_gated: false
source:
  sglang_repo: {repository}
  sglang_ref: main
  sol_engine_repo: {repository}
  sol_engine_ref: main
  fastvideo_repo: {repository}
  fastvideo_ref: main
  kda_pilot_repo: {repository}
  kda_pilot_ref: main
agent:
  command: [{sys.executable}, {agent}]
"""
    )
    return goal


def test_default_runtime_rejects_fabrication_resumes_and_packages(
    tmp_path: Path, fake_git_repo: Path, monkeypatch: object
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        runtime_module,
        "_validate_sol_contract",
        lambda lock, checkout=None: None,
    )
    agent = _prepare_fake_source(fake_git_repo, tmp_path)
    goal = _write_goal(tmp_path, fake_git_repo, agent)
    campaign = initialize(goal, tmp_path / "runs")

    payload = run_campaign_command("run", campaign)
    assert payload["new_state"] == "SEARCHING"
    for _ in range(120):
        payload = run_campaign_command("resume", campaign)
        if payload["new_state"] == "TARGET_REACHED":
            break
        time.sleep(0.05)

    assert payload["new_state"] == "TARGET_REACHED"
    assert (campaign / "baseline-invocations.txt").read_text() == "1"
    events = (campaign / "events.jsonl").read_text()
    assert "executor_resumed" in events
    assert "speedup_tamper" in next(
        path.read_text() for path in (campaign / "executors").rglob("feedback-001.md")
    )
    assert (campaign / "patch/sglang.patch").is_file()
    assert (campaign / "patch/SHA256SUMS").is_file()
    assert (campaign / "patch/apply_and_verify.sh").is_file()
    assert (
        "python/sglang/kernels/agent/runtime.py"
        in (campaign / "patch/sglang.patch").read_text()
    )

    second = run_campaign_command("resume", campaign)
    assert second["new_state"] == "TARGET_REACHED"
    assert (campaign / "baseline-invocations.txt").read_text() == "1"


def test_production_sol_contract_rejects_an_unreviewed_commit() -> None:
    lock = SourceLock(
        name="sol_engine",
        repository="https://github.com/NVlabs/Sana.git",
        requested_ref="main",
        commit="f" * 40,
    )

    with pytest.raises(CampaignRuntimeError, match="reviewed correctness contract"):
        runtime_module._validate_sol_contract(lock)


def test_independent_lpips_rejects_candidate_prompt_symlink(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "run/outputs/frames"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "frame.png").write_bytes(b"outside")
    for index in range(5):
        baseline_prompt = baseline / f"prompt-{index:02d}"
        baseline_prompt.mkdir(parents=True)
        (baseline_prompt / "frame.png").write_bytes(b"baseline")
        if index == 0:
            candidate.mkdir(parents=True)
            (candidate / "prompt-00").symlink_to(outside, target_is_directory=True)
        else:
            candidate_prompt = candidate / f"prompt-{index:02d}"
            candidate_prompt.mkdir(parents=True)
            (candidate_prompt / "frame.png").write_bytes(b"candidate")

    with pytest.raises(CampaignRuntimeError, match="unsafe aligned prompt path"):
        LockedSolQualityEvaluator._aligned_prompt_pairs(
            baseline,
            candidate,
            tmp_path / "run",
        )


def test_sol_round_budget_counts_all_executors_and_resumes(tmp_path: Path) -> None:
    store = StateStore.open(tmp_path / "state.sqlite", tmp_path / "events.jsonl")
    store.create_campaign("campaign-1")
    try:
        store.record_event(
            "campaign-1",
            "executor_spawned",
            "spawn-1",
            {"executor_id": "one", "technique": "kernel"},
        )
        store.record_event(
            "campaign-1",
            "executor_resumed",
            "resume-1",
            {"executor_id": "one", "attempt": 2},
        )
        store.record_event(
            "campaign-1",
            "executor_spawned",
            "spawn-2",
            {"executor_id": "two", "technique": "kernel"},
        )
        store.record_event(
            "campaign-1",
            "executor_spawned",
            "other-lane",
            {"executor_id": "cache-one", "technique": "cache"},
        )
        hooks = object.__new__(FileCampaignHooks)
        hooks.store = store
        hooks.campaign_id = "campaign-1"

        assert hooks._technique_rounds("kernel") == 3
        assert hooks._technique_rounds("cache") == 1
    finally:
        store.close()
