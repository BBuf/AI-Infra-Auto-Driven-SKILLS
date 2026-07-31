from __future__ import annotations

import json
from pathlib import Path

import pytest

import sgl_engine_sglang_diffusion.runtime as runtime_module
from sgl_engine_sglang_diffusion.cli import initialize, main
from sgl_engine_sglang_diffusion.models import CampaignStatus, SourceLock
from sgl_engine_sglang_diffusion.process import run
from sgl_engine_sglang_diffusion.runtime import (
    CampaignRuntimeError,
    LockedSolQualityEvaluator,
    run_campaign_command,
)
from sgl_engine_sglang_diffusion.state import StateStore

pytest_plugins = ("helpers",)


FAKE_BENCHMARK = r"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-path", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--profile", action="store_true")
args, _ = parser.parse_known_args()

output_file = Path(args.output_file)
media = Path(args.output_path)
run_dir = output_file.parent.parent
output_file.parent.mkdir(parents=True, exist_ok=True)
media.mkdir(parents=True, exist_ok=True)
output_file.write_text(json.dumps({
    "results": {
        "successful_requests": 5,
        "failed_requests": 0,
        "total_duration_seconds": 10.0,
        "peak_memory_mb": 100.0,
    }
}) + "\n")
for index in range(5):
    (media / f"prompt-{index}.png").write_bytes(b"fake-image-" + bytes([index]))

if "baseline" in run_dir.parts:
    counter = run_dir.parents[1] / "baseline-invocations.txt"
    previous = int(counter.read_text()) if counter.is_file() else 0
    counter.write_text(str(previous + 1))

if args.profile:
    trace_root = Path(os.environ["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"])
    trace_root.mkdir(parents=True, exist_ok=True)
    (trace_root / "fake.trace.json").write_text('{"traceEvents": []}\n')
"""


def _prepare_fake_source(repository: Path) -> None:
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


def _write_goal(tmp_path: Path, repository: Path) -> Path:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {index}" for index in range(5)) + "\n")
    goal = tmp_path / "goal.yaml"
    goal.write_text(
        f"""schema_version: 2
execution_mode: interactive_single_agent
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
""",
        encoding="utf-8",
    )
    return goal


def test_runtime_yields_to_one_root_agent_and_rejects_one_submission(
    tmp_path: Path,
    fake_git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_validate_sol_contract",
        lambda lock, checkout=None: None,
    )
    _prepare_fake_source(fake_git_repo)
    campaign = initialize(_write_goal(tmp_path, fake_git_repo), tmp_path / "runs")

    launched = run_campaign_command("run", campaign)

    assert launched["new_state"] == "AWAITING_AGENT"
    assert (campaign / "BASELINE.json").is_file()
    assert (campaign / "profiles/0/PROFILE-DIGEST.json").is_file()
    assert (campaign / "ROUTES.json").is_file()
    assert (campaign / "baseline-invocations.txt").read_text() == "1"
    assert not (campaign / "executors").exists()
    assert not list(campaign.rglob("*MASTER*"))

    yielded = run_campaign_command("resume", campaign)
    assert yielded["new_state"] == "AWAITING_AGENT"
    assert (campaign / "baseline-invocations.txt").read_text() == "1"

    manifest = json.loads((campaign / "CAMPAIGN.json").read_text())
    routes = json.loads((campaign / "ROUTES.json").read_text())["routes"]
    assert main(["work", "--campaign", str(campaign), "--json"]) == 0
    work = json.loads(capsys.readouterr().out)
    assert work["execution_mode"] == "interactive_single_agent"
    assert work["legal_actions"]
    assert (
        main(
            [
                "claim",
                "--campaign",
                str(campaign),
                "--technique",
                routes[0],
            ]
        )
        == 0
    )
    claimed = json.loads(capsys.readouterr().out)
    delivery = Path(claimed["claimed_work_order"]["delivery_path"])
    delivery.write_text("{}", encoding="utf-8")
    assert (
        main(["submit", "--campaign", str(campaign), "--delivery", str(delivery)]) == 0
    )
    rejected = json.loads(capsys.readouterr().out)

    assert rejected["status"] == "AWAITING_AGENT"
    assert rejected["verification"]["new_state"] == "AWAITING_AGENT"
    with StateStore.open(campaign / "state.sqlite", campaign / "events.jsonl") as store:
        assert store.status(manifest["campaign_id"]) is CampaignStatus.AWAITING_AGENT
        assert (
            len(store.events(manifest["campaign_id"], event_type="candidate_submitted"))
            == 1
        )
        rejected_events = store.events(
            manifest["campaign_id"], event_type="work_rejected"
        )
        assert len(rejected_events) == 1
        assert rejected_events[0]["payload"]["findings"][0]["code"] == (
            "invalid_delivery"
        )
    assert "executor_resumed" not in (campaign / "events.jsonl").read_text()


def test_production_sol_contract_rejects_an_unreviewed_commit() -> None:
    lock = SourceLock(
        name="sol_engine",
        repository="https://github.com/NVlabs/Sana.git",
        requested_ref="main",
        commit="f" * 40,
    )

    with pytest.raises(CampaignRuntimeError, match="reviewed correctness contract"):
        runtime_module._validate_sol_contract(lock)


def test_runtime_rejects_legacy_multi_agent_campaign(
    tmp_path: Path,
    fake_git_repo: Path,
) -> None:
    campaign = initialize(_write_goal(tmp_path, fake_git_repo), tmp_path / "runs")
    legacy = campaign / "executors" / "1" / "kernel"
    legacy.mkdir(parents=True)
    (legacy / "PROCESS.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        CampaignRuntimeError,
        match="legacy multi-agent campaign cannot be resumed",
    ):
        run_campaign_command("resume", campaign)


def test_locked_lpips_rejects_candidate_prompt_symlink(tmp_path: Path) -> None:
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
