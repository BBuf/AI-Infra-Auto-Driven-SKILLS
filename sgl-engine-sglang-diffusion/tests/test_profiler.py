from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.driver import SGLangDiffusionDriver
from sgl_engine_sglang_diffusion.models import CampaignGoal, ProfileDigest
from sgl_engine_sglang_diffusion.process import CommandResult
from sgl_engine_sglang_diffusion.profiler import ProfileError, Profiler, TechniqueRouter

from test_driver import make_checkout, make_goal


class ProfileRunner:
    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> CommandResult:
        output = Path(argv[argv.index("--output-file") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "results": {
                        "successful_requests": 5,
                        "failed_requests": 0,
                        "total_duration_seconds": 1.0,
                        "peak_memory_mb": 1024.0,
                    }
                }
            )
            + "\n"
        )
        profile_dir = Path(env["SGLANG_DIFFUSION_TORCH_PROFILER_DIR"])
        with gzip.open(profile_dir / "trace.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "traceEvents": [
                        {
                            "ph": "X",
                            "cat": "kernel",
                            "name": "scaled_dot_product_attention",
                            "dur": 400000,
                        },
                        {
                            "ph": "X",
                            "cat": "kernel",
                            "name": "aten::mul",
                            "dur": 120000,
                        },
                        {
                            "ph": "X",
                            "cat": "cuda_runtime",
                            "name": "cudaMemcpyAsync",
                            "dur": 5000,
                        },
                    ]
                },
                handle,
            )
        (profile_dir / "profile-summary.json").write_text(
            json.dumps(
                {
                    "stage_ms": {"denoise": 900.0, "decode": 100.0},
                    "hotspots": [
                        {
                            "name": "aten::mul",
                            "category": "pointwise",
                            "total_ms": 120.0,
                            "calls": 40,
                        },
                        {
                            "name": "scaled_dot_product_attention",
                            "category": "attention",
                            "total_ms": 400.0,
                            "calls": 20,
                        },
                    ],
                }
            )
        )
        return CommandResult(tuple(argv), 0, "native engine", "")


def make_profile_digest(tmp_path: Path) -> ProfileDigest:
    trace = tmp_path / "trace.json.gz"
    trace.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "traceEvents": [
                    {"ph": "X", "cat": "kernel", "name": "aten::mul", "dur": 120000}
                ]
            },
            handle,
        )
    return ProfileDigest(
        run_dir=tmp_path,
        timing_scope="load_excluded_end_to_end",
        stage_ms={"cuda_kernel": 120.0},
        hotspots=[{"name": "aten::mul", "total_ms": 120.0}],
        trace_paths=[trace],
        trace_sha256={str(trace): Profiler._sha256_file(trace)},
        parser_version=Profiler.PARSER_VERSION,
        event_count=1,
    )


def test_profiler_preserves_trace_and_normalizes_summary(tmp_path: Path) -> None:
    goal: CampaignGoal = make_goal(tmp_path)
    driver = SGLangDiffusionDriver(make_checkout(tmp_path), runner=ProfileRunner())
    digest = Profiler(driver).collect(goal, tmp_path / "campaign", epoch=1)
    assert digest.stage_ms == {
        "copy_layout": 5.0,
        "cuda_kernel": 520.0,
        "instrumented:decode": 100.0,
        "instrumented:denoise": 900.0,
    }
    assert [row["name"] for row in digest.hotspots] == [
        "scaled_dot_product_attention",
        "aten::mul",
        "cudaMemcpyAsync",
    ]
    assert digest.trace_paths[0].is_file()
    assert digest.event_count == 3
    assert digest.parser_version == "chrome-trace-v1"
    assert digest.trace_sha256[str(digest.trace_paths[0])]
    assert (tmp_path / "campaign/profiles/1/PROFILE-INVENTORY.json").is_file()
    assert (tmp_path / "campaign/profiles/1/PROFILE-DIGEST.json").is_file()
    repeated = Profiler(driver).collect(goal, tmp_path / "campaign", epoch=1)
    assert repeated == digest


def test_profile_routes_attention_and_glue_hotspots(tmp_path: Path) -> None:
    digest = make_profile_digest(tmp_path)
    router = TechniqueRouter()
    routed = router.route(digest, allow_quality_gated=True, gpu_count=1)
    assert routed == [
        "residency",
        "kernel",
        "cache",
        "pisa",
        "quantization",
        "token_pruning",
    ]
    assert "sglang-residency-history" in router.last_evidence["residency"][
        "knowledge"
    ]
    assert "profile-evidence" in router.last_evidence["kernel"]["knowledge"]


def test_profile_freezes_parallel_topology_for_multi_gpu(tmp_path: Path) -> None:
    digest = make_profile_digest(tmp_path)
    routed = TechniqueRouter().route(digest, allow_quality_gated=False, gpu_count=4)
    assert routed == ["residency", "kernel"]


def test_profile_rejects_corrupt_or_event_empty_raw_trace(tmp_path: Path) -> None:
    trace = tmp_path / "broken.trace.json.gz"
    trace.write_bytes(b"not gzip")
    with pytest.raises(ProfileError, match="cannot parse"):
        Profiler._performance_tables(tmp_path, [trace])

    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": [{"ph": "i", "name": "marker"}]}, handle)
    with pytest.raises(ProfileError, match="no complete positive-duration"):
        Profiler._performance_tables(tmp_path, [trace])


def test_cached_profile_rejects_changed_trace(tmp_path: Path) -> None:
    digest = make_profile_digest(tmp_path)
    digest.trace_paths[0].write_bytes(b"changed")
    with pytest.raises(ProfileError, match="hash changed"):
        Profiler.validate_digest(digest)


def test_collect_archives_broken_cached_digest_and_recaptures(tmp_path: Path) -> None:
    goal: CampaignGoal = make_goal(tmp_path)
    campaign = tmp_path / "campaign"
    cached = campaign / "profiles/1/PROFILE-DIGEST.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_dir": str(tmp_path),
                "timing_scope": goal.workload.timing_scope,
                "stage_ms": {"end_to_end": 1000.0},
                "hotspots": [],
                "trace_paths": [],
            }
        )
    )
    driver = SGLangDiffusionDriver(make_checkout(tmp_path), runner=ProfileRunner())
    digest = Profiler(driver).collect(goal, campaign, epoch=1)
    assert digest.event_count == 3
    assert list(cached.parent.glob("PROFILE-DIGEST.rejected-*.json"))
    assert cached.is_file()
