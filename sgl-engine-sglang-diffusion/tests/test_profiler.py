from __future__ import annotations

import json
from pathlib import Path

from sgl_engine_sglang_diffusion.driver import SGLangDiffusionDriver
from sgl_engine_sglang_diffusion.models import CampaignGoal, ProfileDigest
from sgl_engine_sglang_diffusion.process import CommandResult
from sgl_engine_sglang_diffusion.profiler import Profiler, TechniqueRouter

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
        (profile_dir / "trace.json.gz").write_bytes(b"trace")
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
    return ProfileDigest(
        run_dir=tmp_path,
        timing_scope="load_excluded_end_to_end",
        stage_ms={"denoise": 900.0, "decode": 100.0},
        hotspots=[{"name": "aten::mul", "total_ms": 120.0}],
        trace_paths=[tmp_path / "trace.json.gz"],
    )


def test_profiler_preserves_trace_and_normalizes_summary(tmp_path: Path) -> None:
    goal: CampaignGoal = make_goal(tmp_path)
    driver = SGLangDiffusionDriver(make_checkout(tmp_path), runner=ProfileRunner())
    digest = Profiler(driver).collect(goal, tmp_path / "campaign", epoch=1)
    assert digest.stage_ms == {"decode": 100.0, "denoise": 900.0}
    assert [row["name"] for row in digest.hotspots] == [
        "scaled_dot_product_attention",
        "aten::mul",
    ]
    assert digest.trace_paths[0].is_file()
    assert (digest.run_dir / "PROFILE-DIGEST.json").is_file()


def test_profile_routes_attention_and_glue_hotspots(tmp_path: Path) -> None:
    digest = ProfileDigest(
        run_dir=tmp_path,
        timing_scope="load_excluded_end_to_end",
        stage_ms={"denoise": 900.0, "decode": 100.0},
        hotspots=[
            {"name": "scaled_dot_product_attention", "total_ms": 400.0},
            {"name": "aten::mul", "total_ms": 120.0},
        ],
        trace_paths=[tmp_path / "trace.json.gz"],
    )
    router = TechniqueRouter()
    routed = router.route(digest, allow_quality_gated=True, gpu_count=1)
    assert routed == [
        "kernel",
        "cache",
        "pisa",
        "quantization",
        "token_pruning",
    ]
    assert "profile-evidence" in router.last_evidence["kernel"]["knowledge"]


def test_profile_adds_topology_only_for_multi_gpu(tmp_path: Path) -> None:
    digest = make_profile_digest(tmp_path)
    routed = TechniqueRouter().route(digest, allow_quality_gated=False, gpu_count=4)
    assert routed == ["kernel", "topology"]
