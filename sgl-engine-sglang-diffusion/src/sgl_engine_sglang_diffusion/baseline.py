from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .driver import Activation, BenchmarkRun, DriverError, SGLangDiffusionDriver
from .models import BaselineRecord, CampaignGoal
from .process import run


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".gif", ".webm"}


class BaselineError(RuntimeError):
    """Raised when a frozen baseline cannot be made authentic and durable."""


class BaselineRunner:
    def __init__(self, driver: SGLangDiffusionDriver) -> None:
        self.driver = driver

    def freeze(
        self,
        goal: CampaignGoal,
        campaign_dir: Path,
        *,
        sglang_commit: str,
        activation: Activation | None = None,
    ) -> BaselineRecord:
        campaign_dir = campaign_dir.resolve()
        baseline_path = campaign_dir / "BASELINE.json"
        if baseline_path.exists() or baseline_path.is_symlink():
            raise BaselineError(
                f"frozen baseline already exists and cannot be refreshed: {baseline_path}"
            )

        run_dir = campaign_dir / "baseline" / "run"
        try:
            benchmark = self.driver.run(
                goal, run_dir, activation=activation, profile=False
            )
            frame_root = campaign_dir / "baseline" / "frames"
            self._align_frames(
                benchmark,
                frame_root,
                expected_outputs=goal.workload.prompt_count,
            )
            record = BaselineRecord(
                model_id=goal.model.id,
                total_s=benchmark.normalized["total_s"],
                peak_memory_mib=benchmark.normalized["peak_memory_mib"],
                timing_scope=goal.workload.timing_scope,
                run_dir=benchmark.run_dir,
                baseline_frames=frame_root,
                sglang_commit=sglang_commit,
            )
            self._atomic_write(baseline_path, record)
            return record
        except DriverError as error:
            raise BaselineError(str(error)) from error

    @staticmethod
    def load(path: Path) -> BaselineRecord:
        return BaselineRecord.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _align_frames(
        benchmark: BenchmarkRun,
        destination: Path,
        *,
        expected_outputs: int,
    ) -> None:
        media = sorted(
            path
            for path in benchmark.media_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _IMAGE_SUFFIXES | _VIDEO_SUFFIXES
        )
        if len(media) < expected_outputs:
            raise BaselineError(
                f"baseline produced {len(media)} durable media outputs; "
                f"expected at least {expected_outputs} in {benchmark.media_dir}"
            )
        if destination.exists() or destination.is_symlink():
            raise BaselineError(
                f"aligned frame destination already exists: {destination}"
            )
        destination.mkdir(parents=True)

        ffmpeg = shutil.which("ffmpeg")
        for index, source in enumerate(media[:expected_outputs]):
            prompt_dir = destination / f"prompt-{index:02d}"
            prompt_dir.mkdir()
            suffix = source.suffix.lower()
            if suffix in _IMAGE_SUFFIXES:
                shutil.copy2(source, prompt_dir / f"frame-000001{suffix}")
            else:
                if ffmpeg is None:
                    raise BaselineError(
                        "ffmpeg is required to align baseline video frames"
                    )
                result = run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-loglevel",
                        "error",
                        "-i",
                        str(source),
                        str(prompt_dir / "frame-%06d.png"),
                    ],
                    cwd=benchmark.run_dir,
                    check=False,
                )
                if result.returncode != 0 or not any(prompt_dir.glob("*.png")):
                    raise BaselineError(
                        f"failed to extract aligned frames from {source}: "
                        f"{result.stderr.strip()}"
                    )

    @staticmethod
    def _atomic_write(path: Path, record: BaselineRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                record.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
