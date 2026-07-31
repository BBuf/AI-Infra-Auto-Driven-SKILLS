"""Local deterministic quality metrics for diffusion candidate verification."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable


class MetricUnavailable(RuntimeError):
    """Raised when optional LPIPS runtime support is unavailable."""


FramePair = tuple[Path, Path]


def _load_modules(
    import_module: Callable[[str], object] = importlib.import_module,
) -> tuple[object, object]:
    try:
        lpips_module = import_module("lpips")
    except Exception as error:
        raise MetricUnavailable(f"lpips is not importable: {error}") from error
    try:
        torch_module = import_module("torch")
    except Exception as error:
        raise MetricUnavailable(f"torch is not importable: {error}") from error
    return lpips_module, torch_module


def score_frame_pairs(
    frame_pairs: list[FramePair],
    import_module: Callable[[str], object] = importlib.import_module,
) -> list[float]:
    """Return AlexNet LPIPS for aligned frame pairs."""
    if not frame_pairs:
        raise MetricUnavailable("no aligned frame pairs are available")
    lpips_module, torch_module = _load_modules(import_module)
    factory = getattr(lpips_module, "LPIPS", None)
    load_image = getattr(lpips_module, "load_image", None)
    image_to_tensor = getattr(lpips_module, "im2tensor", None)
    if factory is None or load_image is None or image_to_tensor is None:
        raise MetricUnavailable("lpips model or image helpers are unavailable")
    try:
        try:
            model = factory(net="alex", verbose=False)
        except TypeError:
            model = factory(net="alex")
        if hasattr(model, "eval"):
            model.eval()
        scores: list[float] = []
        with getattr(torch_module, "no_grad")():
            for baseline_frame, candidate_frame in frame_pairs:
                baseline = image_to_tensor(load_image(str(baseline_frame)))
                candidate = image_to_tensor(load_image(str(candidate_frame)))
                value = model(baseline, candidate)
                if hasattr(value, "item"):
                    value = value.item()
                scores.append(float(value))
        return scores
    except MetricUnavailable:
        raise
    except Exception as error:
        raise MetricUnavailable(f"LPIPS scoring failed: {error}") from error
