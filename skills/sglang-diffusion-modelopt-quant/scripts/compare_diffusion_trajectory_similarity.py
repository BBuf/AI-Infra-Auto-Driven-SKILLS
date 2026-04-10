"""Compare diffusion BF16 and quantized runs via trajectory-latent similarity.

This tool runs two SGLang diffusion variants with the same prompt and seed,
captures intermediate denoising latents via `return_trajectory_latents`, and
reports cosine / error metrics for each timestep plus final frame metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


def parse_component_overrides(entries: Sequence[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid component override '{entry}'. Expected format component=path."
            )
        component, path = entry.split("=", 1)
        component = component.strip().replace("-", "_")
        path = path.strip()
        if not component or not path:
            raise ValueError(
                f"Invalid component override '{entry}'. Expected format component=path."
            )
        overrides[component] = path
    return overrides


def _cosine_similarity(flat_a: torch.Tensor, flat_b: torch.Tensor) -> float:
    norm_a = torch.linalg.vector_norm(flat_a).item()
    norm_b = torch.linalg.vector_norm(flat_b).item()
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(F.cosine_similarity(flat_a, flat_b, dim=0).item())


def compute_tensor_metrics(lhs: Any, rhs: Any) -> dict[str, float]:
    lhs_tensor = torch.as_tensor(lhs).detach().cpu().float()
    rhs_tensor = torch.as_tensor(rhs).detach().cpu().float()
    if lhs_tensor.shape != rhs_tensor.shape:
        raise ValueError(
            f"Metric shape mismatch: {tuple(lhs_tensor.shape)} vs {tuple(rhs_tensor.shape)}"
        )

    diff = lhs_tensor - rhs_tensor
    mse = float(diff.square().mean().item())
    rmse = float(math.sqrt(mse))
    mae = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    l2 = float(torch.linalg.vector_norm(diff).item())
    cosine = _cosine_similarity(lhs_tensor.reshape(-1), rhs_tensor.reshape(-1))
    return {
        "cosine_similarity": cosine,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs": max_abs,
        "l2": l2,
    }


def compute_uint8_frame_metrics(lhs: Any, rhs: Any) -> dict[str, float]:
    metrics = compute_tensor_metrics(lhs, rhs)
    mse = metrics["mse"]
    metrics["psnr_db"] = (
        float("inf") if mse == 0.0 else 20 * math.log10(255.0) - 10 * math.log10(mse)
    )
    return metrics


def _normalize_step_index(step_index: int, num_steps: int) -> int:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if step_index < 0:
        step_index += num_steps
    if step_index < 0 or step_index >= num_steps:
        raise IndexError(
            f"Requested step index {step_index} is outside the valid range [0, {num_steps})."
        )
    return step_index


def _maybe_scalar(timestep: torch.Tensor | None, index: int) -> float | None:
    if timestep is None:
        return None
    value = timestep[index]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
    return float(value)


def summarize_trajectory_metrics(
    reference_latents: Any,
    candidate_latents: Any,
    *,
    reference_timesteps: Any = None,
    candidate_timesteps: Any = None,
    step_index: int = -1,
) -> dict[str, Any]:
    ref = torch.as_tensor(reference_latents).detach().cpu().float()
    cand = torch.as_tensor(candidate_latents).detach().cpu().float()
    if ref.shape != cand.shape:
        raise ValueError(
            f"Trajectory shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}"
        )
    if ref.ndim < 2:
        raise ValueError(
            f"Expected trajectory latents with an explicit timestep dimension, got {tuple(ref.shape)}"
        )

    num_steps = ref.shape[1]
    selected_step = _normalize_step_index(step_index, num_steps)
    ref_t = (
        torch.as_tensor(reference_timesteps).detach().cpu()
        if reference_timesteps is not None
        else None
    )
    cand_t = (
        torch.as_tensor(candidate_timesteps).detach().cpu()
        if candidate_timesteps is not None
        else None
    )

    per_step: list[dict[str, Any]] = []
    for idx in range(num_steps):
        metrics = compute_tensor_metrics(ref[:, idx], cand[:, idx])
        metrics["step_index"] = idx
        metrics["reference_timestep"] = _maybe_scalar(ref_t, idx)
        metrics["candidate_timestep"] = _maybe_scalar(cand_t, idx)
        per_step.append(metrics)

    return {
        "trajectory_shape": list(ref.shape),
        "num_steps": num_steps,
        "selected_step_index": selected_step,
        "selected_step_metrics": per_step[selected_step],
        "per_step_metrics": per_step,
    }


def summarize_output_frame_metrics(
    reference_frames: Sequence[Any],
    candidate_frames: Sequence[Any],
) -> dict[str, Any]:
    if len(reference_frames) != len(candidate_frames):
        raise ValueError(
            f"Output frame count mismatch: {len(reference_frames)} vs {len(candidate_frames)}"
        )
    if not reference_frames:
        raise ValueError("No output frames available for comparison.")

    ref_stack = np.stack([np.asarray(frame) for frame in reference_frames], axis=0)
    cand_stack = np.stack([np.asarray(frame) for frame in candidate_frames], axis=0)

    frame0_metrics = compute_uint8_frame_metrics(ref_stack[0], cand_stack[0])
    mid_index = len(reference_frames) // 2
    mid_metrics = compute_uint8_frame_metrics(
        ref_stack[mid_index], cand_stack[mid_index]
    )
    all_metrics = compute_uint8_frame_metrics(ref_stack, cand_stack)

    return {
        "num_frames": len(reference_frames),
        "frame0_metrics": frame0_metrics,
        "mid_frame_index": mid_index,
        "mid_frame_metrics": mid_metrics,
        "all_frames_metrics": all_metrics,
    }


def extract_result_frames(result: Any) -> list[np.ndarray]:
    if result.frames is not None:
        return [np.asarray(frame) for frame in result.frames]

    sample = result.samples
    if sample is None:
        if result.output_file_path:
            output_path = Path(result.output_file_path)
            if not output_path.exists():
                raise ValueError(
                    "GenerationResult did not contain frames or samples, and its "
                    f"output_file_path does not exist: {output_path}"
                )
            if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return [np.asarray(iio.imread(output_path))]
            return [np.asarray(frame) for frame in iio.imiter(output_path)]
        raise ValueError(
            "GenerationResult did not contain frames, samples, or a readable output_file_path."
        )

    if isinstance(sample, torch.Tensor):
        tensor = sample.detach().cpu().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(
                f"Unsupported tensor sample shape for frame extraction: {tuple(tensor.shape)}"
            )
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        frames = tensor.permute(1, 2, 3, 0).contiguous().numpy()
        return [frame for frame in frames]

    array = np.asarray(sample)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            array = array[None, ...]
        else:
            array = array[..., None]
    if array.ndim != 4:
        raise ValueError(
            f"Unsupported numpy sample shape for frame extraction: {tuple(array.shape)}"
        )
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return [frame for frame in array]


def build_server_kwargs(args: argparse.Namespace, *, variant: str) -> dict[str, Any]:
    component_paths = parse_component_overrides(
        getattr(args, f"{variant}_component_path") or []
    )
    transformer_path = getattr(args, f"{variant}_transformer_path")

    kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "backend": args.backend,
        "num_gpus": args.num_gpus,
        "dit_cpu_offload": args.dit_cpu_offload,
        "dit_layerwise_offload": args.dit_layerwise_offload,
        "text_encoder_cpu_offload": args.text_encoder_cpu_offload,
        "vae_cpu_offload": args.vae_cpu_offload,
        "pin_cpu_memory": args.pin_cpu_memory,
        "enable_cfg_parallel": args.enable_cfg_parallel,
        "ulysses_degree": args.ulysses_degree,
    }
    if args.sp_degree is not None:
        kwargs["sp_degree"] = args.sp_degree
    if transformer_path is not None:
        kwargs["transformer_weights_path"] = transformer_path
    if component_paths:
        kwargs["component_paths"] = component_paths
    return kwargs


def build_sampling_kwargs(
    args: argparse.Namespace, *, output_dir: str | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "return_frames": True,
        "return_trajectory_latents": True,
        "return_trajectory_decoded": args.return_trajectory_decoded,
        "save_output": output_dir is not None,
    }
    if output_dir is not None:
        kwargs["output_path"] = output_dir
    if args.num_frames is not None:
        kwargs["num_frames"] = args.num_frames
    if args.guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    return kwargs


def run_variant(
    *,
    server_kwargs: dict[str, Any],
    sampling_kwargs: dict[str, Any],
):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    with DiffGenerator.from_pretrained(local_mode=True, **server_kwargs) as generator:
        result = generator.generate(sampling_params_kwargs=sampling_kwargs)

    if isinstance(result, list):
        if len(result) != 1:
            raise ValueError(
                f"Expected a single generation result, got {len(result)} results."
            )
        result = result[0]
    if result is None:
        raise RuntimeError("Generation returned no result.")
    return result


def _to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(result, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--sp-degree", type=int)
    parser.add_argument("--trajectory-step-index", type=int, default=-1)
    parser.add_argument("--reference-transformer-path")
    parser.add_argument("--candidate-transformer-path")
    parser.add_argument(
        "--reference-component-path",
        action="append",
        default=[],
        help="Repeatable component override in the form component=path.",
    )
    parser.add_argument(
        "--candidate-component-path",
        action="append",
        default=[],
        help="Repeatable component override in the form component=path.",
    )
    parser.add_argument("--save-output-dir")
    parser.add_argument(
        "--return-trajectory-decoded",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--dit-layerwise-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pin-cpu-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-cfg-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    save_output_dir = None
    if args.save_output_dir is not None:
        save_output_dir = Path(args.save_output_dir).expanduser().resolve()
        save_output_dir.mkdir(parents=True, exist_ok=True)

    reference_server_kwargs = build_server_kwargs(args, variant="reference")
    candidate_server_kwargs = build_server_kwargs(args, variant="candidate")
    reference_sampling_kwargs = build_sampling_kwargs(
        args,
        output_dir=(
            str(save_output_dir / "reference") if save_output_dir is not None else None
        ),
    )
    candidate_sampling_kwargs = build_sampling_kwargs(
        args,
        output_dir=(
            str(save_output_dir / "candidate") if save_output_dir is not None else None
        ),
    )

    reference_result = run_variant(
        server_kwargs=reference_server_kwargs,
        sampling_kwargs=reference_sampling_kwargs,
    )
    candidate_result = run_variant(
        server_kwargs=candidate_server_kwargs,
        sampling_kwargs=candidate_sampling_kwargs,
    )

    trajectory_metrics = summarize_trajectory_metrics(
        reference_result.trajectory_latents,
        candidate_result.trajectory_latents,
        reference_timesteps=getattr(reference_result, "timesteps", None),
        candidate_timesteps=getattr(candidate_result, "timesteps", None),
        step_index=args.trajectory_step_index,
    )
    frame_metrics = summarize_output_frame_metrics(
        extract_result_frames(reference_result),
        extract_result_frames(candidate_result),
    )

    report = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "reference_server_kwargs": reference_server_kwargs,
        "candidate_server_kwargs": candidate_server_kwargs,
        "trajectory_metrics": trajectory_metrics,
        "frame_metrics": frame_metrics,
        "reference_output_file_path": reference_result.output_file_path,
        "candidate_output_file_path": candidate_result.output_file_path,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_to_jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(_to_jsonable(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
