"""Materialize the self-contained SGLang Diffusion optimization catalog."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_METHOD_IDS = {
    "cache": (
        "whole_step_denoiser_output_reuse",
        "teacache_style_timestep_aware_reuse",
        "easycache_style_runtime_adaptive_transform_vector_reuse",
        "pab_style_attention_broadcast",
        "block_layer_feature_caching",
        "fora_style_transformer_layer_caching",
        "token_wise_feature_caching",
        "cfg_aware_feature_caching",
        "content_or_motion_adaptive_schedules",
        "predictive_delta_or_forecast_caching",
        "architecture_aware_feature_reuse",
    ),
    "token_pruning": (
        "token_pruning",
        "tome_style_token_merging",
        "importance_preserving_token_merging",
        "token_masking_compute_masking",
        "region_aware_token_reduction",
        "attention_guided_token_reduction",
        "dynamics_aware_token_pruning",
        "cluster_aware_token_pruning",
        "dynamic_token_density_control",
        "video_token_carving",
        "context_reference_token_pruning",
        "token_wise_feature_caching",
        "conservative_aggressive_dual_token_policies",
    ),
    "quantization": (
        "conservative_ffn_only_nvfp4",
        "selective_hot_linear_nvfp4",
        "transformer_engine_recipe_variants",
        "dense_guard_policies",
        "backend_and_padding_policy",
        "fused_epilogue_paths",
    ),
    "sparse_attention": (
        "piecewise_exact_block_sparse_attention",
        "spatial_temporal_head_routing",
        "semantic_aware_token_permutation",
        "online_precise_search_with_mask_reuse",
        "proxy_or_universal_mask_prediction",
        "rotating_anchors_and_long_video_windows",
        "layer_profiling_and_qk_coclustering",
        "head_wise_adaptive_budgets",
        "dynamic_attention_pattern_selection",
    ),
    "kernel": (
        "gemm_epilogue_fusion",
        "norm_modulation_and_residual_fusion",
        "attention_adjacent_fusion",
        "compile_and_graph_capture",
        "memory_layout_and_copy_elimination",
        "launch_overhead_reduction",
        "overlap_streams_and_pipeline_scheduling",
        "decode_vae_and_postprocess_fusion",
    ),
    "topology": (
        "context_and_sequence_parallelism",
        "tensor_parallelism",
        "expert_parallelism",
        "parameter_sharding_and_residency",
        "cfg_execution",
        "device_mesh_and_rank_mapping",
        "collective_scheduling",
    ),
}

_CANDIDATE_SPECS = {
    "cache": (
        (
            "adaptive_delta_forecast",
            "predictive_delta_cache",
            ("has_denoise_steps", "supports_step_cache"),
        ),
        (
            "attention_broadcast",
            "attention_broadcast",
            ("has_attention_layers", "supports_step_cache"),
        ),
        (
            "block_layer_feature_cache",
            "block_layer_feature_cache",
            ("has_transformer_blocks", "supports_step_cache"),
        ),
        (
            "scheduled_step_reuse",
            "whole_step_cache",
            ("has_denoise_steps", "supports_step_cache"),
        ),
        (
            "teacache_signal_reuse",
            "timestep_aware_cache",
            ("has_denoise_steps", "supports_step_cache"),
        ),
    ),
    "token_pruning": (
        (
            "cluster_representative_update",
            "cluster_representative",
            (
                "has_spatiotemporal_token_layout",
                "has_token_sequence_axis",
                "supports_token_gather_scatter",
            ),
        ),
        (
            "feature_norm_prune",
            "feature_norm",
            (
                "has_spatiotemporal_token_layout",
                "has_token_sequence_axis",
                "supports_token_gather_scatter",
            ),
        ),
        (
            "region_dynamic_density",
            "region_density",
            (
                "has_spatiotemporal_token_layout",
                "has_token_sequence_axis",
                "supports_token_gather_scatter",
            ),
        ),
        (
            "shape_stable_compute_mask",
            "shape_stable_compute_mask",
            (
                "has_spatiotemporal_token_layout",
                "has_token_sequence_axis",
                "supports_token_gather_scatter",
            ),
        ),
        (
            "tome_merge_restore",
            "token_merge_restore",
            (
                "has_spatiotemporal_token_layout",
                "has_token_sequence_axis",
                "supports_token_gather_scatter",
            ),
        ),
    ),
    "quantization": tuple(
        (
            candidate_id,
            family,
            ("has_ffn_linear_modules", "supports_native_low_precision_linear"),
        )
        for candidate_id, family in (
            ("backend_padding_policy", "backend_padding_policy"),
            ("conservative_ffn_low_precision", "conservative_ffn"),
            ("dense_guard_policy", "dense_guard_policy"),
            ("profiled_hot_linear_low_precision", "profiled_hot_linear"),
            ("backend_recipe_variant", "backend_recipe_variant"),
        )
    ),
    "sparse_attention": tuple(
        (
            candidate_id,
            family,
            (
                "has_attention_backend_switch",
                "has_attention_layers",
                "has_spatiotemporal_token_layout",
            ),
        )
        for candidate_id, family in (
            ("dynamic_pattern_probe", "dynamic_patterns"),
            ("headwise_adaptive_budgets", "headwise_topk_budget"),
            ("online_mask_search_reuse", "online_mask_reuse"),
            ("piecewise_exact_blocks", "piecewise_sparse"),
            ("proxy_mask_prediction", "proxy_mask_prediction"),
            ("qk_coclustering", "qk_similarity_block_map"),
            ("rotating_anchor_windows", "rotating_anchor_windows"),
            ("semantic_permutation", "semantic_permutation"),
            ("spatial_temporal_head_routing", "head_routing"),
        )
    ),
    "kernel": (
        (
            "backend_selection_probe",
            "backend_selection",
            ("has_attention_backend_switch", "has_attention_layers"),
        ),
        (
            "compile_graph_capture",
            "compile_or_cuda_graph",
            ("has_transformer_blocks", "supports_cuda_graph_probe"),
        ),
        (
            "existing_fast_path_audit",
            "existing_fast_paths",
            ("has_transformer_blocks",),
        ),
        (
            "gemm_epilogue_fusion",
            "gemm_epilogue_fusion",
            ("has_ffn_linear_modules", "has_transformer_blocks"),
        ),
        (
            "layout_copy_elimination",
            "layout_copy_elimination",
            ("has_transformer_blocks",),
        ),
        (
            "norm_modulation_residual_fusion",
            "norm_modulation_residual_fusion",
            ("has_transformer_blocks",),
        ),
    ),
    "topology": (),
}

_COMPOSITION_RECIPES = (
    ("kernel_only", ("kernel",)),
    ("cache_only", ("cache",)),
    ("sparse_attention_only", ("sparse_attention",)),
    ("quantization_only", ("quantization",)),
    ("token_pruning_only", ("token_pruning",)),
    ("kernel_cache", ("kernel", "cache")),
    (
        "kernel_cache_sparse_attention",
        ("kernel", "cache", "sparse_attention"),
    ),
    (
        "compatible_full_stack",
        (
            "kernel",
            "cache",
            "sparse_attention",
            "quantization",
            "token_pruning",
            "topology",
        ),
    ),
)


def build_search_space_catalog(*, output_path: Path) -> dict[str, Any]:
    """Write the bundled opportunity catalog without consulting another repo."""
    families: dict[str, dict[str, Any]] = {}
    for family, method_ids in _METHOD_IDS.items():
        methods = [
            {
                "id": method_id,
                "title": method_id.replace("_", " "),
                "coverage_status": "documented",
            }
            for method_id in method_ids
        ]
        candidates = [
            {
                "id": candidate_id,
                "candidate_family": candidate_family,
                "required_capabilities": list(capabilities),
                "coverage_status": "referenced",
            }
            for candidate_id, candidate_family, capabilities in _CANDIDATE_SPECS[family]
        ]
        families[family] = {
            "methods": methods,
            "candidates": candidates,
            "review_items": [
                *[f"method:{item['id']}" for item in methods],
                *[f"candidate:{item['id']}" for item in candidates],
            ],
        }

    recipes = [
        {
            "id": recipe_id,
            "techniques": list(techniques),
            "coverage_status": "referenced",
        }
        for recipe_id, techniques in _COMPOSITION_RECIPES
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source": "bundled_sglang_diffusion_catalog",
        "catalog_version": 1,
        "families": families,
        "candidate_count": sum(len(value["candidates"]) for value in families.values()),
        "recipes": recipes,
        "recipe_count": len(recipes),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return payload
