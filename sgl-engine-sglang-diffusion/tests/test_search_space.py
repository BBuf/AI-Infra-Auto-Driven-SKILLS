from __future__ import annotations

from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.search_space import (
    SearchSpaceError,
    build_sol_search_space_catalog,
)


DOCUMENTS = {
    "01_cache.md": "- Whole-step reuse: cache output.\n",
    "02_token_pruning.md": "- Token pruning: remove tokens.\n",
    "03_quantization.md": "### 1. Conservative FP8\n",
    "04_sparse_attention.md": "### 1. Piecewise PISA\n",
    "05_kernel_fusion.md": "### 1. GEMM Epilogue Fusion\n",
    "06_parallel_topology.md": "- Context parallelism: shard sequence.\n",
}


def _fake_sol(root: Path) -> Path:
    for name, methods in DOCUMENTS.items():
        path = root / "search_space" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Search\n\n## Method Families\n\n{methods}\n## Search Axes\n")
    implementation = root / "techniques/transforms/sparse_attention.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text(
        '@register_transform("sparse_attention")\n'
        "class SparseAttention:\n"
        "    pass\n"
    )
    method = root / "techniques/methods/step_cache.py"
    method.parent.mkdir(parents=True)
    method.write_text(
        '@register_technique("step_cache")\n'
        "class StepCache:\n"
        "    pass\n"
    )
    candidate = root / "candidates/sparse_attention/piecewise.toml"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        """
kind = "methodology"
purpose = "frontier"
description = "PISA"
model_profile = "cosmos3"

[id]
name = "piecewise"
dimension = "sparse_attention"
family = "pisa"

[references.local]
generic_impl = "techniques/transforms/sparse_attention.py"

[requirements]
capabilities = ["has_attention_backend_switch"]

[efficiency]
kind = "build_transform"
name = "sparse_attention"

[verification]
mode = "gpu"
quality_gate = "baseline-comparable"
"""
    )
    recipe = root / "candidates/cosmos_full.toml"
    recipe.write_text(
        'id = "cosmos_full"\n'
        'kind = "patch"\n'
        'purpose = "optimization"\n'
        'model_profile = "cosmos3"\n'
    )
    site = root / "site_docs/techniques/sparse/pisa.md"
    site.parent.mkdir(parents=True)
    site.write_text("# PISA\n")
    return root


def test_catalog_discovers_families_candidates_recipes_and_implementations(
    tmp_path: Path,
) -> None:
    sol = _fake_sol(tmp_path / "sol")
    output = tmp_path / "SEARCH-SPACE.json"

    catalog = build_sol_search_space_catalog(
        sol_checkout=sol,
        sol_commit="a" * 40,
        output_path=output,
    )

    assert set(catalog["families"]) == {
        "kernel",
        "cache",
        "sparse_attention",
        "quantization",
        "token_pruning",
        "topology",
    }
    assert all(
        method["coverage_status"] == "documented"
        for family in catalog["families"].values()
        for method in family["methods"]
    )
    assert all(
        candidate["coverage_status"] == "referenced"
        for family in catalog["families"].values()
        for candidate in family["candidates"]
    )
    assert all(
        recipe["coverage_status"] == "referenced"
        for recipe in catalog["recipes"]
    )
    assert all(
        implementation["coverage_status"] == "referenced"
        for implementation in catalog["implementations"]
    )
    assert catalog["candidate_count"] == 1
    assert catalog["recipe_count"] == 1
    sparse = catalog["families"]["sparse_attention"]
    assert sparse["candidates"][0]["required_capabilities"] == [
        "has_attention_backend_switch"
    ]
    assert sparse["candidates"][0]["implementation_registered"] is True
    assert "method:piecewise_pisa" in sparse["review_items"]
    assert "candidate:piecewise" in sparse["review_items"]
    assert output.is_file()


def test_catalog_rejects_missing_canonical_document(tmp_path: Path) -> None:
    sol = _fake_sol(tmp_path / "sol")
    (sol / "search_space/06_parallel_topology.md").unlink()

    with pytest.raises(SearchSpaceError, match="missing canonical"):
        build_sol_search_space_catalog(
            sol_checkout=sol,
            sol_commit="a" * 40,
            output_path=tmp_path / "catalog.json",
        )


def test_catalog_rejects_unknown_candidate_dimension(tmp_path: Path) -> None:
    sol = _fake_sol(tmp_path / "sol")
    manifest = sol / "candidates/sparse_attention/piecewise.toml"
    manifest.write_text(
        manifest.read_text().replace(
            'dimension = "sparse_attention"', 'dimension = "unknown"'
        )
    )

    with pytest.raises(SearchSpaceError, match="unknown Sol candidate dimension"):
        build_sol_search_space_catalog(
            sol_checkout=sol,
            sol_commit="a" * 40,
            output_path=tmp_path / "catalog.json",
        )


def test_catalog_rejects_missing_generic_implementation(tmp_path: Path) -> None:
    sol = _fake_sol(tmp_path / "sol")
    (sol / "techniques/transforms/sparse_attention.py").unlink()

    with pytest.raises(SearchSpaceError, match="references missing implementation"):
        build_sol_search_space_catalog(
            sol_checkout=sol,
            sol_commit="a" * 40,
            output_path=tmp_path / "catalog.json",
        )
