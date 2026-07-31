from __future__ import annotations

import json
from pathlib import Path

from sgl_engine_sglang_diffusion.search_space import build_search_space_catalog


def test_bundled_catalog_is_complete_and_self_contained(tmp_path: Path) -> None:
    output = tmp_path / "SEARCH-SPACE.json"
    catalog = build_search_space_catalog(output_path=output)

    assert {
        family: len(value["methods"]) for family, value in catalog["families"].items()
    } == {
        "cache": 11,
        "token_pruning": 13,
        "quantization": 6,
        "sparse_attention": 9,
        "kernel": 8,
        "topology": 7,
    }
    assert catalog["candidate_count"] == 30
    assert catalog["recipe_count"] == 8
    serialized = json.dumps(catalog)
    assert "github.com" not in serialized
    assert "source_path" not in serialized
    assert output.is_file()
