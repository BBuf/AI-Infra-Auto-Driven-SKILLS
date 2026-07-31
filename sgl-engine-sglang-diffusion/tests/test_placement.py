from pathlib import Path

import pytest

from sgl_engine_sglang_diffusion.placement import (
    KernelLayout,
    UnsupportedSGLangLayout,
    detect_placement_contract,
)


def test_detects_current_unified_layout_and_renders_canonical_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sglang"
    (root / "python/sglang/kernels/jit").mkdir(parents=True)
    (root / "python/sglang/kernels/ops/diffusion").mkdir(parents=True)
    (root / "python/sglang/kernels/README.md").write_text("unified\n", encoding="utf-8")

    contract = detect_placement_contract(root)
    rendered = contract.render(model_slug="wan-a14b")

    assert contract.layout is KernelLayout.UNIFIED
    assert contract.aot_root == "python/sglang/kernels/aot"
    assert "python/sglang/kernels/ops/diffusion/agent/wan-a14b/" in rendered
    assert "python/sglang/kernels/jit/csrc/diffusion/agent/wan-a14b/" in rendered
    assert "python/sglang/kernels/aot/csrc/diffusion/agent/wan-a14b/" in rendered
    assert "sglang.kernels.ops namespace" in rendered
    assert "--quality off|auto|<profile-id>" in rendered


def test_detects_legacy_jit_layout_without_using_unlocked_host_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sglang"
    (root / "python/sglang/jit_kernel").mkdir(parents=True)
    (root / "sgl-kernel").mkdir()

    contract = detect_placement_contract(root)

    assert contract.layout is KernelLayout.LEGACY_JIT
    assert contract.jit_source_root == ("python/sglang/jit_kernel/csrc/diffusion/agent")
    assert contract.aot_root == "sgl-kernel"


def test_unknown_layout_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedSGLangLayout, match="neither the unified"):
        detect_placement_contract(tmp_path)
