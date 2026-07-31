"""Revision-aware SGLang kernel placement and registration contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class UnsupportedSGLangLayout(RuntimeError):
    """The locked SGLang checkout does not expose a supported kernel layout."""


class KernelLayout(StrEnum):
    UNIFIED = "unified-kernels"
    LEGACY_JIT = "legacy-jit-kernel"


@dataclass(frozen=True)
class PlacementContract:
    layout: KernelLayout
    profile_root: str
    wrapper_root: str
    jit_source_root: str
    aot_root: str
    test_root: str
    benchmark_root: str

    def render(self, *, model_slug: str) -> str:
        """Render work-order rules whose paths match the locked checkout."""
        profile_root = f"{self.profile_root}/{model_slug}"
        wrapper_root = f"{self.wrapper_root}/{model_slug}"
        jit_source_root = f"{self.jit_source_root}/{model_slug}"
        aot_source_root = f"{self.aot_root}/csrc/diffusion/agent/{model_slug}"
        aot_include_root = f"{self.aot_root}/include/diffusion/agent/{model_slug}"
        aot_python_root = (
            f"{self.aot_root}/python/sgl_kernel/diffusion/agent/{model_slug}"
        )
        return f"""\
Locked SGLang kernel layout: {self.layout.value}.

Agent-owned model profiles, dispatch policy, manifests, and engagement
receipts belong below {profile_root}/. Callable Python operator wrappers belong
below {wrapper_root}/ and runtime code must import them through the canonical
sglang.kernels.ops namespace. Lightweight JIT CUDA sources belong below
{jit_source_root}/. Agent-generated heavyweight AOT/CUTLASS implementation,
header, and Python implementation files belong below {aot_source_root}/,
{aot_include_root}/, and {aot_python_root}/. They must also complete the
{self.aot_root}/ declaration, torch-op registration, sorted build-source,
Python wrapper/export, test, benchmark, and wheel-build steps.

GPU correctness tests belong below {self.test_root}/; focused kernel
benchmarks belong below {self.benchmark_root}/. A microbenchmark can justify a
hypothesis but cannot establish campaign speedup.

Before adding an operator, reuse an existing canonical SGLang operator when
possible. In the unified layout, register metadata without importing torch or
triggering compilation, provide a pure native reference, and use the
BaseFusedOp eligibility/fallback contract for alternative backends. Multiple
registry backends are inventory, not an implicit priority policy.

Every optimized path needs exact model/hardware/shape/dtype eligibility,
native fallback or declared hard error, positive engagement and fallback
counters, parity tests, and a benchmark. The delivered patch must expose
--agent-optimization off|auto|<profile-id>. `off` is the identity path for the
locked source revision. `auto` may activate only when the immutable profile
matches. Native SGLang backend logs are mandatory; any Diffusers fallback
invalidates performance evidence.
"""


def detect_placement_contract(checkout: Path) -> PlacementContract:
    """Select paths from the locked source tree; never guess a future layout."""
    checkout = checkout.resolve()
    unified = checkout / "python/sglang/kernels"
    if (
        (unified / "README.md").is_file()
        and (unified / "jit").is_dir()
        and (unified / "ops/diffusion").is_dir()
    ):
        return PlacementContract(
            layout=KernelLayout.UNIFIED,
            profile_root="python/sglang/kernels/agent/diffusion",
            wrapper_root="python/sglang/kernels/ops/diffusion/agent",
            jit_source_root="python/sglang/kernels/jit/csrc/diffusion/agent",
            aot_root="python/sglang/kernels/aot",
            test_root="test/registered/kernels/ops/diffusion/agent",
            benchmark_root="test/registered/kernels/benchmark/diffusion/agent",
        )

    legacy = checkout / "python/sglang/jit_kernel"
    if legacy.is_dir():
        aot_root = (
            "sgl-kernel"
            if (checkout / "sgl-kernel").is_dir()
            else "python/sglang/kernels/aot"
        )
        return PlacementContract(
            layout=KernelLayout.LEGACY_JIT,
            profile_root="python/sglang/kernels/agent/diffusion",
            wrapper_root="python/sglang/kernels/agent/diffusion/ops",
            jit_source_root="python/sglang/jit_kernel/csrc/diffusion/agent",
            aot_root=aot_root,
            test_root="test/registered/jit/diffusion/agent",
            benchmark_root="benchmark/kernels/diffusion/agent",
        )

    raise UnsupportedSGLangLayout(
        "locked SGLang checkout has neither the unified "
        "python/sglang/kernels/{jit,ops} layout nor the legacy "
        "python/sglang/jit_kernel layout"
    )
