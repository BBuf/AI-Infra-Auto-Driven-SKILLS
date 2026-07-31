from __future__ import annotations

from pathlib import Path


def test_campaign_package_has_no_nested_ai_execution_path() -> None:
    package = Path(__file__).resolve().parents[1] / "src/sgl_engine_sglang_diffusion"
    forbidden = (
        "AgentRunner",
        "ExecutorManager",
        "build_agent_argv",
        "codex exec",
        "claude",
    )
    matches: list[str] = []
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for marker in forbidden:
            if marker.lower() in text:
                matches.append(f"{path.name}: {marker}")
    assert matches == []
