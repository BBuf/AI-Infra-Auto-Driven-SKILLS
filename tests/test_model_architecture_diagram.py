import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "skills"
    / "model-architecture-diagram"
    / "scripts"
    / "model_architecture_diagram.py"
)


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_existing_deepseek_v3_does_not_match_v32() -> None:
    result = run_script("DeepSeek-V3").stdout

    assert "DeepSeek V3 architecture" in result
    assert "DeepSeek V3.2 architecture" not in result


def test_existing_glm5_returns_raw_image() -> None:
    result = json.loads(run_script("zai-org/GLM-5.1-FP8", "--format", "json").stdout)

    assert result["kind"] == "existing"
    assert result["diagrams"][0]["source"] == "InfraTech"
    assert result["diagrams"][0]["url"].startswith("https://raw.githubusercontent.com/")


def test_force_generate_writes_svg_and_mermaid(tmp_path: Path) -> None:
    result = json.loads(
        run_script(
            "Qwen3-Coder-Next",
            "--force-generate",
            "--format",
            "json",
            "--output-dir",
            str(tmp_path),
        ).stdout
    )

    assert result["kind"] == "generated"
    assert result["template"] == "hybrid_delta_moe"
    assert Path(result["svg_path"]).is_file()
    assert Path(result["mermaid_path"]).is_file()
    svg = Path(result["svg_path"]).read_text(encoding="utf-8")
    assert "architecture-grid" in svg
    assert "Hybrid block cycle" in svg
    assert "Gated DeltaNet" in svg


def test_nearby_versions_do_not_match_wrong_originals() -> None:
    qwen3 = json.loads(run_script("Qwen3", "--format", "json").stdout)
    minimax_m27 = json.loads(run_script("MiniMax-M2.7", "--format", "json").stdout)
    kimi_k26 = json.loads(run_script("Kimi-K2.6", "--format", "json").stdout)

    assert qwen3["kind"] == "generated"
    assert qwen3["template"] == "decoder_moe"
    assert minimax_m27["kind"] == "generated"
    assert minimax_m27["template"] == "decoder_moe"
    assert kimi_k26["kind"] == "generated"
    assert kimi_k26["template"] == "moonvit_mla_moe"


def test_ocr_uses_vlm_template() -> None:
    result = json.loads(run_script("DeepSeek-OCR", "--format", "json").stdout)

    assert result["kind"] == "generated"
    assert result["template"] == "vlm"


def test_deepseek_v4_uses_dedicated_mhc_mqa_template(tmp_path: Path) -> None:
    result = json.loads(
        run_script(
            "DeepSeek-V4",
            "--force-generate",
            "--format",
            "json",
            "--output-dir",
            str(tmp_path),
        ).stdout
    )

    assert result["kind"] == "generated"
    assert result["template"] == "deepseek_v4_mhc_mqa_moe"
    svg = Path(result["svg_path"]).read_text(encoding="utf-8")
    assert "MHC residual mixer" in svg
    assert "Compressed MQA attention" in svg
    assert "C4 Indexer" in svg
    assert "top-6 experts" in svg


def test_recent_architecture_overrides_match_code_surfaces() -> None:
    glm_flash = json.loads(run_script("GLM-4.7-Flash", "--format", "json").stdout)
    ministral = json.loads(run_script("Ministral-3", "--format", "json").stdout)
    nemotron = json.loads(
        run_script("NVIDIA Nemotron3-Super", "--format", "json").stdout
    )
    kimi_linear = json.loads(run_script("Kimi-Linear", "--format", "json").stdout)
    ling = json.loads(run_script("Ling-2.5-1T", "--format", "json").stdout)

    assert glm_flash["kind"] == "generated"
    assert glm_flash["template"] == "mla_moe"
    assert ministral["kind"] == "generated"
    assert ministral["template"] == "vlm"
    assert nemotron["kind"] == "generated"
    assert nemotron["template"] == "hybrid_mamba_moe_attention"
    assert kimi_linear["kind"] == "generated"
    assert kimi_linear["template"] == "hybrid_linear_mla_moe"
    assert ling["kind"] == "generated"
    assert ling["template"] == "hybrid_linear_mla_moe"
