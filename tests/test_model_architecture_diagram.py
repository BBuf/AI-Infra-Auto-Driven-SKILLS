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


def test_nearby_versions_without_originals_do_not_match_wrong_originals() -> None:
    qwen36 = json.loads(run_script("Qwen3.6", "--format", "json").stdout)
    minimax_m27 = json.loads(run_script("MiniMax-M2.7", "--format", "json").stdout)
    kimi_k26 = json.loads(run_script("Kimi-K2.6", "--format", "json").stdout)

    assert qwen36["kind"] == "no_match"
    assert minimax_m27["kind"] == "no_match"
    assert kimi_k26["kind"] == "no_match"


def test_newly_indexed_models_return_public_originals() -> None:
    deepseek_v4 = json.loads(run_script("DeepSeek-V4", "--format", "json").stdout)
    qwen3 = json.loads(run_script("Qwen3", "--format", "json").stdout)

    assert deepseek_v4["kind"] == "existing"
    assert deepseek_v4["diagrams"][0]["title"] == "DeepSeek V4 architecture"
    assert qwen3["kind"] == "existing"
    assert [item["title"] for item in qwen3["diagrams"]] == [
        "Qwen3 model structure",
        "Qwen3 MoE structure",
        "Qwen3 shared expert comparison",
    ]


def test_unindexed_models_return_no_match() -> None:
    deepseek_ocr = json.loads(run_script("DeepSeek-OCR", "--format", "json").stdout)
    qwen_image = json.loads(run_script("Qwen-Image", "--format", "json").stdout)

    assert deepseek_ocr["kind"] == "no_match"
    assert qwen_image["kind"] == "no_match"
    assert "No public original architecture diagram" in qwen_image["message"]


def test_list_known_returns_indexed_originals() -> None:
    result = run_script("--list-known").stdout

    assert "DeepSeek V3 architecture" in result
    assert "Qwen3-VL" in result
