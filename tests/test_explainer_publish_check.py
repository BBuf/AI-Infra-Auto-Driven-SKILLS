import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "sglang-explainer-video" / "scripts"
spec = importlib.util.spec_from_file_location("publish_check", SCRIPTS / "publish_check.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def test_xhs_title_counts_ascii_as_half():
    assert C.xhs_len("batch 一大，小 GEMM 为何变慢？") == 15
    assert C.xhs_len("一行 16 万个 logits，放进寄存器？") == 16.5
    assert C.xhs_len("一行 16 万个 logits，放进寄存器？再加一点点字数凑") > 20


def test_hash_before_digit_only_matters_on_xhs():
    body = "PR 47683：tiny GEMM，见 #47683"
    assert any("topic popup" in p for p in C.text_problems(body, xhs=True))
    assert C.text_problems(body, xhs=False) == []
    assert C.text_problems("PR 47683 #SGLang #大模型", xhs=True) == []


def test_pr_status_is_flagged():
    assert C.text_problems("这个 PR 已合并，B200 上快 2 倍", xhs=False)
    assert C.text_problems("Status: OPEN", xhs=False)
    assert C.text_problems("open-source kernel，合并两个 kernel", xhs=False) == []


def test_ai_voice_line():
    assert C.AI_VOICE.search("中文讲解 + 中英双语字幕；配音为 AI 语音合成。")
    assert not C.AI_VOICE.search("SGLang PR 47683 讲解，约 13 分钟。")
