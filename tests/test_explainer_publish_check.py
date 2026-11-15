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


def test_sglang_text_needs_source_and_cookbook_links():
    desc = (
        "SGLang PR 47013：一半回答变成一串感叹号。\n"
        "源码：https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/mla_buffer.py\n"
        "Cookbook：https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash"
    )
    assert C.link_problems(desc, xhs=False) == []
    only_repo = "SGLang PR 47013。GitHub：https://github.com/sgl-project/sglang 文档：https://docs.sglang.io"
    assert len(C.link_problems(only_repo, xhs=False)) == 2
    assert C.link_problems("SGLang PR 47013。Cookbook：cookbook.sglang.io", xhs=False) == [
        "no source link (github.com/sgl-project/<repo>/blob|tree/…)"
    ]


def test_xhs_body_names_the_links_in_plain_text():
    body = "SGLang PR 47013\n源码：sgl-project/sglang 仓库 python/sglang/kernels/ops/kvcache/mla_buffer.py\nCookbook：SGLang Cookbook · GLM-5.3-Flash"
    assert C.link_problems(body, xhs=True) == []
    assert C.link_problems("SGLang PR 47013，讲一个 NaN 的 bug。", xhs=True) == ["no 源码 line", "no Cookbook / 文档 line"]


def test_links_not_required_when_the_video_is_not_about_sglang():
    assert C.link_problems("DeepSeek 的推理引擎怎么省显存。配音为 AI 语音合成。", xhs=False) == []


def test_model_name_is_matched_loosely():
    assert C.names_model("GLM-5.3-Flash一半回答变感叹号", "GLM-5.3")
    assert C.names_model("glm 5.3 flash 的 KV cache", "GLM-5.3-Flash")
    assert C.names_model("DeepSeek首token延迟降了近一半", "deepseek")
    assert not C.names_model("一半回答满屏感叹号，只因一个NaN", "GLM-5.3")
