import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "sglang-explainer-video" / "scripts"
spec = importlib.util.spec_from_file_location("check_links", SCRIPTS / "check_links.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


def test_finds_links_on_the_last_frame():
    html = (
        '<span class="fs-links-url">github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/mla_buffer.py</span>'
        '<span class="fs-links-url">docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash</span>'
        '<span class="fs-follow-url">github.com/sgl-project/sglang</span>'
        '<span class="fs-follow-url">sglang.io/blog</span><span class="fs-follow-url">docs.sglang.io</span>'
        '<div id="f13-prlink">github.com/sgl-project/sglang/pull/47013</div>'
        '<script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>'
    )
    assert L.find_links(html) == [
        "github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/mla_buffer.py",
        "docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash",
        "github.com/sgl-project/sglang",
        "sglang.io/blog",
        "docs.sglang.io",
        "github.com/sgl-project/sglang/pull/47013",
    ]


def test_finds_links_in_publish_text():
    text = "源码：https://github.com/sgl-project/sglang/tree/main/python/sglang/srt。\nCookbook：https://cookbook.sglang.io，欢迎关注。"
    assert L.find_links(text) == ["github.com/sgl-project/sglang/tree/main/python/sglang/srt", "cookbook.sglang.io"]
