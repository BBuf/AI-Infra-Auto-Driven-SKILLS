import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("jieba")

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "sglang-explainer-video" / "scripts"
spec = importlib.util.spec_from_file_location("seg_check", SCRIPTS / "seg_check.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def take(*tokens):
    words, t = [], 0.0
    for tok in tokens:
        words.append({"text": tok, "start": t, "end": t + 0.2})
        t += 0.2
    return {"voices": [{"frame": 9, "words": words}]}


def test_number_and_zhi_in_one_token_is_a_classifier():
    # The take the user flagged in the PR 43570 video: 「乘 24 只要」 read 二十四只（zhī）| 要.
    hits = C.check(take("一共", "1136", "项，", "乘", "24 只", "要", "27264 个", "CTA。"), 0.06)
    assert [h[1] for h in hits] == ["classifier"]


def test_zhi_kept_with_its_verb_is_fine():
    # PR 28185 video: 'int8' | '只能' and 'fp8' | '只有' read right.
    assert C.check(take("int8", "只能", "存", "负", "127"), 0.06) == []
    assert C.check(take("fp8", "只有", "9", "档；"), 0.06) == []
