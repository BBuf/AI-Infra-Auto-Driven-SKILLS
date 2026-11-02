"""Where does the recorded voice cut a Chinese sentence in the wrong place?

Edge's WordBoundary tokens show how the voice segmented each line. Two kinds of mismatch with how a
Chinese reader parses the sentence (jieba) are audible:

  straddle  a voice token crosses a word boundary on one side and cuts a word on the other, e.g.
            值相同时下标 → '时下'|'标' (read as 时下 "nowadays"), 下标取反放在 → '标取'|'反放',
            在 kernel 里直接取 → '里直'|'接取', 得到一个新张量 → '新张'|'量'
  pause     a pause of ≥ --gap seconds inside one word, e.g. 1 除以 T → '除' 100 ms '以'

Fix each hit in the narration (a comma, 的, or a reword), re-synthesize that line, and run the check
again. Number + 个 tokens (7168个) and per-character splits without a pause are normal and not reported.

    $SEV_PY -I $SEV_SKILL/scripts/seg_check.py --meta $P/audio_meta.json [--gap 0.06]
"""

import argparse
import json
import re
import sys

try:
    import jieba
except ImportError:  # pragma: no cover
    sys.exit("✗ seg_check needs jieba: re-run setup_env.sh (it installs jieba into the venv)")

CJK = re.compile(r"[\u4e00-\u9fff]")
TRAIL = re.compile(r"[，。：；、？！）（「」\s]+$")
# Domain words jieba would otherwise split, so a token boundary inside them is not a false alarm.
WORDS = ["下标", "张量", "寄存器", "读数据", "整行", "一整行", "编译期", "线程", "最大值", "最小值", "取反", "队头",
         "无穷大", "两两相加"]
# Words whose cut is natural speech (两|个数, 第几|行, 取|出来, 再分|给): not reported.
BENIGN_CUT = {"个数", "几行", "出来", "取出", "分给", "上来", "下来", "进去", "相加", "两个", "几个", "三个"}


def check(meta, gap_min):
    jieba.setLogLevel(60)
    for w in WORDS:
        jieba.add_word(w, freq=100000)
    hits = []
    for v in meta["voices"]:
        words = v["words"]
        toks = [w["text"].replace(" ", "") for w in words]
        text = "".join(toks)
        bounds, p = {0}, 0
        spans = []
        for w in jieba.lcut(text, HMM=False):
            spans.append((p, p + len(w), w))
            p += len(w)
            bounds.add(p)
        pos, starts = 0, []
        for t in toks:
            starts.append(pos)
            pos += len(t)
        for k, t in enumerate(toks):
            a = starts[k]
            core = TRAIL.sub("", t)
            b = a + len(core)
            if len(CJK.findall(core)) >= 2:
                inner = any(c in bounds for c in range(a + 1, b))
                cut = [w for s, e, w in spans if s < a < e or s < b < e]
                if inner and cut and not all(w in BENIGN_CUT for w in cut):
                    hits.append((v["frame"], "straddle", f"{t!r} cuts 「{'」「'.join(cut)}」", text[max(0, a - 8): b + 6]))
            if k + 1 < len(toks):
                cut, gap = starts[k + 1], words[k + 1]["start"] - words[k]["end"]
                if gap >= gap_min:
                    for s, e, w in spans:
                        if s < cut < e and CJK.search(w) and len(w) >= 2:
                            hits.append((v["frame"], "pause", f"「{w}」 {toks[k]!r}|{toks[k + 1]!r} {gap * 1000:.0f} ms",
                                         text[max(0, s - 8): e + 6]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="audio_meta.json of the recorded take")
    ap.add_argument("--gap", type=float, default=0.06, help="pause inside a word that counts (seconds)")
    a = ap.parse_args()
    hits = check(json.load(open(a.meta, encoding="utf-8")), a.gap)
    for frame, kind, what, ctx in hits:
        print(f"  line {frame}: {kind:8} {what}   …{ctx}…")
    print(f"{'✓' if not hits else '✗'} seg_check: {len(hits)} place(s) where the voice cuts a word")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
