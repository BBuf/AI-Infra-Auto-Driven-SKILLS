"""Check the files and the post text before publishing, so nothing wrong goes out.

Run it on the exact files you are about to upload and the exact text you are about to paste. It checks the
things the sites accept without a word, or that already went wrong once:

  the upload copy is byte-identical to the render you mean to publish (a stale Desktop copy, or the
      previous video still selected in the file dialog, was uploaded once)
  the B站 cover is 1440×1080 (4:3)
  B站 title ≤ 80 characters (the input silently keeps the old value when the text is longer)
  小红书 title ≤ 20 (ASCII counts half)
  B站 简介 ≤ 2000 and 小红书 body ≤ 1000 characters
  B站 简介 carries the AI-voice line (配音为 AI 语音合成)
  no 「#」 right before a digit in the 小红书 body (it opens the topic popup)
  no PR status (OPEN / merged / 已合并 ...) in any text

It prints the video's duration and both text lengths: compare them with what the sites show after submit
(references/publishing.md §Checks).

    $SEV_PY -I $SEV_SKILL/scripts/publish_check.py --video ~/Desktop/小红书上传/NN_slug.mp4 \\
        --render $P/renders/video.mp4 --cover ~/Desktop/小红书上传/NN_slug_封面_B站.png \\
        --bili-title "..." --xhs-title "..." --bili-desc desc.txt --xhs-body body.txt
"""

import argparse
import filecmp
import re
import subprocess
import sys

STATUS = re.compile(
    r"\b(OPEN|MERGED|DRAFT)\b|已合并|未合并|待合并|已合入|还没合|review 中|待 review"
)
HASH_DIGIT = re.compile(r"#\s*\d")
AI_VOICE = re.compile(r"AI\s*语音合成|AI\s*配音")


def xhs_len(s):
    """小红书 title length: ASCII counts half."""
    return sum(0.5 if ord(c) < 128 else 1 for c in s)


def text_problems(text, *, xhs):
    out = []
    m = STATUS.search(text)
    if m:
        out.append(f"PR status in the text: {m.group(0)!r}")
    if xhs:
        m = HASH_DIGIT.search(text)
        if m:
            out.append(f"「#」 before a digit opens the topic popup: {m.group(0)!r}")
    return out


def probe(path, entries):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries"]
        + [entries, "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip().split(",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video", required=True, help="the file you will pick in the dialog"
    )
    ap.add_argument("--render", help="the render you mean to publish")
    ap.add_argument("--cover", help="the 4:3 cover for both sites")
    ap.add_argument("--bili-title")
    ap.add_argument("--xhs-title")
    ap.add_argument("--bili-desc", help="file with the B站 简介")
    ap.add_argument("--xhs-body", help="file with the 小红书 body")
    a = ap.parse_args()
    bad = []

    def check(ok, msg):
        print(f"  {'✓' if ok else '✗'} {msg}")
        if not ok:
            bad.append(msg)

    dur = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration"]
            + ["-of", "csv=p=0", a.video],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    print(f"  · video {a.video}: {dur:.1f} s ({int(dur // 60)}:{int(dur % 60):02d})")
    if a.render:
        check(
            filecmp.cmp(a.video, a.render, shallow=False), f"same bytes as {a.render}"
        )
    if a.cover:
        w, h = probe(a.cover, "stream=width,height")
        check((w, h) == ("1440", "1080"), f"cover {w}×{h} (want 1440×1080)")
    if a.bili_title is not None:
        check(len(a.bili_title) <= 80, f"B站 title {len(a.bili_title)}/80")
        for p in text_problems(a.bili_title, xhs=False):
            check(False, f"B站 title: {p}")
    if a.xhs_title is not None:
        n = xhs_len(a.xhs_title)
        check(n <= 20, f"小红书 title {n:g}/20")
    if a.bili_desc:
        d = open(a.bili_desc, encoding="utf-8").read().strip()
        check(0 < len(d) <= 2000, f"B站 简介 {len(d)}/2000")
        check(bool(AI_VOICE.search(d)), "B站 简介 has the AI-voice line")
        for p in text_problems(d, xhs=False):
            check(False, f"B站 简介: {p}")
    if a.xhs_body:
        b = open(a.xhs_body, encoding="utf-8").read().strip()
        check(0 < len(b) <= 1000, f"小红书 body {len(b)}/1000")
        for p in text_problems(b, xhs=True):
            check(False, f"小红书 body: {p}")
    print(f"{'✓' if not bad else '✗'} publish_check: {len(bad)} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
