"""Transcribe every recorded line and diff it against the narration, syllable by syllable (pinyin).

pron_check only follows English terms. This listens to the Chinese too: each line of the take goes through
Whisper, both sides become pinyin, and every place where they disagree is printed. What it caught in the
tiny GEMM video (PR 47683, 2026-11-03), all fixed by rewording:

  清空 L2 时是 0.88   heard 「L20 是」: 「L2 时」 is èr shí, the same sound as 二十
  batch 是 1 的时候   heard 「batch 11 的时候」 (shì yī vs shí yī); 「batch 为 1」 is 唯一 exactly

Whisper writes characters, so a polyphone read the wrong way usually comes back as the same character
(行, 长, 重): use poly_probe.py for those. Expect noise on homophones (旧/就, 维/位), on numbers and on a
letter read as Chinese (y → 外); read each hit, re-synthesize only the lines where the sound itself is
wrong or ambiguous.

    $SEV_PY -I $SEV_SKILL/scripts/line_asr.py --script $P/SCRIPT.md --meta $P/audio_meta.json [--lines 5,41]
"""

import argparse
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pron_check import WHISPER, script_lines  # noqa: E402

try:
    import mlx_whisper
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover
    sys.exit("✗ line_asr needs mlx-whisper and pypinyin: re-run setup_env.sh")

HAN = re.compile(r"[一-鿿]")
# Whisper writes numbers either way; a hit made only of numerals is not a misreading.
NUMERALS = set("零一二两三四五六七八九十百千万点")


def diff(written, heard):
    a = [c for c in written if HAN.match(c)]
    b = [c for c in heard if HAN.match(c)]
    pa = lazy_pinyin("".join(a), style=Style.TONE3, neutral_tone_with_five=True)
    pb = lazy_pinyin("".join(b), style=Style.TONE3, neutral_tone_with_five=True)
    bare = lambda s: re.sub(r"\d", "", s)  # noqa: E731
    sm = difflib.SequenceMatcher(
        None, [bare(x) for x in pa], [bare(x) for x in pb], autojunk=False
    )
    hits = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if set(b[j1:j2]) <= NUMERALS and set(a[i1:i2]) <= NUMERALS:
            continue
        ctx = (
            "".join(a[max(0, i1 - 4) : i1])
            + "["
            + "".join(a[i1:i2])
            + "]"
            + "".join(a[i2 : i2 + 3])
        )
        hits.append(f"{op:7} {ctx} → [{''.join(b[j1:j2])}] {' '.join(pb[j1:j2])}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="SCRIPT.md (the written narration)")
    ap.add_argument(
        "--meta", required=True, help="audio_meta.json of the recorded take"
    )
    ap.add_argument("--lines", help="only these line numbers, e.g. 5,41")
    a = ap.parse_args()
    only = {int(x) for x in a.lines.split(",")} if a.lines else None
    text = {int(k): t for k, t in script_lines(a.script) if k.isdigit()}
    root = os.path.dirname(os.path.abspath(a.meta))
    flagged = 0
    for v in json.load(open(a.meta, encoding="utf-8"))["voices"]:
        n = v["frame"]
        if (only and n not in only) or n not in text:
            continue
        wav = v["path"] if os.path.isabs(v["path"]) else os.path.join(root, v["path"])
        heard = mlx_whisper.transcribe(wav, path_or_hf_repo=WHISPER, language="zh")[
            "text"
        ].strip()
        hits = diff(text[n], heard)
        if hits:
            flagged += 1
            print(f"  line {n}: heard {heard}")
            for h in hits:
                print(f"      · {h}")
    print(
        f"{'✓' if not flagged else '·'} line_asr: {flagged} line(s) to read (polyphones: poly_probe.py)"
    )


if __name__ == "__main__":
    main()
