"""Listen to the recorded take: does every 行 take the reading the narration means (háng vs xíng)?

poly_probe.py measures a sentence's risk on fresh takes. This checks the take the video actually uses. For
each 行 in the narration it estimates the character's time from the take's word boundaries, cuts a 0.6 s
window around it, and compares that window (DTW over log-mel, as poly_probe does) with fresh references of
the same line where that one character is swapped for unambiguous homophones of each reading (航/杭 for
háng, 型/形 for xíng). The window sits next to the reading it used.

行 is meant xíng after 执/运/并/串/进/可/流/发/就 and before 为/动; every other 行 is meant háng (row, line).

    $SEV_PY -I $SEV_SKILL/scripts/hang_take.py --script $P/SCRIPT.md --meta $P/audio_meta.json [--lines 24,35]
    #  L24  háng  WRONG  margin -4.99  …128 字节行，超过…
    #  ✗ hang_take: 1 of 32 行 take the wrong reading

Calibrated 2026-11-05 on 32 control takes with a known reading (16 航, 16 型): 32/32 right at ±0.25,
±0.35 and ±0.5 s windows. Two pitfalls it fixes: Edge's WordBoundary words carry no punctuation, so
character indices are counted without punctuation or spaces; and Edge groups characters into words
differently per take, so the window is centred on the character, not on its word. A margin under about
1.5 is worth one more run. Needs numpy and ffmpeg.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_tts_engine as E  # noqa: E402
import poly_probe as PP  # noqa: E402
import pronounce  # noqa: E402
from pron_check import script_lines  # noqa: E402

HANG, XING = ["航", "杭"], ["型", "形"]
XING_BEFORE = set("执运并串进可流发就")
XING_AFTER = set("为动")
HALF = 0.3
SR = 16000


def strip(s):
    """What WordBoundary words carry: letters, digits and Han (no spaces, no punctuation)."""
    return re.sub(r"[^0-9A-Za-z一-鿿]", "", s)


def char_time(words, idx):
    pos = 0
    for w in words:
        t = strip(w["text"])
        if pos <= idx < pos + len(t):
            return w["start"] + (idx - pos + 0.5) / len(t) * (w["end"] - w["start"])
        pos += len(t)
    return None


def cut(path, t, out):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, t - HALF):.3f}",
            "-to",
            f"{t + HALF:.3f}",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(SR),
            out,
        ],
        check=True,
    )
    return PP.logmel(out)


async def synth_all(texts, td):
    sem = asyncio.Semaphore(6)

    async def one(k, text):
        async with sem:
            p = os.path.join(td, f"r{k}.mp3")
            for attempt in range(3):
                try:
                    return p, await E.synth(text, E.DEFAULT_VOICE, "+0%", p)
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)

    return await asyncio.gather(*[one(k, t) for k, t in enumerate(texts)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="SCRIPT.md (the written narration)")
    ap.add_argument(
        "--meta", required=True, help="audio_meta.json of the recorded take"
    )
    ap.add_argument("--lines", help="only these line numbers, e.g. 24,35")
    a = ap.parse_args()
    only = {int(x) for x in a.lines.split(",")} if a.lines else None
    text = {int(k): t for k, t in script_lines(a.script) if k.isdigit()}
    root = os.path.dirname(os.path.abspath(a.meta))
    td = tempfile.mkdtemp(prefix="hang-take-")
    jobs, plan = [], []
    for v in json.load(open(a.meta, encoding="utf-8"))["voices"]:
        n = v["frame"]
        if (only and n not in only) or n not in text or "行" not in text[n]:
            continue
        vo = text[n]
        spoken = pronounce.to_spoken(vo)[0]
        sp_pos = [m.start() for m in re.finditer("[行航]", spoken)]
        for k, m in enumerate(re.finditer("行", vo)):
            i = m.start()
            hang = not (vo[i - 1 : i] in XING_BEFORE or vo[i + 1 : i + 2] in XING_AFTER)
            want, avoid = (HANG, XING) if hang else (XING, HANG)
            j = sp_pos[k]
            refs = []
            for h in want + avoid:
                jobs.append(spoken[:j] + h + spoken[j + 1 :])
                refs.append((len(jobs) - 1, h in want, len(strip(spoken[:j]))))
            plan.append((n, vo, i, hang, v, refs))
    if not plan:
        print("✓ hang_take: no 行 in these lines")
        return
    res = asyncio.run(synth_all(jobs, td))
    bad = 0
    for n, vo, i, hang, v, refs in plan:
        take = v["path"] if os.path.isabs(v["path"]) else os.path.join(root, v["path"])
        t = char_time(v["words"], len(strip(vo[:i])))
        x = cut(take, t, os.path.join(td, "take.wav"))
        d = {True: [], False: []}
        for jj, is_want, ridx in refs:
            path, words = res[jj]
            rt = char_time(words, ridx)
            if rt is not None:
                d[is_want].append(
                    PP.dtw(x, cut(path, rt, os.path.join(td, f"ref{jj}.wav")))
                )
        margin = np.mean(d[False]) - np.mean(d[True])
        wrong = margin < 0
        bad += wrong
        if wrong or margin < 1.5:
            print(
                f"  L{n:<3} {'háng' if hang else 'xíng'}  {'WRONG' if wrong else 'close'}  margin {margin:+.2f}  "
                f"…{vo[max(0, i - 6): i + 5]}…"
            )
    print(
        f"{'✓' if not bad else '✗'} hang_take: {bad} of {len(plan)} 行 take the wrong reading"
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
