"""Which reading of a Chinese polyphone does the voice use, and how often?

Whisper can't tell 行 háng from 行 xíng (it writes 行 either way), so this listens acoustically instead.
Each take of the line is compared (DTW over log-mel frames) with the same line where the polyphone is
swapped for unambiguous homophones of each reading, e.g. 行 → 航/杭 (háng) vs 型/形 (xíng). The take
sits next to the variant whose reading it used. Edge picks the reading per request, so one take proves
little: the tool synthesizes several and reports the misread rate, for the text as written and for the
spoken form after pronunciation.json.

    $SEV_PY -I $SEV_SKILL/scripts/poly_probe.py --text "就是这一行在表里的位置。" --at 这一行 \\
        --char 行 --want 航,杭 --avoid 型,形
    #  as written: wrong 3/6   as spoken (这一航): wrong 0/6

Needs numpy (installed with mlx-whisper) and ffmpeg.
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_tts_engine as E  # noqa: E402
import pronounce  # noqa: E402

SR = 16000


def _mel_fb(n_fft=512, n_mels=40):
    f = np.linspace(0, SR / 2, n_fft // 2 + 1)
    pts = 700 * (
        10 ** (np.linspace(0, 2595 * np.log10(1 + SR / 2 / 700), n_mels + 2) / 2595) - 1
    )
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        lo, c, hi = pts[i : i + 3]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return fb


FB = _mel_fb()


def logmel(path):
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    x = np.frombuffer(pcm, np.float32)
    nz = np.nonzero(np.abs(x) > 1e-3)[0]
    x = x[nz[0] : nz[-1] + 1] if len(nz) else x
    win = np.hanning(400)
    frames = np.stack([x[i : i + 400] * win for i in range(0, len(x) - 400, 160)])
    lm = np.log((np.abs(np.fft.rfft(frames, 512)) ** 2) @ FB.T + 1e-8)
    return lm - lm.mean(0)


def dtw(a, b):
    """Length-normalized DTW cost; each row is a vectorized min-plus scan."""
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    prev = np.concatenate([[0.0], np.full(d.shape[1], np.inf)])
    for i in range(d.shape[0]):
        t = d[i] + np.minimum(prev[1:], prev[:-1])
        c = np.cumsum(d[i])
        prev = np.concatenate([[np.inf], c + np.minimum.accumulate(t - c)])
    return prev[-1] / sum(d.shape)


async def _synth_all(texts, voice, td):
    sem = asyncio.Semaphore(6)

    async def one(k, t):
        async with sem:
            p = os.path.join(td, f"{k}.mp3")
            for attempt in range(3):
                try:
                    await E.synth(t, voice, "+0%", p)
                    return p
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)

    return await asyncio.gather(*[one(k, t) for k, t in enumerate(texts)])


def locate(text, at, char, ref=None):
    """Index of `char` inside the context `at`. In the spoken form the char may have been respelled
    (这一行 → 这一航): take the context match nearest the written position `ref` (0..1)."""
    k = at.index(char)
    if text.count(at) == 1:
        return text.index(at) + k
    if ref is None:
        sys.exit(
            f"✗ context {at!r} must occur exactly once in {text!r}: give a longer --at"
        )
    rx = "(?=" + re.escape(at[:k]) + "([\u4e00-\u9fff])" + re.escape(at[k + 1 :]) + ")"
    cands = [m.start(1) for m in re.finditer(rx, text)]
    if not cands:
        sys.exit(f"✗ context {at!r} not found in {text!r}")
    return min(cands, key=lambda c: abs(c / len(text) - ref))


def probe(text, at, char, want, avoid, takes, voice, td, ref=None):
    i = locate(text, at, char, ref)
    swap = lambda h: text[:i] + h + text[i + 1 :]  # noqa: E731
    texts = [text] * takes + [swap(h) for h in want + avoid]
    paths = asyncio.run(_synth_all(texts, voice, td))
    refs = [logmel(p) for p in paths[takes:]]
    wrong, rows = 0, []
    for p in paths[:takes]:
        a = logmel(p)
        dw = np.mean([dtw(a, r) for r in refs[: len(want)]])
        da = np.mean([dtw(a, r) for r in refs[len(want) :]])
        wrong += da < dw
        rows.append(
            f"{'WRONG' if da < dw else 'ok':5} d(want)={dw:.2f} d(avoid)={da:.2f}"
        )
    return wrong, rows, text[max(0, i - 2) : i + 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="the narration line")
    ap.add_argument(
        "--at",
        required=True,
        help="a short context holding the polyphone once, e.g. 这一行",
    )
    ap.add_argument("--char", required=True, help="the polyphone, e.g. 行")
    ap.add_argument(
        "--want", required=True, help="homophones of the right reading, e.g. 航,杭"
    )
    ap.add_argument(
        "--avoid", required=True, help="homophones of the wrong reading, e.g. 型,形"
    )
    ap.add_argument("--takes", type=int, default=6)
    ap.add_argument("--voice", default=E.DEFAULT_VOICE)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    want, avoid = a.want.split(","), a.avoid.split(",")
    spoken = pronounce.to_spoken(a.text)[0]
    forms = [("as written", a.text)] + (
        [("as spoken", spoken)] if spoken != a.text else []
    )
    ref = locate(a.text, a.at, a.char) / len(a.text)
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for k, (name, t) in enumerate(forms):
            sub = os.path.join(td, str(k))
            os.makedirs(sub)
            wrong, rows, ctx = probe(
                t, a.at, a.char, want, avoid, a.takes, a.voice, sub, ref
            )
            print(f"  {name} ({ctx}): wrong {wrong}/{a.takes}")
            if a.verbose:
                print("\n".join("    " + r for r in rows))
            bad = wrong  # the last form is what the video will say
    print(
        f"{'✓' if not bad else '✗'} poly_probe: {bad}/{a.takes} takes use the avoided reading"
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
