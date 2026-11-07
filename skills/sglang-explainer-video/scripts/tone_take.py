"""Listen to the recorded take for tone-only polyphones: did 切 come out qiē or qiè?

poly_probe.py and hang_take.py compare log-mel frames, which carry almost no pitch, so they cannot hear a
reading that differs only in tone (切 qiē / qiè). This measures pitch. For every character listed under
`tones` in pronunciation.json it:

  1. synthesizes the same line with the character swapped for single-reading control characters of each
     candidate tone (切 → 接/揭/歇/包/通 for tone 1, 借/谢/窃/卸/报 for tone 4);
  2. force-aligns the take and the controls with the Montreal Forced Aligner (mandarin_mfa), so the
     character's vowel and coda are found exactly. Edge's word boundaries are estimates, and a window placed
     with them often lands on the neighbour (the 不 in 「也不切它」 is itself falling);
  3. measures the pitch slope over that vowel + coda with Praat (parselmouth), in semitones, and puts the
     take next to the tone whose controls it is closer to.

Comparing against controls of the same line cancels sentence position and declination, which a fixed
threshold cannot. Calibrated 2026-11-07 with --calibrate on 12 sentences of the DCP explainer video: 116/120
controls judged right, leave-one-out (a fixed threshold managed 84 %; 些 made a poor tone-1 control and was
dropped). The misses were tone-1 controls judged tone 4 with margins of 0.2 to 3.1 st, so a WRONG under 1 st
is marked weak: listen before rewording. MFA alone, choosing between tone variants of the word, scored
58/108: its GMM features carry no pitch even though its phones are tone-marked. A text Edge reads one way it
reads the same way every time, so a WRONG line needs a reword, not a retake. Needs the tone env:
bash setup_tone.sh.

    $SEV_PY -I $SEV_SKILL/scripts/tone_take.py --script $P/SCRIPT.md --meta $P/audio_meta.json [--lines 5,41]
    #  L5   切  want qiē  heard qiè  WRONG  margin +7.6 st  …DCP 也不切它。…
    #  ✗ tone_take: 1 of 8 tone-only polyphones took the wrong tone
    $SEV_PY -I $SEV_SKILL/scripts/tone_take.py --calibrate 切 "按 token 切到各张卡上。" "把 KV 切成 8 份。" ...
    #  leave-one-out on the controls themselves: how well this character's tones separate
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TONE_MARK = re.compile(r"[˥˧˨˩˦]+$")
TOKEN = re.compile(r"[一-鿿]|[A-Za-z0-9][A-Za-z0-9.+_\-]*")
HAN = re.compile(r"[一-鿿]")
MFA_ACOUSTIC = "mandarin_mfa.zip"
MFA_DICT = "mandarin_china_mfa.dict"


def workspace():
    return os.environ.get("SEV_WS") or os.path.expanduser(
        "~/.cache/sglang-explainer-video"
    )


def mfa_env():
    return os.environ.get("SEV_MFA_ENV") or os.path.join(workspace(), "mfa-env")


def mfa_models():
    return os.environ.get("SEV_MFA_MODELS") or os.path.join(workspace(), "mfa-models")


def tone_table():
    data = json.load(open(os.path.join(HERE, "pronunciation.json"), encoding="utf-8"))
    return {k: v for k, v in data.get("tones", {}).items() if not k.startswith("_")}


def intended(text, i, spec):
    """The tone the narration means for text[i]: a listed word around it, else the default."""
    for tone, words in spec.get("words", {}).items():
        for w in words:
            for k, c in enumerate(w):
                if c == text[i] and text[i - k : i - k + len(w)] == w:
                    return int(tone)
    return int(spec["default"])


def verdict(take, groups, min_sep=1.0):
    """(heard tone or None, margin in st). `groups` maps tone -> control slopes of the same line."""
    centre = {t: sorted(v)[len(v) // 2] for t, v in groups.items() if v}
    if len(centre) < 2:
        return None, 0.0
    order = sorted(centre, key=lambda t: abs(take - centre[t]))
    best, other = order[0], order[1]
    if abs(centre[best] - centre[other]) < min_sep:
        return None, 0.0
    return best, abs(take - centre[other]) - abs(take - centre[best])


def lab_and_words(text, j, target, mfa_dict):
    """MFA transcript of `text` with the character at j renamed `target`, plus the dictionary entries it needs."""
    toks, words = [], {}
    for m in TOKEN.finditer(text):
        t = m.group(0)
        if m.start() == j:
            words[target] = mfa_dict.get(t, ["spn"])
            toks.append(target)
            continue
        key = t.lower()
        if key not in words:
            words[key] = mfa_dict.get(t, ["spn"]) if HAN.match(t) else ["spn"]
        toks.append(key)
    return " ".join(toks), words


def load_mfa_dict(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and len(p[0]) == 1:
            d.setdefault(p[0], []).append(p[-1])
    return d


def to_wav(src, dst):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
        check=True,
    )


async def synth_all(texts, td):
    sys.path.insert(0, HERE)
    import edge_tts_engine as E  # noqa: E402

    sem = asyncio.Semaphore(6)

    async def one(k, text):
        async with sem:
            mp3 = os.path.join(td, f"ctl{k}.mp3")
            for attempt in range(3):
                try:
                    await E.synth(text, E.DEFAULT_VOICE, "+0%", mp3)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)
            return mp3

    return await asyncio.gather(*[one(k, t) for k, t in enumerate(texts)])


def align(utts, td):
    """utts: [(uid, wav, text, j, target_char_for_dict)] -> {uid: (start, end) of the target's vowel + coda}."""
    env_bin = os.path.join(mfa_env(), "bin")
    models = mfa_models()
    for need in (
        os.path.join(env_bin, "mfa"),
        os.path.join(models, MFA_ACOUSTIC),
        os.path.join(models, MFA_DICT),
    ):
        if not os.path.exists(need):
            sys.exit(
                f"✗ tone_take needs the tone env ({need} missing): bash {HERE}/setup_tone.sh"
            )
    base = load_mfa_dict(os.path.join(models, MFA_DICT))
    corpus = os.path.join(td, "corpus", "spk")
    os.makedirs(corpus)
    words = {}
    for uid, wav, text, j, _ in utts:
        shutil.copy(wav, os.path.join(corpus, f"{uid}.wav"))
        lab, w = lab_and_words(text, j, f"zzt{uid}", base)
        words.update(w)
        open(os.path.join(corpus, f"{uid}.lab"), "w", encoding="utf-8").write(
            lab + "\n"
        )
    dict_path = os.path.join(td, "corpus.dict")
    with open(dict_path, "w", encoding="utf-8") as f:
        for w, prons in words.items():
            for pr in dict.fromkeys(prons):
                f.write(f"{w}\t{pr}\n")
    out = os.path.join(td, "aligned")
    env = dict(
        os.environ,
        PATH=env_bin + os.pathsep + os.environ.get("PATH", ""),
        MFA_ROOT_DIR=os.path.join(workspace(), "mfa-root"),
    )
    r = subprocess.run(
        [
            os.path.join(env_bin, "mfa"),
            "align",
            os.path.dirname(corpus),
            dict_path,
            os.path.join(models, MFA_ACOUSTIC),
            out,
            "--clean",
            "--no_tokenization",
            "--single_speaker",
            "--output_format",
            "json",
            "--beam",
            "100",
            "--retry_beam",
            "400",
            "--num_jobs",
            "4",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode:
        sys.exit("✗ mfa align failed:\n" + (r.stdout + r.stderr)[-2000:])
    spans = {}
    for uid, *_ in utts:
        p = os.path.join(out, "spk", f"{uid}.json")
        if not os.path.exists(p):
            continue
        tiers = json.load(open(p))["tiers"]
        wt = next(v for k, v in tiers.items() if "word" in k)["entries"]
        pt = next(v for k, v in tiers.items() if "phone" in k)["entries"]
        w = next((e for e in wt if e[2] == f"zzt{uid}"), None)
        if not w:
            continue
        vowels = [
            e
            for e in pt
            if e[0] >= w[0] - 1e-3 and e[1] <= w[1] + 1e-3 and TONE_MARK.search(e[2])
        ]
        spans[uid] = (vowels[0][0] if vowels else w[0], w[1])
    return spans


PITCH = r"""
import json, sys
import numpy as np
import parselmouth
out = {}
for uid, wav, a, b in json.load(open(sys.argv[1])):
    p = parselmouth.Sound(wav).to_pitch_ac(time_step=0.005, pitch_floor=float(sys.argv[2]), pitch_ceiling=float(sys.argv[3]))
    f, t = p.selected_array["frequency"], p.xs()
    m = (t >= a) & (t <= b) & (f > 0)
    if m.sum() < 5 or not (f > 0).any():
        continue
    st = 12 * np.log2(f[m] / np.median(f[f > 0]))
    q = max(2, len(st) // 4)
    out[uid] = float(np.median(st[-q:]) - np.median(st[:q]))
json.dump(out, sys.stdout)
"""


def slopes(spans, wavs, td, floor, ceiling):
    jobs = [(uid, wavs[uid], a, b) for uid, (a, b) in spans.items()]
    jf = os.path.join(td, "pitch_jobs.json")
    json.dump(jobs, open(jf, "w"))
    r = subprocess.run(
        [
            os.path.join(mfa_env(), "bin", "python"),
            "-c",
            PITCH,
            jf,
            str(floor),
            str(ceiling),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode:
        sys.exit(
            "✗ pitch step failed (praat-parselmouth in the tone env?):\n"
            + r.stderr[-1500:]
        )
    return json.loads(r.stdout)


def nth(text, ch, k):
    return [m.start() for m in re.finditer(re.escape(ch), text)][k]


def build(cases, table, td):
    """cases: [(key, spoken text, j, char, take wav or None)] -> utterances + per-case control uids by tone."""
    texts, meta = [], []
    for key, spoken, j, ch, _ in cases:
        for tone, chars in table[ch]["controls"].items():
            for c in chars:
                texts.append(spoken[:j] + c + spoken[j + 1 :])
                meta.append((key, int(tone), c, j))
    mp3s = asyncio.run(synth_all(texts, td))
    utts, wavs, groups = [], {}, {}
    for k, (mp3, (key, tone, c, j)) in enumerate(zip(mp3s, meta)):
        uid = f"c{k}"
        wavs[uid] = mp3[:-4] + ".wav"
        to_wav(mp3, wavs[uid])
        utts.append((uid, wavs[uid], texts[k], j, c))
        groups.setdefault(key, []).append((uid, tone, c))
    for n, (key, spoken, j, ch, take) in enumerate(cases):
        if take:
            uid = f"t{n}"
            wavs[uid] = os.path.join(td, f"take{n}.wav")
            to_wav(take, wavs[uid])
            utts.append((uid, wavs[uid], spoken, j, ch))
    return utts, wavs, groups


def check(a, table):
    sys.path.insert(0, HERE)
    import pronounce  # noqa: E402
    from pron_check import script_lines  # noqa: E402

    only = {int(x) for x in a.lines.split(",")} if a.lines else None
    chars = set(a.chars.split(",")) if a.chars else set(table)
    text = {int(k): t for k, t in script_lines(a.script) if k.isdigit()}
    root = os.path.dirname(os.path.abspath(a.meta))
    cases, info = [], {}
    for v in json.load(open(a.meta, encoding="utf-8"))["voices"]:
        n = v["frame"]
        if (only and n not in only) or n not in text:
            continue
        vo = text[n]
        spoken = pronounce.to_spoken(vo)[0]
        for ch in chars & set(vo):
            pos = [m.start() for m in re.finditer(re.escape(ch), vo)]
            if len(pos) != spoken.count(ch):
                print(f"  L{n:<3} {ch}  skipped: the spoken form rewrites it")
                continue
            for k, i in enumerate(pos):
                key = f"L{n}_{ch}_{k}"
                take = (
                    v["path"]
                    if os.path.isabs(v["path"])
                    else os.path.join(root, v["path"])
                )
                cases.append((key, spoken, nth(spoken, ch, k), ch, take))
                info[key] = (n, vo, i, ch, intended(vo, i, table[ch]))
    if not cases:
        print(f"✓ tone_take: no {'/'.join(sorted(chars))} in these lines")
        return
    td = tempfile.mkdtemp(prefix="tone-take-")
    utts, wavs, groups = build(cases, table, td)
    sl = slopes(align(utts, td), wavs, td, a.floor, a.ceiling)
    bad = 0
    for n_case, (key, *_rest) in enumerate(cases):
        n, vo, i, ch, want = info[key]
        rd = table[ch]["readings"]
        take = sl.get(f"t{n_case}")
        g = {}
        for uid, tone, _ in groups[key]:
            if uid in sl:
                g.setdefault(tone, []).append(sl[uid])
        heard, margin = verdict(take, g) if take is not None else (None, 0.0)
        ctx = vo[max(0, i - 6) : i + 5]
        if heard is None:
            print(
                f"  L{n:<3} {ch}  want {rd[str(want)]}  unsure (controls do not separate here)  …{ctx}…"
            )
            continue
        wrong = heard != want
        bad += wrong
        if wrong or margin < 0.5 or a.verbose:
            tag = (
                ("WRONG" if margin >= 1.0 else "WRONG (weak, listen)")
                if wrong
                else ("close" if margin < 0.5 else "ok")
            )
            print(
                f"  L{n:<3} {ch}  want {rd[str(want)]}  heard {rd[str(heard)]}  {tag}  margin {margin:+.1f} st  …{ctx}…"
            )
            if wrong and table[ch].get("reword"):
                print(f"        reword: {table[ch]['reword']}")
    print(
        f"{'✓' if not bad else '✗'} tone_take: {bad} of {len(cases)} tone-only polyphones took the wrong tone"
    )
    if not a.keep:
        shutil.rmtree(td, ignore_errors=True)
    sys.exit(1 if bad else 0)


def calibrate(a, table):
    """Leave-one-out on the controls: each control is judged against the other controls of its sentence."""
    ch = a.calibrate
    sys.path.insert(0, HERE)
    import pronounce  # noqa: E402

    cases = []
    for k, s in enumerate(a.sentences):
        spoken = pronounce.to_spoken(s)[0]
        if ch not in spoken:
            sys.exit(f"✗ {ch} not in: {s}")
        cases.append((f"s{k}", spoken, spoken.index(ch), ch, None))
    td = tempfile.mkdtemp(prefix="tone-cal-")
    utts, wavs, groups = build(cases, table, td)
    sl = slopes(align(utts, td), wavs, td, a.floor, a.ceiling)
    ok = n = 0
    margins = {True: [], False: []}
    for key, spoken, *_ in cases:
        rows = [(uid, tone, c) for uid, tone, c in groups[key] if uid in sl]
        for uid, tone, c in rows:
            g = {}
            for u2, t2, _ in rows:
                if u2 != uid:
                    g.setdefault(t2, []).append(sl[u2])
            heard, margin = verdict(sl[uid], g)
            n += 1
            ok += heard == tone
            margins[heard == tone].append(margin)
            if heard != tone:
                print(
                    f"  {c} (tone {tone}) judged {heard}  margin {margin:+.1f} st  {spoken}"
                )
    print(
        f"{ch}: {ok}/{n} controls judged right ({ok / max(n, 1) * 100:.0f}%), leave-one-out over {len(cases)} sentences"
    )
    for right, m in margins.items():
        if m:
            m = sorted(m)
            print(
                f"  margins when {'right' if right else 'wrong'}: min {m[0]:.1f}  median {m[len(m) // 2]:.1f}  max {m[-1]:.1f} st"
            )
    if not a.keep:
        shutil.rmtree(td, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", help="SCRIPT.md (the written narration)")
    ap.add_argument("--meta", help="audio_meta.json of the recorded take")
    ap.add_argument("--lines", help="only these line numbers, e.g. 5,41")
    ap.add_argument("--chars", help="only these characters, e.g. 切")
    ap.add_argument(
        "--calibrate",
        metavar="CHAR",
        help="judge the controls of CHAR in the given sentences",
    )
    ap.add_argument("sentences", nargs="*")
    ap.add_argument(
        "--floor", type=float, default=100, help="pitch floor in Hz (100 for Xiaoxiao)"
    )
    ap.add_argument("--ceiling", type=float, default=500, help="pitch ceiling in Hz")
    ap.add_argument(
        "--keep", action="store_true", help="keep the temp corpus and alignments"
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="print every occurrence"
    )
    a = ap.parse_args()
    table = tone_table()
    if a.calibrate:
        calibrate(a, table)
    elif a.script and a.meta:
        check(a, table)
    else:
        ap.error("give --script and --meta, or --calibrate CHAR sentence ...")


if __name__ == "__main__":
    main()
