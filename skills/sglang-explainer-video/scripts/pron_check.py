"""Check that the narration's English terms are read right and without breaks.

    # before TTS: every English term of the script, read in a carrier sentence and transcribed
    $SEV_PY -I $SEV_SKILL/scripts/pron_check.py --script $P/SCRIPT.md --asr
    # after TTS: pauses inside clauses + every line transcribed in context
    $SEV_PY -I $SEV_SKILL/scripts/pron_check.py --meta $P/audio_meta.json --hyperframes $P --asr
    # lexicon health: every respelling still passes, and is still needed (voice / Edge updates)
    $SEV_PY -I $SEV_SKILL/scripts/pron_check.py --audit

A term is OK when an unbiased Whisper pass writes it back (case/spaces ignored), OK~ when only a pass
primed with the term does (plausible; listen once), FAIL when neither does — then add a respelling to
pronunciation.json (the tool tries a few and prints the first that passes) or reword the line.
Verdicts are cached per voice + spoken form in $SEV_WS/pron_cache.json. Without --asr only the
rewrites and the pause check run. ASR needs mlx-whisper (Apple Silicon; setup_env.sh installs it).
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_tts_engine as E  # noqa: E402
import pronounce  # noqa: E402

WHISPER = os.environ.get("SEV_WHISPER", "mlx-community/whisper-large-v3-turbo")
CACHE = os.path.join(
    os.environ.get("SEV_WS", os.path.expanduser("~/.cache/sglang-explainer-video")),
    "pron_cache.json",
)
PUNCT = "，。、；：！？,.;:!?…—）)」』”’"
PAUSE_S = 0.2


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def script_lines(path):
    """Narration lines of SCRIPT.md: the 4-space-indented block under each `## Line`."""
    out, cur = [], None
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"## Line (\S+)", ln)
        if m:
            cur = m.group(1)
        elif cur and ln.startswith("    ") and ln.strip():
            out.append((cur, ln.strip()))
    return out


def terms(text):
    """English tokens worth checking: anything with a letter, in written form."""
    return [
        m.group(0)
        for m in pronounce.TOKEN.finditer(text)
        if re.search(r"[A-Za-z]", m.group(0))
    ]


class Asr:
    def __init__(self):
        try:
            import mlx_whisper
        except ImportError:
            sys.exit(
                "✗ --asr needs mlx-whisper in this venv (Apple Silicon): re-run setup_env.sh"
            )
        self.w = mlx_whisper

    def __call__(self, audio, prompt=None):
        kw = {"initial_prompt": prompt} if prompt else {}
        return self.w.transcribe(audio, path_or_hf_repo=WHISPER, language="zh", **kw)[
            "text"
        ].strip()


def heard(asr, written, audio, context):
    """('OK'|'OK~'|'FAIL', transcript) for `written` inside `audio`."""
    t = asr(audio)
    if norm(written) in norm(t):
        return "OK", t
    tb = asr(audio, prompt=f"术语：{written}。{context}")
    return ("OK~" if norm(written) in norm(tb) else "FAIL"), t


def carrier(asr, voice, written, spoken):
    with tempfile.TemporaryDirectory() as td:
        mp3 = os.path.join(td, "t.mp3")
        asyncio.run(E.synth(f"接下来看 {spoken} 这一部分。", voice, "+0%", mp3))
        v, t = heard(asr, written, mp3, f"读作 {spoken}。" if spoken != written else "")
    return v, t.removeprefix("接下来看").removesuffix("。").removesuffix("这一部分")


def respellings(t):
    """Candidate spoken forms to try when the written form fails."""
    c = [
        re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t),
        t.replace("-", " "),
        t.lower(),
        t.capitalize(),
    ]
    if t.isupper() and 2 <= len(t) <= 5:
        c.append(" ".join(t))
    return [x for i, x in enumerate(c) if x != t and x not in c[:i]]


def check_terms(lines, voice, asr):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    vc = cache.setdefault(voice, {})
    seen, bad = set(), 0
    for _, text in lines:
        for w in terms(text):
            if w in seen:
                continue
            seen.add(w)
            sp = pronounce.spoken_token(w)
            if w in pronounce.data().get("approved", []):
                print(f"  OK   {w:<44} (approved by ear)")
                continue
            if sp not in vc:
                v, t = carrier(asr, voice, w, sp)
                vc[sp] = {"verdict": v, "heard": t}
                json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)
            v, t = vc[sp]["verdict"], vc[sp]["heard"]
            shown = w if sp == w else f"{w} → {sp}"
            line = f"  {v:<4} {shown:<44} heard: {t}"
            if v == "FAIL":
                bad += 1
                for cand in respellings(sp):
                    cv, ct = carrier(asr, voice, w, cand)
                    if cv == "OK":
                        line += f'\n       try lexicon "{w}": "{cand}"  (heard: {ct})'
                        break
                else:
                    line += "\n       no respelling passed — listen, or reword the line"
            print(line, flush=True)
    return bad


def check_meta(meta_path, hf, asr):
    voices = json.load(open(meta_path))["voices"]
    issues = 0
    for v in voices:
        ws, lid = v["words"], v.get(
            "frame", v.get("id")
        )  # audio_meta: frame; engine meta: id
        for a, b in zip(ws, ws[1:]):
            gap = b["start"] - a["end"]
            if gap > PAUSE_S and a["text"][-1] not in PUNCT:
                issues += 1
                print(
                    f"  PAUSE line {lid}: {gap:.2f}s between “{a['text']}” and “{b['text']}”"
                )
        if asr:
            text = " ".join(
                w["text"] for w in ws
            )  # spaces keep adjacent terms apart (fused top-k)
            wav = os.path.join(hf, v["path"])
            t, tb, out = asr(wav), None, []
            for w in dict.fromkeys(terms(text)):
                sp = pronounce.spoken_token(w)
                # denoise → 去噪: the transcript holds the Chinese the voice said
                said_han = re.search(r"[一-鿿]", sp) and sp.replace(
                    " ", ""
                ) in t.replace(" ", "")
                if (
                    w not in pronounce.data().get("approved", [])
                    and norm(w) not in norm(t)
                    and not said_han
                ):
                    tb = tb or asr(
                        wav, prompt=pronounce.to_spoken(text.replace(" ", ""))[0]
                    )
                    verdict = "OK~" if norm(w) in norm(tb) else "FAIL"
                    issues += verdict == "FAIL"
                    out.append(f"{verdict} {w}")
            if out:
                print(f"  line {lid}: {' · '.join(out)}\n      heard: {t}")
    return issues


def audit(voice, asr):
    """Re-measure every lexicon entry: the spoken form must pass, and the written form should still fail
    (if it now passes, the voice improved and the entry can go)."""
    bad = 0
    for w, sp in pronounce.data()["lexicon"].items():
        v_sp, t_sp = carrier(asr, voice, w, sp)
        v_w, _ = carrier(asr, voice, w, w)
        if v_sp == "FAIL":
            bad += 1
            note = "respelling no longer passes: re-probe this term"
        elif v_w == "OK":
            note = "written form passes in the carrier now: check it in a real sentence before dropping the entry"
        else:
            note = ""
        print(
            f"  {v_sp:<4} {w} → {sp:<24} heard: {t_sp:<28} (as written: {v_w}) {note}",
            flush=True,
        )
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", help="SCRIPT.md (before TTS)")
    ap.add_argument(
        "--text", action="append", default=[], help="ad-hoc narration line(s)"
    )
    ap.add_argument("--meta", help="audio_meta.json (after TTS)")
    ap.add_argument("--hyperframes", default=".")
    ap.add_argument("--voice", default=E.DEFAULT_VOICE)
    ap.add_argument("--asr", action="store_true")
    ap.add_argument(
        "--audit",
        action="store_true",
        help="re-measure every lexicon entry (implies --asr)",
    )
    a = ap.parse_args()
    asr = Asr() if (a.asr or a.audit) else None

    lines = (script_lines(a.script) if a.script else []) + [("text", t) for t in a.text]
    bad = 0
    if lines:
        print("· rewrites (voice reads the right side, captions keep the left):")
        for lid, text in lines:
            ch = pronounce.changes(text)
            if ch:
                print(
                    f"  line {lid}: " + " · ".join(f"{w} → {s.strip()}" for w, s in ch)
                )
        if asr:
            print(f"· terms ({a.voice}, carrier sentence):")
            bad += check_terms(lines, a.voice, asr)
    if a.audit:
        print(f"· lexicon audit ({a.voice}):")
        bad += audit(a.voice, asr)
    if a.meta:
        print(
            "· recorded lines (pauses inside clauses"
            + (", terms in context):" if asr else "):")
        )
        bad += check_meta(a.meta, a.hyperframes, asr)
    print(f"{'✓' if not bad else '✗'} pron_check: {bad} issue(s)")


if __name__ == "__main__":
    main()
