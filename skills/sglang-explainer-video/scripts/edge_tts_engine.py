"""edge-tts drop-in for the HyperFrames media audio engine (``HF_MEDIA_ENGINE``).

Same CLI surface as ``media-use/audio/scripts/audio.mjs``::

    --request audio_request.json --hyperframes DIR --out audio_engine_meta.json --only tts,bgm|sfx

Writes the engine's neutral meta: ``voices[{id, path, duration_s, words[{id, text, start, end}]}]``.
Word timings come from edge-tts WordBoundary events (no Whisper pass); the punctuation that
follows each word in the script is re-attached so the caption grouper can split clauses.
BGM / SFX are not provided: ``bgm`` stays null and ``sfx`` stays empty.

Env:
    EDGE_TTS_VOICE      default zh-CN-YunyangNeural (overridden by the request's ``voice``)
    EDGE_TTS_RATE       default +0%
    EDGE_TTS_PAD_LAST   seconds of silence appended to the LAST line (closing hold), default 2.0
    EDGE_TTS_LEGACY_HOST=1  keep edge-tts' own synthesis host (skip the 403 workaround)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile

import edge_tts
import edge_tts.communicate as _comm
from edge_tts.constants import TRUSTED_CLIENT_TOKEN

if os.environ.get("EDGE_TTS_LEGACY_HOST") != "1":
    # rany2/edge-tts#491: since Oct 2026 speech.platform.bing.com answers the synthesis
    # WebSocket with 403; Edge 155 synthesizes through api.msedgeservices.com with the
    # same token. Voice listing still uses the legacy host.
    _comm.WSS_URL = (
        "wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1"
        f"?Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}"
    )

DEFAULT_VOICE = os.environ.get("EDGE_TTS_VOICE", "zh-CN-YunyangNeural")
PUNCT = set("，。、；：！？,.;:!?…—）)」』”’")


def _ascii(t):
    return all(ord(c) < 128 for c in t)


def merge_split_terms(text, words):
    """Re-join Latin/number terms edge-tts splits apart: FP32 → FP|32, FLUX.2 → FLUX.|2,
    torch.equal → torch.|equal, H100 → H|100. Two consecutive ASCII words merge when the
    script has no whitespace between them; CJK words are never merged (karaoke granularity).
    """
    out, pos, prev_end = [], 0, None
    for w in words:
        idx = text.find(w["text"], pos)
        if idx < 0:
            out.append(dict(w))
            prev_end = None
            continue
        end = idx + len(w["text"])
        gap = text[prev_end:idx] if (prev_end is not None and idx >= prev_end) else None
        if (
            out
            and gap is not None
            and set(gap) <= set("._-/")
            and _ascii(out[-1]["text"])
            and _ascii(w["text"])
        ):
            out[-1]["text"] += gap + w["text"]
            out[-1]["end"] = w["end"]
        else:
            out.append(dict(w))
        pos = prev_end = end
    return out


def attach_punct(text, words):
    """Append the punctuation that follows each word in ``text`` to that word."""
    out, pos = [], 0
    for w in words:
        idx = text.find(w["text"], pos)
        if idx < 0:
            out.append(w)
            continue
        end = idx + len(w["text"])
        tail = ""
        while end < len(text) and (text[end] in PUNCT or text[end] == " "):
            if text[end] != " ":
                tail += text[end]
            end += 1
        out.append({**w, "text": w["text"] + tail})
        pos = end
    return out


async def synth(text, voice, rate, mp3_path):
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    words = []
    with open(mp3_path, "wb") as f:
        async for ch in comm.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                start = ch["offset"] / 1e7
                words.append(
                    {
                        "text": ch["text"],
                        "start": round(start, 3),
                        "end": round(start + ch["duration"] / 1e7, 3),
                    }
                )
    return words


def duration(path):
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return round(float(r.stdout.strip()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--hyperframes", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="tts,bgm")
    args, _ = ap.parse_known_args()
    only = set(args.only.split(","))
    hf = os.path.abspath(args.hyperframes)

    with open(args.request) as f:
        request = json.load(f)
    prev = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            prev = json.load(f)
    voice = request.get("voice") or DEFAULT_VOICE
    rate = os.environ.get("EDGE_TTS_RATE", "+0%")
    pad_last = float(os.environ.get("EDGE_TTS_PAD_LAST", "2.0"))

    voices = prev.get("voices", [])
    if "tts" in only:
        voices = []
        lines = [
            ln for ln in request.get("lines", []) if str(ln.get("text", "")).strip()
        ]
        os.makedirs(os.path.join(hf, "assets", "voice"), exist_ok=True)
        print(
            f"· tts: edge-tts · voice {voice} · rate {rate} · {len(lines)} line(s)",
            file=sys.stderr,
        )
        for k, line in enumerate(lines):
            text = str(line["text"]).strip()
            lid = str(line["id"])
            rel = f"assets/voice/{lid}.wav"
            with tempfile.TemporaryDirectory() as td:
                mp3 = os.path.join(td, "line.mp3")
                words = asyncio.run(synth(text, voice, rate, mp3))
                af = (
                    ["-af", f"apad=pad_dur={pad_last}"]
                    if (k == len(lines) - 1 and pad_last > 0)
                    else []
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        mp3,
                        *af,
                        "-ar",
                        "44100",
                        "-ac",
                        "1",
                        os.path.join(hf, rel),
                    ],
                    check=True,
                )
            words = attach_punct(text, merge_split_terms(text, words))
            dur = duration(os.path.join(hf, rel))
            voices.append(
                {
                    "id": lid,
                    "path": rel,
                    "duration_s": dur,
                    "words": [{"id": f"w{i}", **w} for i, w in enumerate(words)],
                }
            )
            print(f"  voice {lid}: {rel} ({dur}s, {len(words)} words)", file=sys.stderr)

    meta = {
        "tts_provider": "edge-tts" if "tts" in only else prev.get("tts_provider"),
        "voice_id": voice if "tts" in only else prev.get("voice_id"),
        "bgm": None if "bgm" in only else prev.get("bgm"),
        "bgm_pending": False,
        "voices": voices,
        "sfx": [] if "sfx" in only else prev.get("sfx", []),
        "total_duration_s": round(sum(v["duration_s"] for v in voices), 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"✓ edge-tts engine → {args.out} · voices: {len(voices)} · total {meta['total_duration_s']}s"
    )


if __name__ == "__main__":
    main()
