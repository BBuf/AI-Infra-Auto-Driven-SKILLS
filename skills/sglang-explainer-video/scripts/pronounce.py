"""Spoken forms for the narration: what edge-tts reads vs what the captions show.

The narration keeps the written form (FlashInfer, flashmla_sparse_q8, 8×B300); only the text sent to
edge-tts is rewritten so the voice reads English terms correctly and without breaks:

    spoken, spans = to_spoken(text)       # synthesize `spoken`
    words = map_back(spoken, spans, words)  # WordBoundary words → written tokens (captions)

Rules, in order (data in pronunciation.json, measured on zh-CN-XiaoxiaoNeural):
1. patterns   — regex rewrites for symbols the voice misreads (8×B300 → 8 卡 B300, 3.7× → 3.7 倍,
                #42565 → 42565, +5% → 提升 5%).
2. lexicon    — whole-token respellings (FlashInfer → Flash Infer, VAE → V A E, LoRA → Lora); exact case
                 first, then case-insensitive (flashinfer). A hyphen between two letters becomes a space
                 (bit-exact → bit exact, top-k → top k).
3. identifiers — snake_case / SCREAMING_CASE never reach the voice with "_" (it says 下划线):
                 split on "_", map each part through `parts`, upper-case known acronyms and parts with
                 digits, lower-case the rest (SGLANG_DSA_FUSE_TOPK → SGLang DSA fuse top k).
4. spaces      — a space between two Han characters makes the voice pause ~0.3 s, so it is dropped
                 ("改回 -1" → "改回负1", not "改回 负 1").
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LEXICON_PATH = os.environ.get(
    "SEV_PRON_LEXICON", os.path.join(HERE, "pronunciation.json")
)
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-+]*[A-Za-z0-9]|[A-Za-z]")
HAN = re.compile(r"[\u4e00-\u9fff]")

_data = None


def data():
    global _data
    if _data is None:
        with open(LEXICON_PATH, encoding="utf-8") as f:
            _data = json.load(f)
    return _data


def identifier(tok):
    """snake_case / SCREAMING_CASE → words the voice reads as one phrase."""
    d = data()
    parts, acr = d["parts"], set(d["acronyms"])
    out = []
    for p in (x for x in tok.split("_") if x):
        k = p.lower()
        if k in parts:
            out.append(parts[k])
        elif k in acr or any(c.isdigit() for c in k):
            out.append(p.upper())
        else:
            out.append(k)
    return " ".join(out)


def spoken_token(tok):
    lex = data()["lexicon"]
    if tok in lex:
        return lex[tok]
    low = {k.lower(): v for k, v in lex.items()}
    if tok.lower() in low:  # flashinfer / FLASHINFER read like FlashInfer
        return low[tok.lower()]
    if "_" in tok:
        return identifier(tok)
    if re.search(
        r"[A-Za-z]-[A-Za-z]", tok
    ):  # bit-exact, top-k: the hyphen blurs the second word
        return re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", tok)
    return tok


def segments(text):
    """[(written, spoken)] covering `text`; written == spoken where nothing changes."""
    segs = [(text, text)]
    for pat, rep in data()["patterns"]:
        rx, nxt = re.compile(pat), []
        for w, s in segs:
            if w != s:
                nxt.append((w, s))
                continue
            pos = 0
            for m in rx.finditer(w):
                if m.start() > pos:
                    nxt.append((w[pos : m.start()],) * 2)
                nxt.append((m.group(0), m.expand(rep)))
                pos = m.end()
            if pos < len(w):
                nxt.append((w[pos:],) * 2)
        segs = nxt
    out = []
    for w, s in segs:
        if w != s:
            out.append((w, s))
            continue
        pos = 0
        for m in TOKEN.finditer(w):
            sp = spoken_token(m.group(0))
            if sp != m.group(0):
                if m.start() > pos:
                    out.append((w[pos : m.start()],) * 2)
                out.append((m.group(0), sp))
                pos = m.end()
        if pos < len(w):
            out.append((w[pos:],) * 2)
    return out


def to_spoken(text):
    """(spoken text, spans) with spans = [(start, end, written)] in spoken-text offsets."""
    chars, written = [], []  # chars: (char, index into `written` or None)
    for w, s in segments(text):
        k = None
        if w != s:
            written.append(w)
            k = len(written) - 1
        chars += [(c, k) for c in s]

    def near(i, step):
        while 0 <= i < len(chars) and chars[i][0] == " ":
            i += step
        return chars[i][0] if 0 <= i < len(chars) else ""

    chars = [
        (c, k)
        for i, (c, k) in enumerate(chars)
        if not (
            c == " "
            and HAN.match(near(i - 1, -1) or "a")
            and HAN.match(near(i + 1, 1) or "a")
        )
    ]
    spoken = "".join(c for c, _ in chars)
    spans = []
    for k, w in enumerate(written):
        at = [i for i, (_, kk) in enumerate(chars) if kk == k]
        spans.append((at[0], at[-1] + 1, w))
    return spoken, spans


def map_back(spoken, spans, words):
    """Collapse the WordBoundary words that fall inside a rewritten span into its written token.

    edge-tts segments Chinese on its own, so a word can straddle a span edge ("改回负1" → 改 | 回负 | 1);
    such a word is cut at the edge and its time shared out by characters.
    """
    out, pos, cur = [], 0, None
    for w in words:
        i = spoken.find(w["text"], pos)
        if i < 0:
            out.append(dict(w))
            cur = None
            continue
        j = pos = i + len(w["text"])
        cuts = sorted({i, j} | {e for sp in spans for e in sp[:2] if i < e < j})
        dur = w["end"] - w["start"]
        for a, b in zip(cuts, cuts[1:]):
            t0 = round(w["start"] + dur * (a - i) / (j - i), 3)
            t1 = round(w["start"] + dur * (b - i) / (j - i), 3)
            span = next((sp for sp in spans if sp[0] <= a < sp[1]), None)
            if span is None:
                out.append({**w, "text": spoken[a:b], "start": t0, "end": t1})
                cur = None
            elif span is cur:
                out[-1]["end"] = t1
            else:
                out.append({**w, "text": span[2], "start": t0, "end": t1})
                cur = span
    return out


def changes(text):
    """[(written, spoken)] for the tokens that get rewritten — for logs and pron_check."""
    return [(w, s) for w, s in segments(text) if w != s]
