"""Re-time already-built frames to a new narration take without rebuilding them.

Use when only the voice changed (new voice, reworded line, a fresh edge-tts take — takes are
not bit-identical, so word timings drift by a few hundred ms). Each frame's timeline stays cued
to the take it was built against; this wraps it in an outer timeline driven through a
piecewise-linear map whose anchors are clause starts matched across the two takes.

    python retime_frames.py --project DIR --ref-meta audio_meta.built.json [--chain] [--frames 02-hook=2 ...]

``--ref-meta`` is a copy of audio_meta.json from when the frames were built (keep one!).
``--frames`` maps frame file id → storyboard frame number; default: the STORYBOARD.md `- src:` of
each `## Frame N` (so 19b-*.html works), else frames/NN-*.html as frame NN+1 (cover = 00 → frame 1).
Re-running is safe: an existing WARP array is replaced — correct when --ref-meta is the take the
frames were built against. ``--chain``: --ref-meta is the take of an EARLIER re-time instead (the
built take is lost); the new anchors are composed through the frame's existing WARP.
"""

import argparse
import difflib
import glob
import json
import os
import re

PUNCT_END = re.compile(r"[，。；：！？、,.;:!?]$")


def norm(t):
    return re.sub(r"[，。；：！？、,.;:!?%]", "", t)


def anchors(old, new):
    ow, nw = old["words"], new["words"]
    sm = difflib.SequenceMatcher(
        None,
        [norm(w["text"]) for w in ow],
        [norm(w["text"]) for w in nw],
        autojunk=False,
    )
    match = {}
    for b in sm.get_matching_blocks():
        for k in range(b.size):
            match[b.a + k] = b.b + k
    pairs = [(0.0, 0.0)]
    for i in [0] + [
        i + 1 for i, w in enumerate(ow[:-1]) if PUNCT_END.search(w["text"])
    ]:
        j = i
        while (
            j < len(ow)
            and j not in match
            and not (j > i and PUNCT_END.search(ow[j - 1]["text"]))
        ):
            j += 1
        if j in match:
            n_t, o_t = nw[match[j]]["start"], ow[j]["start"]
            if n_t > pairs[-1][0] + 0.2 and o_t > pairs[-1][1] + 0.2:
                pairs.append((n_t, o_t))
    on, nn = old["duration_s"], new["duration_s"]
    pairs += [(nn, on), (nn + 0.5, on + 0.5)]
    return [[round(a, 3), round(b, 3)] for a, b in pairs], nn


WRAP = """
{ind}// Re-timed to a newer narration take: the timeline above is cued to the take it was built
{ind}// against; WARP maps [voiceTime, cueTime] (anchors = clause starts matched across takes).
{ind}var WARP = {warp};
{ind}var warpEnd = WARP[WARP.length - 1];
{ind}var warpEase = function (p) {{
{ind}  var t = p * warpEnd[0];
{ind}  for (var k = 1; k < WARP.length; k++) {{
{ind}    if (t <= WARP[k][0]) {{
{ind}      var a = WARP[k - 1], b = WARP[k];
{ind}      return (a[1] + ((t - a[0]) / (b[0] - a[0])) * (b[1] - a[1])) / warpEnd[1];
{ind}    }}
{ind}  }}
{ind}  return 1;
{ind}}};
{ind}var outer = gsap.timeline({{ paused: true }});
{ind}outer.add(tl.tweenFromTo(0, warpEnd[1], {{ duration: warpEnd[0], ease: warpEase }}), 0);
{ind}window.__timelines[{key}] = outer;"""


def interp(warp, t):
    """Piecewise-linear [voiceTime, cueTime] lookup, clamped at both ends."""
    if t <= warp[0][0]:
        return warp[0][1]
    for (a0, b0), (a1, b1) in zip(warp, warp[1:]):
        if t <= a1:
            return b0 + (t - a0) / (a1 - a0) * (b1 - b0) if a1 > a0 else b1
    return warp[-1][1]


def storyboard_frames(proj):
    """{frame file id: storyboard frame number} from STORYBOARD.md `## Frame N` + `- src:`."""
    out, n = {}, None
    path = os.path.join(proj, "STORYBOARD.md")
    if not os.path.exists(path):
        return out
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"## Frame (\d+)", ln)
        if m:
            n = m.group(1)
        m = re.match(r"- src: compositions/frames/(\S+)\.html", ln)
        if m and n:
            out[m.group(1)] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--ref-meta", required=True)
    ap.add_argument("--frames", nargs="*", default=[])
    ap.add_argument("--chain", action="store_true")
    a = ap.parse_args()
    proj = os.path.abspath(a.project)
    with open(a.ref_meta) as f:
        ref = {v["frame"]: v for v in json.load(f)["voices"]}
    with open(os.path.join(proj, "audio_meta.json")) as f:
        cur = {v["frame"]: v for v in json.load(f)["voices"]}
    mapping = dict(kv.split("=") for kv in a.frames) or storyboard_frames(proj)
    if not mapping:
        for p in sorted(
            glob.glob(os.path.join(proj, "compositions", "frames", "[0-9][0-9]-*.html"))
        ):
            fid = os.path.basename(p)[:-5]
            mapping[fid] = str(int(fid[:2]) + 1)
    for fid, n in mapping.items():
        n = int(n)
        if n not in ref or n not in cur:
            continue
        warp, nn = anchors(ref[n], cur[n])
        p = os.path.join(proj, "compositions", "frames", f"{fid}.html")
        with open(p) as f:
            s = f.read()
        old = re.search(r"var WARP = (\[\[.*?\]\]);", s)
        if a.chain and old:
            prev = json.loads(old.group(1))
            warp = [[n_t, round(interp(prev, o_t), 3)] for n_t, o_t in warp]
        if "var WARP = " in s:
            s = re.sub(
                r"var WARP = \[\[.*?\]\];",
                "var WARP = " + json.dumps(warp) + ";",
                s,
                count=1,
            )
        else:
            m = re.search(r"\n([ \t]*)window\.__timelines\[([^\]]+)\]\s*=\s*tl;", s)
            if not m:
                print(f"skip {fid}: no `window.__timelines[...] = tl;` registration")
                continue
            s = (
                s[: m.start()]
                + WRAP.format(ind=m.group(1), warp=json.dumps(warp), key=m.group(2))
                + s[m.end() :]
            )
        s = re.sub(
            r'data-duration="[\d.]+"', f'data-duration="{round(nn + 0.5, 3)}"', s
        )
        with open(p, "w") as f:
            f.write(s)
        print(f"✓ {fid}: {ref[n]['duration_s']}s → {nn}s, {len(warp)} anchors")


if __name__ == "__main__":
    main()
