"""Shared parts of a per-video cover (house-style.md §Cover).

Every video gets its own cover design, so the design lives in the project's ``work/make_cover.py``:
a CSS function, a body-markup function and the GSAP reveal steps. This module supplies what is the
same for every cover.

    import cover_kit as ck

    def css(width, dy):  # layout for a width x 1080 canvas: 1920 (frame, 16:9 still) or 1440 (Bilibili);
        return "#cv{position:absolute;inset:0;background:#F1ECE0} #cv-in{position:absolute;inset:0;top:%dpx} ..." % dy

    def body(prefix):    # markup inside <div id="cv">; prefix is "" in the frame, "file://<project>/" in stills
        return '<div id="cv-tag">SGLang PR 讲解 · #47013</div> ...'

    ck.write_stills(P, css, body)                    # cover.png (1920x1080) + cover_bili.png (1440x1080)
    ck.write_frame(P, css, body, voice_s, timeline, fade=["#cv-tag", "#cv-title"])

``timeline`` is JavaScript that adds tweens to ``tl`` (``tl.fromTo("#cv-title", {...}, {...}, 2.6);``).
``fade`` lists the text that dissolves just before the crossfade into frame 2.

One layout serves the frame and both stills when its content sits inside y 135-880. That keeps it above
the caption band in the video. ``dy`` is 0 for the frame and STILL_DY for the stills: the design moves its
content (not its ground) down by that much, which centres it in the 16:9 band (y 135-945) that Bilibili
crops from the 1440x1080 cover.

Compare a new design with the covers already published before keeping it:

    python cover_kit.py wall --out wall.png [--add cover_bili.png] [--last 8]
"""

import argparse
import glob
import os

from make_cover import _shoot, _ws_fonts

STILL_DY = 40

FRAME_FACES = """
@font-face{font-family:"EB Garamond";font-weight:400;font-style:normal;font-display:block;src:url("assets/fonts/EBGaramond-400.woff2") format("woff2");}
@font-face{font-family:"EB Garamond";font-weight:700;font-style:normal;font-display:block;src:url("assets/fonts/EBGaramond-700.woff2") format("woff2");}
@font-face{font-family:"Inter";font-weight:400;font-style:normal;font-display:block;src:url("assets/fonts/Inter-400.woff2") format("woff2");}
@font-face{font-family:"Inter";font-weight:700 900;font-style:normal;font-display:block;src:url("assets/fonts/Inter-700.woff2") format("woff2");}
@font-face{font-family:"JetBrains Mono";font-weight:400;font-style:normal;font-display:block;src:url("assets/fonts/JetBrainsMono-400.woff2") format("woff2");}
@font-face{font-family:"JetBrains Mono";font-weight:700;font-style:normal;font-display:block;src:url("assets/fonts/JetBrainsMono-700.woff2") format("woff2");}
@font-face{font-family:"Noto Sans SC";font-weight:100 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSansSC-Subset.woff2") format("woff2");}
@font-face{font-family:"Noto Serif SC";font-weight:200 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSerifSC-Subset.woff2") format("woff2");}
"""


def still_faces(project):
    """The frame's fonts for a standalone still: the full CJK fonts, so no glyph is missing."""
    ws = _ws_fonts()
    faces = FRAME_FACES.replace(
        'url("assets/fonts/', f'url("file://{project}/assets/fonts/'
    )
    for name in ("NotoSansSC", "NotoSerifSC"):
        faces = faces.replace(
            f'url("file://{project}/assets/fonts/{name}-Subset.woff2") format("woff2")',
            f'url("file://{ws}/{name}.ttf")',
        )
    return faces


def write_stills(project, css, body):
    """cover.png (1920x1080) and cover_bili.png (1440x1080), fully revealed."""
    project = os.path.abspath(project)
    for width, name in ((1920, "cover.png"), (1440, "cover_bili.png")):
        doc = (
            '<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0}'
            f"#stage{{position:relative;width:{width}px;height:1080px;overflow:hidden}}"
            f"{still_faces(project)}{css(width, STILL_DY)}</style></head>"
            f'<body><div id="stage"><div id="cv">{body("file://" + project + "/")}</div></div></body></html>'
        )
        _shoot(doc, os.path.join(project, name), width)


def write_frame(project, css, body, voice, timeline, fade):
    """compositions/frames/00-cover.html: frame 1, lasting the cover narration plus the crossfade."""
    total = round(voice + 0.5, 3)  # +0.5 s: the outgoing crossfade overlap
    fade_at = round(total - 0.7, 3)
    targets = ", ".join(f'"{s}"' for s in fade)
    frame = f"""<template>
  <div id="root" data-composition-id="00-cover" data-start="0" data-duration="{total}" data-width="1920" data-height="1080">
    <style>{FRAME_FACES}
#root {{ position:relative; width:1920px; height:1080px; overflow:hidden; }}
{css(1920, 0)}
    </style>
    <div id="cv-bg" class="clip" data-start="0" data-duration="{total}" data-track-index="0"
         style="position:absolute;left:0;top:0;width:1920px;height:1080px">
      <div id="cv">{body("")}</div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
    <script>
      (function () {{
        window.__timelines = window.__timelines || {{}};
        var tl = gsap.timeline({{ paused: true }});
{timeline.rstrip()}
        // the cover's text dissolves just before the crossfade, so it never blends with frame 2
        tl.to([{targets}], {{ opacity: 0, duration: 0.25, ease: "power2.in" }}, {fade_at});
        tl.set({{}}, {{}}, {total});
        window.__timelines["00-cover"] = tl;
      }})();
    </script>
  </div>
</template>
"""
    out = os.path.join(project, "compositions", "frames", "00-cover.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(frame)
    print("✓ cover frame →", out, f"({total} s)")


def wall(out, add=(), last=8, projects=None):
    """Tile the newest covers (and any --add candidates, outlined) into one sheet to compare designs."""
    projects = projects or os.path.join(
        os.environ.get("SEV_WS", os.path.expanduser("~/.cache/sglang-explainer-video")),
        "projects",
    )
    add = [os.path.abspath(p) for p in add]
    found = sorted(
        glob.glob(os.path.join(projects, "*", "cover_bili.png")), key=os.path.getmtime
    )
    recent = [p for p in found if p not in add][-last:]
    tiles = [(p, False) for p in recent] + [(p, True) for p in add]
    cells = "".join(
        f'<figure class="{"new" if new else ""}"><img src="file://{p}">'
        f"<figcaption>{'NEW · ' if new else ''}{os.path.basename(os.path.dirname(p))}</figcaption></figure>"
        for p, new in tiles
    )
    doc = (
        '<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:#fff}'
        "#w{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;padding:24px;width:1872px}"
        "figure{margin:0}img{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;border:1px solid #ccc}"
        "figure.new img{outline:6px solid #111}"
        "figcaption{font:16px/1.6 -apple-system,sans-serif;color:#333}"
        f'</style></head><body><div id="w">{cells}</div></body></html>'
    )
    _shoot(doc, out, 1920)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wall", help="contact sheet of recent covers")
    w.add_argument("--out", required=True)
    w.add_argument(
        "--add", action="append", default=[], help="a candidate cover to place last"
    )
    w.add_argument("--last", type=int, default=8)
    w.add_argument(
        "--projects", help="directory of video projects (default: $SEV_WS/projects)"
    )
    a = ap.parse_args()
    wall(a.out, a.add, a.last, a.projects)


if __name__ == "__main__":
    main()
