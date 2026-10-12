"""The unified SGLang explainer cover ("A · brand orange"): video frame + still PNG.

Writes ``compositions/frames/00-cover.html`` (a HyperFrames frame: template root first,
seek-safe GSAP entrances) and renders the same layout, fully revealed, to a still PNG.

    python make_cover.py --project DIR --series "SGLang PR 讲解" --badge "#43238" \
        --title-zh "mixed chunk 下，decode tail 该走哪个 kernel？" \
        --title-en "Route BF16 mixed-batch decode tails through decode attention" \
        --meta "trtllm_mha_backend.py · +63 / −9 · @zcnrex" \
        --duration 7.46 --cue-title 2.6 --cue-meta 5.2 --png DIR/cover.png

Design (fixed for every video): full-bleed SGLang orange #D55816; the SGL wordmark on a peach
#FADDCD card with a rust #A5300F border; the logo's node-graph motif, oversized and faint, at
right; white bold series label + peach underline; bold Chinese title, peach EB Garamond italic
English title, mono meta line; the badge (PR number or a keyword) in the logo-"L" style —
outlined italic with trailing stripes — bottom-right. Series label: "SGLang PR 讲解" for PRs,
"SGLang 技术解读" for article videos.
"""

import argparse
import html
import os
import shutil
import signal
import subprocess
import tempfile
import time

ORANGE, RUST, PEACH = "#D55816", "#A5300F", "#FADDCD"
HERE = os.path.dirname(os.path.abspath(__file__))
BRAND = os.path.join(os.path.dirname(HERE), "assets", "brand")


def graph_svg(w, h, sw=14):
    """The logo's motif: two nodes joined by curves into a </> code box (peach line art)."""
    return f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">
  <path d="M{w*0.27} {h*0.47} C{w*0.40} {h*0.48} {w*0.52} {h*0.58} {w*0.53} {h*0.70}" fill="none" stroke="{PEACH}" stroke-width="{sw}"/>
  <path d="M{w*0.48} {h*0.53} C{w*0.62} {h*0.52} {w*0.73} {h*0.42} {w*0.74} {h*0.30}" fill="none" stroke="{PEACH}" stroke-width="{sw}"/>
  <circle cx="{w*0.18}" cy="{h*0.46}" r="{w*0.10}" fill="{ORANGE}" stroke="{PEACH}" stroke-width="{sw}"/>
  <circle cx="{w*0.74}" cy="{h*0.20}" r="{w*0.10}" fill="{ORANGE}" stroke="{PEACH}" stroke-width="{sw}"/>
  <rect x="{w*0.38}" y="{h*0.70}" width="{w*0.30}" height="{h*0.19}" rx="{w*0.035}" fill="{ORANGE}" stroke="{PEACH}" stroke-width="{sw}"/>
  <path d="M{w*0.47} {h*0.75} l-{w*0.04} {h*0.045} l{w*0.04} {h*0.045} M{w*0.59} {h*0.75} l{w*0.04} {h*0.045} l-{w*0.04} {h*0.045} M{w*0.555} {h*0.745} l-{w*0.03} {h*0.10}" fill="none" stroke="{PEACH}" stroke-width="{sw*0.8}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def badge_size(text):
    return int(min(190, 900 / (0.62 * max(len(text), 1))))


def cover_markup(a, logo_src, caption_safe=False):
    """(css, body) shared by the frame and the still. Element ids are prefixed 'cv-'.

    caption_safe (the video frame): the meta line and badge sit above the bottom ~17% caption
    band and the English title narrows so it never meets the badge; the still keeps the layout.
    """
    meta_top, badge_bottom, en_width = (
        (800, 216, 960) if caption_safe else (930, 96, 1180)
    )
    size = badge_size(a.badge)
    trail = ", ".join(
        f"{k * size * 0.035:.1f}px 0 0 rgba(165,48,15,{op})"
        for k, op in enumerate((0.55, 0.35, 0.18), start=1)
    )
    css = f"""
  .cv-bg {{ position:absolute; left:0; top:0; width:1920px; height:1080px; background:{ORANGE}; }}
  .cv-graph {{ position:absolute; right:-120px; top:40px; opacity:0.22; }}
  .cv-logo {{ position:absolute; left:96px; top:88px; background:{PEACH}; border:5px solid {RUST};
             border-radius:22px; padding:22px 34px; }}
  .cv-logo img {{ height:84px; display:block; }}
  .cv-series {{ position:absolute; left:96px; top:300px; font-family:"Inter","Noto Sans SC",sans-serif;
               font-size:76px; font-weight:900; color:#FFFFFF; letter-spacing:0.02em; white-space:nowrap; }}
  .cv-rule {{ position:absolute; left:96px; top:420px; width:120px; height:10px; background:{PEACH};
             border-radius:5px; transform-origin:0 50%; }}
  .cv-title {{ position:absolute; left:96px; top:478px; width:1180px; font-family:"Inter","Noto Sans SC",sans-serif;
              font-size:58px; font-weight:700; color:#FFFFFF; line-height:1.3;
              text-wrap:balance; word-break:keep-all; line-break:strict; }}
  .cv-en {{ position:absolute; left:96px; top:660px; width:{en_width}px; font-family:"EB Garamond","Noto Serif SC",serif;
           font-size:40px; font-style:italic; color:{PEACH}; line-height:1.25; }}
  .cv-meta {{ position:absolute; left:96px; top:{meta_top}px; font-family:"JetBrains Mono","Noto Sans SC",monospace;
             font-size:28px; color:rgba(255,255,255,0.82); white-space:nowrap; }}
  .cv-badge {{ position:absolute; right:110px; bottom:{badge_bottom}px; font-family:"Inter","Noto Sans SC",sans-serif;
              font-size:{size}px; font-weight:900; font-style:italic; letter-spacing:-0.02em; line-height:1;
              color:{PEACH}; -webkit-text-stroke:6px {RUST}; paint-order:stroke fill; text-shadow:{trail};
              white-space:nowrap; }}"""
    e = html.escape
    body = f"""
    <div id="cv-bg" class="clip cv-bg" data-start="0" data-duration="__DUR__" data-track-index="0">
      <div id="cv-graph" class="cv-graph">{graph_svg(900, 1000)}</div>
    </div>
    <div id="cv-stage" class="clip" data-start="0" data-duration="__DUR__" data-track-index="1"
         style="position:absolute;left:0;top:0;width:1920px;height:1080px">
      <div id="cv-logo" class="cv-logo"><img src="{logo_src}" alt="SGLang"></div>
      <div id="cv-series" class="cv-series">{e(a.series)}</div>
      <div id="cv-rule" class="cv-rule"></div>
      <div id="cv-title" class="cv-title">{e(a.title_zh)}</div>
      <div id="cv-en" class="cv-en">{e(a.title_en)}</div>
      <div id="cv-meta" class="cv-meta">{e(a.meta)}</div>
      <div id="cv-badge" class="cv-badge">{e(a.badge)}</div>
    </div>"""
    return css, body


FACES_FRAME = """
      @font-face{font-family:"EB Garamond";font-weight:400;font-style:normal;font-display:block;src:url("assets/fonts/EBGaramond-400.woff2") format("woff2");}
      @font-face{font-family:"Inter";font-weight:400;font-style:normal;font-display:block;src:url("assets/fonts/Inter-400.woff2") format("woff2");}
      @font-face{font-family:"Inter";font-weight:700;font-style:normal;font-display:block;src:url("assets/fonts/Inter-700.woff2") format("woff2");}
      @font-face{font-family:"JetBrains Mono";font-weight:400;font-style:normal;font-display:block;src:url("assets/fonts/JetBrainsMono-400.woff2") format("woff2");}
      @font-face{font-family:"Noto Sans SC";font-weight:100 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSansSC-Subset.woff2") format("woff2");}
      @font-face{font-family:"Noto Serif SC";font-weight:200 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSerifSC-Subset.woff2") format("woff2");}"""


def write_frame(a):
    css, body = cover_markup(a, "assets/sglang/sglang-logo.png", caption_safe=True)
    total = round(a.duration + 0.5, 3)  # +0.5s: the outgoing crossfade overlap
    body = body.replace("__DUR__", str(total))
    t_title = a.cue_title if a.cue_title is not None else 1.0
    t_meta = a.cue_meta if a.cue_meta is not None else t_title + 1.6
    frame = f"""<template>
  <div id="root" data-composition-id="00-cover" data-start="0" data-duration="{total}" data-width="1920" data-height="1080">
    <style>{FACES_FRAME}
      #root {{ position:relative; width:1920px; height:1080px; overflow:hidden; container-type:size; }}{css}
    </style>
{body}
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
    <script>
      (function () {{
        var byId = function (id) {{ return document.getElementById(id); }};
        window.__timelines = window.__timelines || {{}};
        var tl = gsap.timeline({{ paused: true }});
        // brand ground holds; the node-graph motif and logo card arrive first
        tl.fromTo(byId("cv-graph"), {{ opacity: 0, scale: 0.94 }}, {{ opacity: 0.22, scale: 1, duration: 1.2, ease: "power3.out" }}, 0);
        tl.fromTo(byId("cv-logo"), {{ opacity: 0, y: -18 }}, {{ opacity: 1, y: 0, duration: 0.7, ease: "power3.out" }}, 0.1);
        tl.fromTo(byId("cv-series"), {{ opacity: 0, y: 24 }}, {{ opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }}, 0.4);
        tl.fromTo(byId("cv-rule"), {{ scaleX: 0 }}, {{ scaleX: 1, duration: 0.6, ease: "power3.out" }}, 0.8);
        tl.fromTo(byId("cv-badge"), {{ opacity: 0, x: 60 }}, {{ opacity: 1, x: 0, duration: 0.9, ease: "power3.out" }}, 1.0);
        // the title lands on the voiceover's topic cue, the meta line on the next cue
        tl.fromTo(byId("cv-title"), {{ opacity: 0, y: 18 }}, {{ opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }}, {t_title});
        tl.fromTo(byId("cv-en"), {{ opacity: 0, y: 12 }}, {{ opacity: 1, y: 0, duration: 0.7, ease: "power3.out" }}, {t_title + 0.45});
        tl.fromTo(byId("cv-meta"), {{ opacity: 0, y: 8 }}, {{ opacity: 1, y: 0, duration: 0.6, ease: "power3.out" }}, {t_meta});
        // white-on-orange text dissolves just before the outgoing crossfade (keeps contrast during the blend)
        tl.to(["#cv-logo", "#cv-series", "#cv-rule", "#cv-title", "#cv-en", "#cv-meta", "#cv-badge"].map(function (s) {{ return document.querySelector(s); }}),
              {{ opacity: 0, duration: 0.25, ease: "power2.in" }}, {round(max(a.duration - 0.2, 0), 3)});
        tl.set({{}}, {{}}, {total});
        window.__timelines["00-cover"] = tl;
      }})();
    </script>
  </div>
</template>
"""
    os.makedirs(os.path.join(a.project, "compositions", "frames"), exist_ok=True)
    os.makedirs(os.path.join(a.project, "assets", "sglang"), exist_ok=True)
    shutil.copy(
        os.path.join(BRAND, "sglang-logo.png"),
        os.path.join(a.project, "assets", "sglang", "sglang-logo.png"),
    )
    out = os.path.join(a.project, "compositions", "frames", "00-cover.html")
    with open(out, "w") as f:
        f.write(frame)
    print("✓ cover frame →", out)


def render_png(a):
    chrome = (
        os.environ.get("CHROME")
        or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    if not os.path.exists(chrome):
        print("⚠ no Chrome — skipped cover.png (set CHROME)")
        return
    ws_fonts = os.path.join(
        os.environ.get("SEV_WS", os.path.expanduser("~/.cache/sglang-explainer-video")),
        "fonts",
    )
    pf = os.path.join(os.path.abspath(a.project), "assets", "fonts")
    css, body = cover_markup(a, "file://" + os.path.join(BRAND, "sglang-logo.png"))
    faces = f"""
@font-face{{font-family:"EB Garamond";font-weight:400;src:url("file://{pf}/EBGaramond-400.woff2");}}
@font-face{{font-family:"Inter";font-weight:400;src:url("file://{pf}/Inter-400.woff2");}}
@font-face{{font-family:"Inter";font-weight:700;src:url("file://{pf}/Inter-700.woff2");}}
@font-face{{font-family:"JetBrains Mono";font-weight:400;src:url("file://{pf}/JetBrainsMono-400.woff2");}}
@font-face{{font-family:"Noto Sans SC";font-weight:100 900;src:url("file://{ws_fonts}/NotoSansSC.ttf");}}
@font-face{{font-family:"Noto Serif SC";font-weight:200 900;src:url("file://{ws_fonts}/NotoSerifSC.ttf");}}"""
    doc = (
        f'<!doctype html><html><head><meta charset="utf-8"><style>{faces}'
        f"html,body{{margin:0;width:1920px;height:1080px;overflow:hidden}}{css}</style></head>"
        f"<body>{body.replace('__DUR__', '1')}</body></html>"
    )
    td = tempfile.mkdtemp(prefix="sev-cover-")
    hp, png = os.path.join(td, "cover.html"), os.path.abspath(a.png)
    with open(hp, "w") as f:
        f.write(doc)
    if os.path.exists(png):
        os.remove(png)
    # Headless Chrome writes the PNG but may not exit on macOS: poll for a stable file, then kill.
    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--allow-file-access-from-files",
            "--window-size=1920,1080",
            "--virtual-time-budget=3000",
            f"--user-data-dir={td}/ud",
            f"--screenshot={png}",
            "file://" + hp,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    t0, last = time.time(), -1
    while time.time() - t0 < 60:
        if os.path.exists(png):
            size = os.path.getsize(png)
            if size > 0 and size == last:
                break
            last = size
        time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGKILL)
    shutil.rmtree(td, ignore_errors=True)
    print("✓ cover png →", png if os.path.exists(png) else "(failed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--series", default="SGLang PR 讲解")
    ap.add_argument(
        "--badge",
        required=True,
        help='"#43238" for a PR, a short keyword for an article',
    )
    ap.add_argument("--title-zh", required=True)
    ap.add_argument("--title-en", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument(
        "--duration", type=float, required=True, help="frame 1 voice length (s)"
    )
    ap.add_argument("--cue-title", type=float, help="voice time the title should land")
    ap.add_argument(
        "--cue-meta", type=float, help="voice time the meta line should land"
    )
    ap.add_argument("--png", help="also render the still cover here")
    ap.add_argument("--png-only", action="store_true")
    a = ap.parse_args()
    if not a.png_only:
        write_frame(a)
    if a.png:
        render_png(a)


if __name__ == "__main__":
    main()
