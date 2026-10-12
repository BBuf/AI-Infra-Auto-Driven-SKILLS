"""Ship Noto Sans/Serif SC subsets that cover every character the project renders.

    python subset_fonts.py --project DIR [--fonts-dir $SEV_WS/fonts]

Scans compositions/**/*.html, caption_groups.json, caption-translations.json, SCRIPT.md and
STORYBOARD.md, subsets the variable OFL fonts to woff2 in assets/fonts/, and fails if any CJK
character the frames use is still missing. Re-run after any text change (new narration,
captions or on-screen labels) — a missing glyph silently falls back to a system face.
"""

import argparse
import glob
import os
import string
import subprocess
import sys
import tempfile

from fontTools.ttLib import TTFont

EXTRA = "，。、；：！？（）《》「」“”‘’…—·×≈→←↑↓①②③④⑤⑥✱✓¼½ "
FONTS = (
    ("NotoSansSC.ttf", "NotoSansSC-Subset.woff2"),
    ("NotoSerifSC.ttf", "NotoSerifSC-Subset.woff2"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument(
        "--fonts-dir",
        default=os.path.join(
            os.environ.get(
                "SEV_WS", os.path.expanduser("~/.cache/sglang-explainer-video")
            ),
            "fonts",
        ),
    )
    a = ap.parse_args()
    proj = os.path.abspath(a.project)
    sources = glob.glob(
        os.path.join(proj, "compositions", "**", "*.html"), recursive=True
    )
    sources += [
        os.path.join(proj, n)
        for n in (
            "caption_groups.json",
            "caption-translations.json",
            "SCRIPT.md",
            "STORYBOARD.md",
        )
        if os.path.exists(os.path.join(proj, n))
    ]
    chars = set(string.printable) | set(EXTRA)
    for p in sources:
        with open(p, encoding="utf-8") as f:
            chars |= set(f.read())
    chars = {c for c in chars if c.isprintable() and c not in "\r\n\t"}
    out_dir = os.path.join(proj, "assets", "fonts")
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as t:
        t.write("".join(sorted(chars)))
        text_file = t.name
    used_cjk = set()
    for p in glob.glob(
        os.path.join(proj, "compositions", "**", "*.html"), recursive=True
    ):
        with open(p, encoding="utf-8") as f:
            used_cjk |= {c for c in f.read() if ord(c) > 0x2E80}
    ok = True
    for src, dst in FONTS:
        out = os.path.join(out_dir, dst)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "fontTools.subset",
                os.path.join(a.fonts_dir, src),
                f"--text-file={text_file}",
                "--flavor=woff2",
                "--layout-features=*",
                f"--output-file={out}",
            ],
            check=True,
        )
        cmap = TTFont(out).getBestCmap()
        missing = "".join(sorted(c for c in used_cjk if ord(c) not in cmap))
        print(
            f"✓ {dst}: {os.path.getsize(out) // 1024} KB · missing CJK: {missing or 'none'}"
        )
        ok &= not missing
    os.unlink(text_file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
