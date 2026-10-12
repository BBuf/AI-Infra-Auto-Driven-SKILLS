"""Put the SGLang brand on a HyperFrames project (run after Step 1 ingest, before frames).

    python apply_brand.py --project DIR

1. merges the SGLang palette into capture/extracted/tokens.json (keeps title/description);
2. runs pr-to-video's build-frame.mjs, which remixes the code-editorial preset onto those
   colors (coral → SGLang orange, navy code surface → warm dark brown, cream ground kept light);
3. replaces .hyperframes/caption-skin.html with the CJK + bilingual caption skin;
4. copies the SGLang logos into assets/sglang/.
"""

import argparse
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
BRAND_COLORS = [
    {"hex": "#1F1410", "role": "ink"},
    {"hex": "#FBF6F1", "role": "canvas"},
    {"hex": "#D55816", "role": "accent"},
    {"hex": "#A5300F", "role": "accent-2"},
]
# The hue remix turns the preset's half-step surfaces into strong reds; pin them to light peach
# so cards stay readable (the brand voltage is the orange accent, not the card fill).
FRAME_OVERRIDES = {"tile": "#F9E6DB", "tile-strong": "#F2CFBE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    a = ap.parse_args()
    proj = os.path.abspath(a.project)
    tok_path = os.path.join(proj, "capture", "extracted", "tokens.json")
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    tokens = {}
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tokens = json.load(f)
    tokens["colors"] = BRAND_COLORS
    tokens.setdefault("fonts", [])
    with open(tok_path, "w") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)
    skill_dir = os.environ["SKILL_DIR"]
    subprocess.run(
        [
            "node",
            os.path.join(skill_dir, "scripts", "build-frame.mjs"),
            "--preset",
            "code-editorial",
            "--hyperframes",
            proj,
        ],
        check=True,
    )
    frame_md = os.path.join(proj, "frame.md")
    with open(frame_md) as f:
        md = f.read()
    for key, hexv in FRAME_OVERRIDES.items():
        md = re.sub(
            rf'(?m)^(  {re.escape(key)}: )"#[0-9A-Fa-f]{{6}}"',
            rf'\1"{hexv}"',
            md,
            count=1,
        )
    with open(frame_md, "w") as f:
        f.write(md)
    shutil.copy(
        os.path.join(ASSETS, "caption-skin.html"),
        os.path.join(proj, ".hyperframes", "caption-skin.html"),
    )
    os.makedirs(os.path.join(proj, "assets", "sglang"), exist_ok=True)
    for n in ("sglang-logo.png", "sglang-logo-square.png"):
        shutil.copy(
            os.path.join(ASSETS, "brand", n), os.path.join(proj, "assets", "sglang", n)
        )
    print(
        "✓ SGLang brand applied (frame.md remixed, CJK/bilingual caption skin, logos)"
    )


if __name__ == "__main__":
    main()
