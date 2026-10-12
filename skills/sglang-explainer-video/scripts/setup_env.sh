#!/usr/bin/env bash
# Prepare (idempotently) everything sglang-explainer-video needs, then print the env file to source.
#
#   bash setup_env.sh            # workspace defaults to ~/.cache/sglang-explainer-video
#   SGLANG_VIDEO_HOME=/path bash setup_env.sh
#
# Installs nothing globally: a Python venv, a pinned HyperFrames checkout (skills + scripts),
# and the Noto SC fonts all live in the workspace. Needs Node >= 22, ffmpeg/ffprobe, git, curl,
# and a Chrome/Chromium for the still cover PNG.
set -euo pipefail

SKILL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${SGLANG_VIDEO_HOME:-$HOME/.cache/sglang-explainer-video}"
HF_COMMIT="${HF_COMMIT:-d0b551d0330d4d2cd932348a911b5457ed1ed3c8}" # heygen-com/hyperframes, 2026-10-12
HF_CLI_VERSION="${HF_CLI_VERSION:-0.8.149}"
mkdir -p "$WS"

# 1. Node >= 22 (HyperFrames CLI engines field). A keg-only Homebrew node@22 is fine.
NODE_BIN=""
for c in "$(command -v node || true)" /opt/homebrew/opt/node@22/bin/node /usr/local/opt/node@22/bin/node; do
  if [ -n "$c" ] && [ -x "$c" ] && [ "$("$c" -p 'process.versions.node.split(".")[0]')" -ge 22 ]; then
    NODE_BIN="$(dirname "$c")"
    break
  fi
done
if [ -z "$NODE_BIN" ]; then
  echo "✗ Node >= 22 not found. macOS: brew install node@22 (keg-only; your default node is untouched)." >&2
  exit 1
fi

# 2. ffmpeg + ffprobe (HyperFrames render/audio, QA contact sheets).
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "✗ ffmpeg/ffprobe not found. macOS: brew install ffmpeg" >&2
  exit 1
fi

# 3. Chrome/Chromium for the still cover PNG.
CHROME_BIN="${CHROME:-}"
if [ -z "$CHROME_BIN" ]; then
  for c in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "$(command -v google-chrome || true)" "$(command -v chromium || true)" "$(command -v chromium-browser || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then CHROME_BIN="$c"; break; fi
  done
fi
[ -n "$CHROME_BIN" ] || echo "⚠ no Chrome/Chromium found — set CHROME=... before make_cover.py renders cover.png" >&2

# 4. Python venv: edge-tts (narration + word boundaries), fonttools (CJK subsetting), pillow.
PY="${PYTHON:-python3}"
[ -x "$WS/venv/bin/python" ] || "$PY" -m venv "$WS/venv"
"$WS/venv/bin/pip" install -q "edge-tts>=7.2" fonttools brotli pillow

# 5. HyperFrames (pinned) — the pr-to-video skill + scripts this workflow drives — plus our CJK/bilingual
#    caption patch. Cloned into the workspace instead of `npx hyperframes skills` so nothing global changes.
if [ ! -d "$WS/hyperframes/.git" ]; then
  git clone -q --filter=blob:none https://github.com/heygen-com/hyperframes.git "$WS/hyperframes"
fi
git -C "$WS/hyperframes" fetch -q origin "$HF_COMMIT" 2>/dev/null || true
git -C "$WS/hyperframes" -c advice.detachedHead=false checkout -q "$HF_COMMIT"
PATCH="$SKILL_ROOT/patches/hyperframes-captions-cjk-bilingual.patch"
if git -C "$WS/hyperframes" apply --check "$PATCH" 2>/dev/null; then
  git -C "$WS/hyperframes" apply "$PATCH"
  echo "· applied $(basename "$PATCH")"
fi

# 6. Noto Sans/Serif SC (OFL, google/fonts). Never use local("PingFang SC") etc.: the .ttc picks the
#    wrong face in HyperFrames Studio and renders random glyphs (乱码). subset_fonts.py ships subsets.
mkdir -p "$WS/fonts"
for pair in "notosanssc/NotoSansSC%5Bwght%5D.ttf:NotoSansSC.ttf" "notoserifsc/NotoSerifSC%5Bwght%5D.ttf:NotoSerifSC.ttf"; do
  src="${pair%%:*}"
  dst="${pair##*:}"
  [ -s "$WS/fonts/$dst" ] || curl -fsSL -o "$WS/fonts/$dst" "https://raw.githubusercontent.com/google/fonts/main/ofl/$src"
done

# 7. `hf` wrapper for the pinned CLI (a plain $HF variable does not word-split in zsh).
mkdir -p "$WS/bin"
printf '#!/usr/bin/env bash\nexec npx -y hyperframes@%s "$@"\n' "$HF_CLI_VERSION" >"$WS/bin/hf"
chmod +x "$WS/bin/hf"

# 8. Env file every later command sources.
cat >"$WS/env.sh" <<EOF
export PATH="$WS/bin:$NODE_BIN:\$PATH"
export HYPERFRAMES_SKIP_SKILLS=1 DO_NOT_TRACK=1 HYPERFRAMES_NO_TELEMETRY=1
export SEV_SKILL="$SKILL_ROOT"
export SEV_WS="$WS"
export SKILL_DIR="$WS/hyperframes/skills/pr-to-video"
export MEDIA_DIR="$WS/hyperframes/skills/media-use"
export HF_MEDIA_ENGINE="$SKILL_ROOT/scripts/edge_tts_engine.mjs"
export EDGE_ENGINE_PYTHON="$WS/venv/bin/python"
export SEV_PY="$WS/venv/bin/python"
export CHROME="$CHROME_BIN"
EOF
PATH="$NODE_BIN:$PATH" npx -y "hyperframes@$HF_CLI_VERSION" --version >/dev/null
echo "✓ ready — run: source $WS/env.sh"
