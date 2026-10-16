#!/usr/bin/env bash
# Re-voice a finished video, keeping every frame: new take -> retime -> captions -> render.
#   revoice.sh <project dir> [stop-after: tts|retime|check]      (VOICE=zh-CN-... to override,
#   SKIP_TTS=1 to reuse the take already in audio_meta.json)
# Writes renders/video_revoiced.mp4; never touches renders/video.mp4.
# Safe to re-run: the old state is backed up once to revoice_backup/ and frames are restored from
# it before every retime (a chained WARP must never be composed twice). A project whose built take
# was lost by an earlier retime (no audio_meta.built.json) is chained through its existing WARP.
set -euo pipefail
source ~/.cache/sglang-explainer-video/env.sh
P=$1
STOP=${2:-}
VOICE=${VOICE:-zh-CN-XiaoxiaoNeural}
cd "$P"
B=revoice_backup
if [ ! -f $B/.done ]; then
  mkdir -p $B
  cp -R compositions $B/compositions
  cp STORYBOARD.md SCRIPT.md audio_meta.json $B/
  cp -R assets/voice $B/voice
  if [ -f audio_meta.built.json ]; then cp audio_meta.built.json $B/; else touch $B/.chain; fi
  touch $B/.done
fi
# reference take = the one the frames are cued to; without it (an earlier retime lost it) chain through WARP
CHAIN=""
if [ -f $B/.chain ]; then
  cp $B/audio_meta.json audio_meta.built.json
  CHAIN=--chain
fi

if [ "${SKIP_TTS:-0}" != 1 ]; then  # SKIP_TTS=1 keeps the take already in audio_meta.json
sed -i.bak -E "s/^\*\*Voice:\*\* [^ ]+/**Voice:** $VOICE/" SCRIPT.md && rm -f SCRIPT.md.bak
node $SKILL_DIR/scripts/audio.mjs --script ./SCRIPT.md --storyboard ./STORYBOARD.md --hyperframes . \
  --out ./audio_meta.json --voice $VOICE 2>&1 | (grep -E "·|✓|✗|rror" || true) | cut -c1-220
node $SKILL_DIR/scripts/audio.mjs sync-durations --audio-meta ./audio_meta.json --storyboard ./STORYBOARD.md | tail -1
cp audio_meta.json audio_meta.revoiced.json
fi
[ "$STOP" = tts ] && exit 0

rm -rf compositions/frames && cp -R $B/compositions/frames compositions/frames
$SEV_PY -I $SEV_SKILL/scripts/retime_frames.py --project "$P" --ref-meta audio_meta.built.json $CHAIN | tail -3
echo "retimed: $(grep -l 'var WARP = ' compositions/frames/*.html | wc -l | tr -d ' ') / $(ls compositions/frames/*.html | wc -l | tr -d ' ') frames"
[ "$STOP" = retime ] && exit 0

$SEV_PY -I $SEV_SKILL/scripts/pron_check.py --meta audio_meta.json --hyperframes . | tail -4
node $SKILL_DIR/scripts/captions.mjs build --storyboard ./STORYBOARD.md --audio-meta ./audio_meta.json \
  --hyperframes . --out ./caption_groups.json 2>&1 | tail -2
$SEV_PY -I $SEV_SKILL/scripts/subset_fonts.py --project "$P" | tail -2
node $SKILL_DIR/scripts/assemble-index.mjs --storyboard ./STORYBOARD.md --hyperframes . | tail -1
node $SKILL_DIR/scripts/transitions.mjs inject --storyboard ./STORYBOARD.md --hyperframes . | tail -1
node $SKILL_DIR/scripts/transitions.mjs verify --storyboard ./STORYBOARD.md --index ./index.html | tail -1
hf lint 2>&1 | tail -1
hf check 2>&1 | tail -1
[ "$STOP" = check ] && exit 0

hf render --skill=pr-to-video --quality high --output renders/video_revoiced.mp4 2>&1 | tail -2
ffprobe -v error -show_entries format=duration -of csv=p=0 renders/video_revoiced.mp4
echo REVOICE_DONE
