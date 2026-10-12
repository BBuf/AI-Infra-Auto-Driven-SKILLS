# Troubleshooting (every failure seen while building this workflow)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `npx hyperframes` errors / engines | HyperFrames CLI needs Node ≥ 22 | `brew install node@22`; `setup_env.sh` puts it first on PATH in `env.sh` |
| `init` writes into `~/.claude/skills` | init refreshes global skills | `HYPERFRAMES_SKIP_SKILLS=1` (set by `env.sh`); skills come from the pinned workspace checkout |
| `fetch-pr: gh pr view "owner/repo#N" failed … not a git repository` | gh can't resolve `owner/repo#N` outside a checkout | pass the full PR URL |
| edge-tts `WSServerHandshakeError: 403` | Oct 2026: legacy `speech.platform.bing.com` refuses synthesis | `edge_tts_engine.py` points `WSS_URL` at `api.msedgeservices.com` (rany2/edge-tts#491); `EDGE_TTS_LEGACY_HOST=1` to disable once upstream ships the fix |
| Kokoro Chinese sounds robotic | HyperFrames' offline Kokoro phonemizes Chinese via espeak | use the edge-tts engine (`HF_MEDIA_ENGINE`) — default in `env.sh` |
| 乱码 (random glyphs) in Studio, fine in snapshots | frames declared `@font-face{src:local("PingFang SC")}`; the .ttc picks the wrong face | bundle Noto SC subsets (`subset_fonts.py`); `check_frames.mjs` rejects `local()`/PingFang/Songti |
| `font_family_without_font_face` lint error | a CJK family is used without `@font-face` | declare the bundled Noto faces with `url("assets/fonts/…-Subset.woff2")` |
| A character renders in a fallback face | subset made before the text changed | re-run `subset_fonts.py` after every text edit (it fails if a used CJK glyph is missing) |
| `assemble-index: frame contract: composition id is missing on the template root` | worker put `<style>` before the root div | root div first inside `<template>`, styles/scripts inside it; re-dispatch with the finding |
| `frame packet is N bytes (limit 48000)` | blueprint body + cited rule recipes too large | drop one cited rule id from that frame's Scenes, or use `blueprint: compose` |
| 150+ `id_requires_css_escape` warnings | ids like `01-hook-num` start with a digit | harmless (workers use `getElementById`); new dispatches prefix ids with a letter |
| Captions: spaces between every Chinese word | captions.mjs joined words with `" "` | the CJK patch (`patches/…`) joins without spaces and keeps one gap at Latin seams |
| Captions: `FP 32`, `FLUX. / 2`, `torch. / equal` | edge-tts splits terms; `.` ended the group | engine `merge_split_terms` re-joins ASCII terms with no whitespace (or only `._-/`) between them |
| Captions flash 1-word groups in enumerations | TTS pauses after 、 triggered the silence split | patch: 、 is not a split, an overflow breaks after the last 、 |
| Re-running TTS shifts everything by 0.1–1 s | edge-tts takes are not bit-identical | keep `audio_meta.built.json`; `retime_frames.py` re-maps frames to the new take |
| Closing frame feels rushed | last line ends abruptly | engine pads the last line with `EDGE_TTS_PAD_LAST` (2 s) |
| `check` contrast warning on separators | `·` separators too faint during a crossfade | darken to `rgb(137,134,128)` |
| Headless Chrome screenshot never exits (macOS) | Chrome keeps running after writing the PNG | `make_cover.py` polls for a stable file and kills the process group |
| Cover PNG shows boxes | `make_cover.py --png` couldn't find the Noto TTFs | run `setup_env.sh` (downloads them to `$SEV_WS/fonts`) |
| Studio preview tiny / can't verify glyphs | Browser pane resolution | verify on contact sheets of the rendered MP4 (`qa_sheet.sh`) |
| ✱ / ✓ / ✗ show as tofu or a fallback face | not in EB Garamond / Inter / JetBrains Mono / Noto SC | draw them as inline SVG (workers do this), or accept the system symbol fallback on the render host |
| `code-typing` block missing in the project | `hf add` was skipped or ran outside `$P` | run `(cd $P && hf add code-typing)` before packets, or let the worker inline an adapted typing engine |
