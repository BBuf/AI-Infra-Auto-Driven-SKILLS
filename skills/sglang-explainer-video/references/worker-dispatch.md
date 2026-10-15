# Frame-worker dispatch context

Paste this into every worker prompt (Agent tool, background), after the line
`ROLE FILE: $P/.hyperframes/frame-packets/_role.md` and an instruction to read the role first.
Fill the `<…>` slots. At most three workers; balance packets by count and size; keep frames that
share a stage (same diagram before/after, two code frames) on the same worker.

```text
## Dispatch context

- PROJECT_DIR: <absolute $P>
- Assigned packets (build in order; one output each at compositions/frames/<frame_id>.html):
  1. .hyperframes/frame-packets/<id>.md
  …
- <PR video | ARTICLE video>. <Article only: frames <ids> use the author's real figures
  assets/article/figN.png — allowed: <img> inside a hairline card, object-fit: contain, never cropped or filtered.>
- <Code frames: the <code-*> block is installed at compositions/<block>.html; fill it from the packet's
  ### Source excerpt and fit its timing to the Scene window (you may inline an adapted copy).>
- <Worker holding the LAST frame only: frame <id> is the video's last frame. Paste the SGLang follow strip from
  <absolute $SEV_SKILL>/assets/brand/follow-strip.html verbatim: its CSS goes into your <style>, its <div> inside #root.
  It sits at y 792–872, so keep that frame's own content above y≈770. Reveal it by the start of the last
  clause and keep it to the end. Never edit its text or URLs, and never narrate it.>
- Read only: the role file, your packets, frame.md (SGLang brand remix: cream #FBF6F1 ground, peach cards
  #F9E6DB / #F2CFBE, ink #1F1410, accent #D55816 at most once per frame, navy = warm near-black code surface)
  and the assets named above (the last-frame worker also reads follow-strip.html). Never open STORYBOARD.md,
  SCRIPT.md, capture/*, or other workers' frames.
- HARD RULES (each one broke an earlier run):
  1. Frame contract: the FIRST element inside <template> is the composition root
     <div id="root" data-composition-id="<frame_id>" data-start="0" data-duration="<packet duration>"
     data-width="1920" data-height="1080">, and EVERY <style> and <script> (including the gsap CDN load)
     lives INSIDE that root div.
  2. Chinese fonts: only the bundled faces —
     @font-face{font-family:"Noto Sans SC";font-weight:100 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSansSC-Subset.woff2") format("woff2");}
     @font-face{font-family:"Noto Serif SC";font-weight:200 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSerifSC-Subset.woff2") format("woff2");}
     plus the EB Garamond / Inter / JetBrains Mono faces from frame.md. NEVER local(), PingFang, Songti,
     Hiragino or any system CJK font (乱码 in Studio). Stacks: display "EB Garamond","Noto Serif SC",serif;
     body "Inter","Noto Sans SC",sans-serif; mono "JetBrains Mono","Noto Sans SC",monospace.
     No negative letter-spacing on Chinese text.
  3. Every element id and class starts with a letter (prefix f<NN>-…), never a digit.
  4. Copy every on-screen string and number exactly from the packet; technical terms stay English.
     Never invent figures; never show PR status (OPEN / in review).
  5. Primary content stays above y≈896 (bottom ~17% = bilingual caption band).
  6. Timeline length = the packet's duration; reveal each element at the second its Scene line names.
- Self-check before finishing (allowed; not the hyperframes CLI):
  source ~/.cache/sglang-explainer-video/env.sh && node $SEV_SKILL/scripts/check_frames.mjs <PROJECT_DIR>
  — your files must print OK.
- Do not run the hyperframes CLI; edit only your output files (and an installed block you were told to fill).
  Reply with one line per file: path + one-sentence summary.
```

Retry (once per frame): SendMessage to the same worker with the exact validator/lint finding and the
check command; ask it to change only what the finding names.
