---
version: alpha
name: Kimi dark — Frame (video / frame layer)
description: >
  A dark, quiet research-page register taken from kimi.com and the Kimi K3 tech blog: a near-black
  #121212 ground, soft off-white type in light weights, hairline panels, generous negative space, and
  a muted semantic diagram palette sampled from Moonshot's official architecture figures (blue, green,
  slate, brick, rose, sand). One Kimi-blue #1A88FF accent per frame, at most. First used for the
  Kimi K3 video (2026-10).
unit: the frame — 1920×1080
principle: quiet ground · light type · diagram colours carry meaning · numbers come from the script

colors:
  ink: "#EDEDED"
  canvas: "#121212"
  panel: "#1B1B1B"
  panel-2: "#232323"
  block: "#38373D"
  ink-2: "#9E9E9E"
  ink-3: "#757575"
  accent: "#1A88FF"
  kda: "#345074"
  kda-line: "#8FA3C4"
  moe: "#4A6552"
  moe-line: "#93B29C"
  routed: "#444864"
  routed-line: "#8E92B0"
  mla: "#6E403C"
  mla-line: "#B9837D"
  rose: "#C4908F"
  sand: "#A89C80"

borders: { hairline: "1px solid rgba(255,255,255,.10)", hairline-strong: "1px solid rgba(255,255,255,.18)", dashed: "2px dashed rgba(255,255,255,.28)" }
shadows: { none: "none" }

typography:
  body:      { fontFamily: "Noto Sans SC", px: 28, weight: 330, lineHeight: 1.6, color: "rgba(255,255,255,.62)" }
  lead:      { fontFamily: "Noto Sans SC", px: 34, weight: 330, lineHeight: 1.5, color: "#D6D6D6" }
  card-title:{ fontFamily: "Noto Sans SC", px: 34, weight: 450, lineHeight: 1.3, color: "#EDEDED" }
  label:     { fontFamily: "Noto Sans SC", px: 24, weight: 400, color: "rgba(255,255,255,.42)" }
  kicker:    { fontFamily: "JetBrains Mono", px: 24, weight: 400, tracking: "0.12em", color: "rgba(255,255,255,.42)" }
  code:      { fontFamily: "JetBrains Mono", px: 26, weight: 400, lineHeight: 1.6 }
  headline:  { fontFamily: "Noto Sans SC", px: 62, weight: 450, lineHeight: 1.25, tracking: "0.005em", color: "#EDEDED" }
  display:   { fontFamily: "Noto Sans SC", px: 96, weight: 450, lineHeight: 1.1, tracking: "-0.01em", color: "#EDEDED" }
  number-hero:{ fontFamily: "Noto Sans SC", px: 180, weight: 300, lineHeight: 1.0, tracking: "-0.02em", color: "#EDEDED" }
  number-unit:{ fontFamily: "Noto Sans SC", px: 40, weight: 330, color: "rgba(255,255,255,.42)" }

spacing:
  slide-pad: "120px"
  gap-md: "32px"
  radius-block: "10px"
  radius-card: "18px"
  radius-frame: "22px"
  radius-pill: "9999px"

components:
  panel-card:
    backgroundColor: "#1B1B1B"
    border: "1px solid rgba(255,255,255,.10)"
    rounded: "18px"
    description: "The content card. Flat fill one step above the ground plus a hairline. No shadow, no glow, no gradient."
  module-frame:
    border: "2px dashed rgba(255,255,255,.28)"
    rounded: "22px"
    description: "A dashed rounded frame that groups a module (as Moonshot's official figures do). Use it to say 'this is one unit'."
  diagram-block:
    rounded: "10px"
    border: "2px solid <the block's -line colour>"
    fill: "one of kda / moe / routed / mla / block"
    text: "Noto Sans SC 500, #F4F4F4"
    description: "Filled rounded rectangles in the diagram palette. The colour is the semantic, fixed per video in the Video direction (e.g. the K3 video: KDA / linear attention = kda blue, MoE / shared expert = moe green, routed expert = routed slate, MLA / Linear projections = mla brick, generic block / Embedding = block grey, residual / cross-layer paths = rose). Keep the meanings identical in every frame."
  wire:
    stroke: "rgba(255,255,255,.72), 2px, small solid arrowheads"
    description: "Data-flow lines. AttnRes / depth-retrieval paths are rose #C4908F (thin parallel bus lines + α circles: 22px radius, #1B1B1B fill, rose stroke, italic α)."
  kicker-dot:
    description: "The eyebrow: a 10px Kimi-blue dot + JetBrains Mono 24px, tracking .12em, white @42% — e.g. '● 02 · ATTENTION RESIDUALS'. Section number + short label, never a sentence."
  number-lockup:
    description: "A light (300) Noto Sans SC figure with a smaller white@42% unit / denominator ('16 / 896', '2.8T'). Light weight is the Kimi look — never bold numerals."
  code-surface:
    backgroundColor: "#161616 body, #1E1E1E title bar"
    border: "1px solid rgba(255,255,255,.08)"
    rounded: "14px"
    text: "JetBrains Mono 26px, #D6D6D6"
    syntax: "keywords #8FB3FF, strings #A9C7A0, numbers #D6B98C, comments rgba(255,255,255,.38), function names #E6E6E6"
    description: "The code window. Title bar holds the file path (mono 22px, white@42%) and three 10px dots in #3A3A3A. Highlight a focus line with a 4px left bar in Kimi blue (that is the frame's one accent) and a rgba(26,136,255,.08) row tint."
  accent:
    description: "Kimi blue #1A88FF appears at most ONCE per frame: the focus-line bar, one key number, one underline. The kicker dot does not count against the budget."
---

# Kimi dark — Frame

## The look in one paragraph

Think of a Kimi research-blog page (e.g. kimi.com/blog/kimi-k3) as a moving page. The ground is a flat
near-black #121212 with nothing on it but content: no vignettes, no gradients, no glows, no textures,
no tilt. Type is Noto Sans SC in light weights (330 body, 450 headings, 300 for big numbers) in
off-white at different opacities — hierarchy comes from size and opacity, not from bold or colour.
Content sits on flat #1B1B1B panels with 1px hairlines, or inside dashed rounded module frames, like
Moonshot's official architecture figures (when the project ships one under assets/article/, look at it
before you start: your diagrams must look like they belong next to it). Colour lives in the diagrams and means
something; the one Kimi-blue accent is rationed to a single moment per frame.

## Frame craft bar

- **Squint**: one focal element dominates (a diagram, a number, or a code window). The headline is
  62px/450 at top-left; everything else is smaller or lighter.
- **Ground**: #121212 everywhere. Panels #1B1B1B / #232323. Never pure black, never pure white text
  (max #EDEDED), never a light-themed card.
- **Type**: Noto Sans SC only for prose and numbers (Latin included, so "KDA", "token" match the
  Chinese), JetBrains Mono only for kickers, code and file paths. No serif, no italic except the α glyph
  and math variables.
- **Diagram palette is semantic** (see components.diagram-block). Do not invent new hues. Muted
  tints (the -line colours) for borders; fills are the darker values.
- **Accent budget**: at most one Kimi-blue #1A88FF moment per frame.
- **Layout**: slide-pad 120px. Kicker at y≈96, left-aligned headline at y≈150; content y 280–860;
  the bottom band y ≥ 897 belongs to the captions — nothing there.
- **Motion**: calm. Elements fade + rise 12–20px (power3.out, 0.5–0.8s), wires draw on with
  stroke-dashoffset, numbers count up, highlights slide. No bounce, no elastic, no spinning, no
  whole-frame camera moves.

## Treatments

1. **Statement** — kicker + a single 96px/450 line (display) left-aligned, optional 34px lead under it.
2. **Diagram** — kicker + headline + one big diagram built from diagram-blocks, wires, α nodes and
   module-frames; labels in Noto Sans SC 24–28px. Build it up in the order the narration names parts.
3. **Number** — number-lockup at left (180px/300) + a label + a supporting visual at right
   (grid, bar, or mini-diagram).
4. **Code** — code-surface taking ~60% width; right column holds 2–3 short annotation cards that
   appear as the narration reaches each line; focus line marked with the blue left bar.
5. **Formula** — formula typeset with HTML/CSS or SVG text (Noto Sans SC, italic for variables) on a
   panel-card; each term gets an annotation that fades in when spoken. No MathJax/KaTeX CDN.
6. **Comparison** — two panel-cards side by side (before/after, baseline vs new), identical structure,
   the difference highlighted with the frame's one accent.
7. **Figure** — an official figure (contain, never cropped/stretched/recoloured) on the ground with
   a mono source line under it (the Scene line's 「图源：…」), plus rose/blue outline boxes that point at
   the region being discussed.

## Numerals & claims (hard rule)

Never invent numbers, layer counts, dimensions or formulas. Every figure on screen is copied
character by character from the Scene line, which comes from public sources.

## Font loading (offline)

Paste inside every frame's <style>; never link Google Fonts, never use local() or PingFang:

```html
<style>
@font-face{font-family:"Noto Sans SC";font-weight:100 900;font-style:normal;font-display:block;src:url("assets/fonts/NotoSansSC-Subset.woff2") format("woff2");}
@font-face{font-family:"JetBrains Mono";font-weight:400;font-style:normal;font-display:block;src:url("assets/fonts/JetBrainsMono-400.woff2") format("woff2");}
@font-face{font-family:"JetBrains Mono";font-weight:700;font-style:normal;font-display:block;src:url("assets/fonts/JetBrainsMono-700.woff2") format("woff2");}
</style>
```
