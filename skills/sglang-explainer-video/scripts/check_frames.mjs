#!/usr/bin/env node
// Validate every compositions/frames/*.html against pr-to-video's frame contract (the same check
// assemble-index.mjs runs), flag system-font CJK faces, which garble in HyperFrames Studio, and require
// the SGLang follow strip (assets/brand/follow-strip.html) on the last frame.
//   node check_frames.mjs <project-dir>
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const proj = process.argv[2] || ".";
const { validateFrameHtml } = await import(join(process.env.SKILL_DIR, "scripts/lib/frame-contract.mjs"));
const dir = join(proj, "compositions/frames");
// The strip's three links, as follow-strip.html writes them (each URL is a whole <span>).
const FOLLOW = ["github.com/sgl-project/sglang", "sglang.io/blog", "docs.sglang.io"];
const frames = readdirSync(dir).filter((n) => n.endsWith(".html")).sort();
let bad = 0;
for (const f of frames) {
  const html = readFileSync(join(dir, f), "utf8");
  const problems = [];
  try {
    validateFrameHtml(html, { expectedId: f.replace(/\.html$/, "") });
  } catch (e) {
    problems.push(e.message);
  }
  if (/src:\s*local\(|PingFang|Songti|Hiragino|Microsoft YaHei/.test(html))
    problems.push('system CJK font (local()/PingFang/Songti) — use the bundled "Noto Sans SC"/"Noto Serif SC"');
  if (f === frames.at(-1)) {
    const missing = FOLLOW.filter((u) => !html.includes(`>${u}<`));
    if (!html.includes('id="fs-follow"') || missing.length)
      problems.push(
        `last frame lacks the SGLang follow strip (assets/brand/follow-strip.html)` +
          (missing.length ? `: missing ${missing.join(", ")}` : ""),
      );
  }
  if (problems.length) bad++;
  console.log(problems.length ? `FAIL ${f} — ${problems.join("; ")}` : `OK   ${f}`);
}
process.exit(bad ? 1 : 0);
