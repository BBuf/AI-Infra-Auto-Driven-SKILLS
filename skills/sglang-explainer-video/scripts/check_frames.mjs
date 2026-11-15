#!/usr/bin/env node
// Validate every compositions/frames/*.html against pr-to-video's frame contract (the same check
// assemble-index.mjs runs), flag system-font CJK faces, which garble in HyperFrames Studio, and require
// the SGLang ending block (assets/brand/follow-strip.html) on the last frame: the links card with this
// video's source and cookbook / docs links, and the follow strip.
//   node check_frames.mjs <project-dir>
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const proj = process.argv[2] || ".";
const { validateFrameHtml } = await import(join(process.env.SKILL_DIR, "scripts/lib/frame-contract.mjs"));
const dir = join(proj, "compositions/frames");
// The strip's three links, as follow-strip.html writes them (each URL is a whole <span>).
const FOLLOW = ["github.com/sgl-project/sglang", "sglang.io/blog", "docs.sglang.io"];
// The links card needs one URL of each kind (house-style.md §Ending links).
const LINKS = [
  ["a source link (github.com/sgl-project/<repo>/blob|tree/…)", /^github\.com\/sgl-project\/[\w.-]+\/(blob|tree)\/\S+$/],
  ["a cookbook or docs page (cookbook.sglang.io, docs.sglang.io/…)", /^(cookbook\.sglang\.io(\/\S*)?|docs\.sglang\.io\/\S+)$/],
];
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
    const urls = [...html.matchAll(/class="fs-links-url">([^<]*)</g)].map((m) => m[1].trim());
    const lacking = LINKS.filter(([, re]) => !urls.some((u) => re.test(u))).map(([kind]) => kind);
    if (!html.includes('id="fs-links"') || lacking.length)
      problems.push(
        `last frame lacks the ending links card (#fs-links in assets/brand/follow-strip.html)` +
          (lacking.length ? `: missing ${lacking.join(" and ")}` : ""),
      );
    const long = urls.filter((u) => u.length > 112);
    if (long.length) problems.push(`ending link longer than 112 characters, it overflows the row: ${long.join(", ")}`);
  }
  if (problems.length) bad++;
  console.log(problems.length ? `FAIL ${f} — ${problems.join("; ")}` : `OK   ${f}`);
}
process.exit(bad ? 1 : 0);
