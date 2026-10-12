#!/usr/bin/env node
// Validate every compositions/frames/*.html against pr-to-video's frame contract (the same check
// assemble-index.mjs runs) and flag system-font CJK faces, which garble in HyperFrames Studio.
//   node check_frames.mjs <project-dir>
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const proj = process.argv[2] || ".";
const { validateFrameHtml } = await import(join(process.env.SKILL_DIR, "scripts/lib/frame-contract.mjs"));
const dir = join(proj, "compositions/frames");
let bad = 0;
for (const f of readdirSync(dir).filter((n) => n.endsWith(".html")).sort()) {
  const html = readFileSync(join(dir, f), "utf8");
  const problems = [];
  try {
    validateFrameHtml(html, { expectedId: f.replace(/\.html$/, "") });
  } catch (e) {
    problems.push(e.message);
  }
  if (/src:\s*local\(|PingFang|Songti|Hiragino|Microsoft YaHei/.test(html))
    problems.push('system CJK font (local()/PingFang/Songti) — use the bundled "Noto Sans SC"/"Noto Serif SC"');
  if (problems.length) bad++;
  console.log(problems.length ? `FAIL ${f} — ${problems.join("; ")}` : `OK   ${f}`);
}
process.exit(bad ? 1 : 0);
