#!/usr/bin/env node
// HF_MEDIA_ENGINE shim: HyperFrames' audio adapter spawns `node <engine> ...`; forward that CLI
// to the edge-tts Python engine next to this file (venv python from EDGE_ENGINE_PYTHON).
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const python = process.env.EDGE_ENGINE_PYTHON || "python3";
const r = spawnSync(python, ["-I", join(here, "edge_tts_engine.py"), ...process.argv.slice(2)], {
  stdio: "inherit",
});
process.exit(r.status ?? 1);
