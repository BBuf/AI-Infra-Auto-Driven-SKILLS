# SGLang-Auto-Driven-SKILLS

Agent skills for SGLang development and profiling.

This repo also includes a repo-local Codex plugin, `agent-loop-guardrails`, for stronger enforcement.
The plugin adds hooks that can block premature `Stop`, protect loop state files, and give you a local installation path for Codex without copying the plugin out of this repo.

## Plugin

The Codex plugin lives at:

```text
plugins/agent-loop-guardrails/
```

Key pieces:

```text
plugins/agent-loop-guardrails/
├── .codex-plugin/plugin.json
├── hooks/hooks.json
├── hooks/
│   ├── protected-write.sh
│   ├── protected-edit.sh
│   ├── protected-bash.sh
│   └── stop-guard.sh
├── scripts/
│   ├── init-agent-loop.sh
│   ├── record-review-verdict.sh
│   ├── show-agent-loop-status.sh
│   ├── close-agent-loop.sh
│   └── install-local-plugin.sh
└── skills/
    └── agent-loop-governed/
        └── SKILL.md
```

Repo-local marketplace metadata lives at:

```text
.agents/plugins/marketplace.json
```

## Install The Plugin For Local Codex

Run:

```bash
bash plugins/agent-loop-guardrails/scripts/install-local-plugin.sh
```

This will:

- symlink the repo plugin into `~/plugins/agent-loop-guardrails`
- create or update `~/.agents/plugins/marketplace.json`
- enable the plugin in `~/.codex/config.toml`

## Notes From Installing `humanize` As A Codex Plugin

When adapting [`humania-org/humanize`](https://github.com/humania-org/humanize) to local Codex plugin mode, these were the main gotchas:

- `codex` plugin support may be disabled by default. Check with `codex features list` and enable it with `codex features enable plugins`.
- Merely editing `~/.agents/plugins/marketplace.json` and `~/.codex/config.toml` is not enough. A plugin can show up as `enabled = true` but still be `installed = false` internally, and then fresh `codex exec` sessions will not expose it.
- The reliable path is to run Codex's own plugin install flow after the marketplace entry exists. In our case, calling the underlying `plugin/install` RPC was what flipped `installed` to `true`.
- Codex expects a real `.codex-plugin/plugin.json`. Upstream `humanize` currently ships Claude-oriented packaging and Codex skill-install docs, so plugin mode needs an extra Codex manifest layer.
- In Codex plugin manifests, `interface.defaultPrompt` must be a string. Using an array causes manifest parse failure and the plugin interface is dropped.
- Some `humanize` skill files assume skill-runtime hydration with `{{HUMANIZE_RUNTIME_ROOT}}`. For plugin mode, those paths had to be adapted to `${CLAUDE_PLUGIN_ROOT}` so the plugin can resolve its own `scripts/`, `hooks/`, and `prompt-template/` assets.
- Always verify from a fresh session. The easiest smoke test is a new `codex exec` run that asks which plugins/skills containing `humanize` or `agent-loop` are explicitly available.

If you need to debug plugin state directly, the most useful distinction is:

- `enabled = true` in `~/.codex/config.toml`: configured to load
- `installed = true` from Codex app-server `plugin/list`: actually installed and discoverable by Codex

## Structure

```text
skills/
├── h100/
│   └── SKILL.md
├── h100-sglang-diffusion/
│   └── SKILL.md
├── sglang-torch-profiler-analysis/
│   ├── SKILL.md
│   └── scripts/
│       ├── analyze_sglang_torch_profile.py
│       ├── analyze_sglang_llm_torch_profile.py
│       ├── analyze_sglang_profiler_overlap.py
│       └── profile_common.py
├── sglang-kimi-k2-k25-optimization/
│   ├── SKILL.md
│   └── references/
│       ├── playbook.md
│       └── pr-history.md
├── sglang-minimax-m2-m25-optimization/
│   ├── SKILL.md
│   └── references/
│       ├── playbook.md
│       └── pr-history.md
└── vllm-kimi-k2-k25-optimization/
    ├── SKILL.md
    └── references/
        ├── playbook.md
        └── pr-history.md
```

## Placeholders

The `h100` and `h100-sglang-diffusion` skills use placeholder values. Replace them before use:

| Placeholder | Meaning |
|---|---|
| `<your-h100-host>` | SSH host alias in `~/.ssh/config` |
| `<your-container>` | Docker container name on the remote host |
| `<your-repo-path>` | Absolute path to the SGLang repo inside the container |
| `<your-cache-path>` | Host path mounted as `/root/.cache` |
| `<your-hf-token>` | Hugging Face access token (never commit the real value) |

## Install

```bash
cp -r skills/h100 ~/.codex/skills/h100
# or
cp -r skills/h100 ~/.cursor/skills/h100
```
