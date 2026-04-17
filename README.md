# SGLang-Auto-Driven-SKILLS

Agent skills for SGLang development and profiling.

## Structure

```
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
