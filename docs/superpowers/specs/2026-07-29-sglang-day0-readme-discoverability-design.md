# SGLang Day-0 Skill Discoverability Design

## Goal

Make the merged `sglang-model-day0-support` skill discoverable and installable
through every repository-level entry point without changing the skill itself.

## Scope

Update:

- the root `README.md` overview, core-skill table, counts, installation
  commands, invocation examples, and repository map;
- `skills/model-optimization/README.md` so the directory documents both shared
  model-optimization skills;
- `.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json` so their
  descriptions include SGLang model Day-0 support and their versions advance
  from `0.1.0` to `0.2.0`;
- repository metadata tests so they enforce the new counts, installation
  entry, and matching plugin versions.

Do not add a per-skill `README.md`: the repository does not use that pattern
for core skills, and `SKILL.md` remains the authoritative operational guide.

## Public Counts

- Core skills: 12.
- Claude plugin skills after reload: 13, consisting of the 12 core skills plus
  the `model-pr-optimization-history` knowledge skill.

## README Presentation

Add the new skill next to `model-pr-diff-dossier` in the core table and model
optimization subtree. Describe it as the workflow for architecture gap maps,
parallel/kernel adaptation planning, seven release gates, public PR evidence,
and sanitized Kimi K3/DeepSeek V4 precedents.

Add its symlink and copy commands to the existing installation blocks and add
its invocation name to the examples. Keep the surrounding README structure
unchanged.

## Plugin Metadata

Use the same `0.2.0` version in both plugin metadata files. Extend descriptions
only enough to mention model Day-0 support; preserve the existing category,
tags, repository, and ownership fields.

## Validation

- Assert the root README contains `core_skills-12`, the 13-skill plugin count,
  the core table entry, both installation commands, and the repository-map
  entry.
- Assert both plugin metadata versions match and equal `0.2.0`.
- Run the focused repository metadata tests.
- Run `pre-commit run --all-files` and `git diff --check`.
- Push the single documentation/metadata commit directly to `main`.
