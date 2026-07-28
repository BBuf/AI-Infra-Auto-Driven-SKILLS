# Diff-Reviewed PR Card Schema

Every model optimization PR card should follow this structure:

````markdown
### PR #12345 - Title

- Link: https://github.com/<org>/<repo>/pull/12345
- Source head: `<full 40-character repository commit>`
- Captured at: `YYYY-MM-DD`
- State: merged/open/closed-unmerged
- PR head: `<full 40-character PR head commit>`
- Merged at: `<timestamp or not merged>`
- Diff coverage: full diff fetched, N lines, M files
- Motivation:
  - ...
- Key implementation:
  - ...
- Key code excerpts:

```diff
...
```

- Reviewed files:
  - runtime: ...
  - tests: ...
  - docs: ...
- Validation implications:
  - ...
- Validation evidence:
  - ...
- Limitations:
  - ...
````

Rules:

- Keep snippets short. Prefer 5-12 high-signal changed lines per PR.
- Write every card manually after opening the PR diff and reading the changed source files.
- Do not bulk-fill cards from scripts, PR titles, or generated summaries.
- `Source head` is immutable and belongs to the framework repository. `PR head`
  is immutable and belongs to the reviewed PR. Do not substitute `main`,
  branch names, floating tags, or abbreviated SHAs.
- Classify PR state explicitly. An open PR is candidate evidence and must not be
  presented as behavior contained in the recorded source head. A closed PR that
  was not merged is `closed-unmerged`.
- If the PR is docs-only or config-only, say that explicitly and quote the relevant command/config line.
- If the PR touches shared runtime files such as model loaders, config parsers, serving arguments, scheduler paths, attention backends, or tokenizer/chat-template code, call out cross-model blast radius.
- If the PR touches tests, include the test file names and what regression lane they represent.
- Validation evidence names what was actually reported or reproduced.
  Validation implications describe what a maintainer should still run.
- Limitations must identify missing public evidence, unverified shapes/hardware,
  open-PR instability, or any other boundary on the card's conclusions.
