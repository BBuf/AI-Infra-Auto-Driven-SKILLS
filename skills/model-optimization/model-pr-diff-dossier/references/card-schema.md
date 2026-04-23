# Diff-Reviewed PR Card Schema

Every model optimization PR card should follow this structure:

````markdown
### PR #12345 - Title

- Link: https://github.com/<org>/<repo>/pull/12345
- State: merged/open/closed
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
````

Rules:

- Keep snippets short. Prefer 5-12 high-signal changed lines per PR.
- Write every card manually after opening the PR diff and reading the changed source files.
- Do not bulk-fill cards from scripts, PR titles, or generated summaries.
- If the PR is docs-only or config-only, say that explicitly and quote the relevant command/config line.
- If the PR touches shared runtime files such as model loaders, config parsers, serving arguments, scheduler paths, attention backends, or tokenizer/chat-template code, call out cross-model blast radius.
- If the PR touches tests, include the test file names and what regression lane they represent.
