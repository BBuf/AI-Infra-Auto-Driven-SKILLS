# DeepSeek-V4 PR History

Evidence sweep:

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
- Manual diff review date: `2026-04-24`
- Searched paths: DeepSeek-V4 cookbook, command generator snippet, docs navigation.
- Searched PR terms: `DeepSeek V4`, `DeepSeek-V4`, `deepseek-v4-deployment`, `FP8`, `FP4`, `H200`, `GB300`, `B200`.

## Runtime and Docs Surfaces

- `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`
- `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`
- `docs_new/docs.json`
- `docs_new/cookbook/autoregressive/intro.mdx`

## Diff-Reviewed PR Cards

### PR #23605 - Add DeepSeek V4 cookbook

- Link: https://github.com/sgl-project/sglang/pull/23605
- State: merged at `2026-04-24T05:10:29Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `2113` lines, `4` files.
- Motivation: DeepSeek-V4 needed day-0 SGLang deployment docs with a concrete hardware and recipe matrix rather than a generic DeepSeek V3/V3.2 command. The launch surface is different because V4 has Flash and Pro variants, Blackwell FP4-mixed checkpoints, H200 FP8 repackagings, DeepEP buffer constraints, MTP recipe choices, and DeepSeek-V4 parsers.
- Key implementation: adds `DeepSeek-V4.mdx`, registers it in docs navigation, and creates `deepseek-v4-deployment.jsx`. The generator maps `(hardware, modelSize)` to model slugs, TP, multinode state, PD role TP, environment variables, and recipe-specific flags. Unverified cells are commented out with a banner so copying them is a shell no-op.
- Key code excerpts:

```jsx
const HW_SIZE_SPEC = {
  "b200|small": { slug: "deepseek-ai/DeepSeek-V4-Flash", tp: 4 },
  "b200|big": { slug: "deepseek-ai/DeepSeek-V4-Pro", tp: 8 },
  "gb300|big": { slug: "deepseek-ai/DeepSeek-V4-Pro", tp: 4 },
  "h200|big": { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-hopper>", tp: 16, multinode: true, nnodes: 2 },
};
```

```jsx
const BEING_VERIFIED_NOTE =
  "# NOTE: this recipe is being verified on the latest checkpoint";
```

- Reviewed files: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`, `docs_new/docs.json`, `docs_new/cookbook/autoregressive/intro.mdx`.
- Validation implications: command-generation tests should diff against the intended launch matrix and ensure unverified recipes remain commented until validated.

### PR #23617 - Publish the H200 Flash FP8 checkpoint slug

- Link: https://github.com/sgl-project/sglang/pull/23617
- State: merged at `2026-04-24T05:23:50Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `33` lines, `1` file.
- Motivation: H200 cannot run the default DeepSeek-V4 FP4-mixed repos. The first cookbook version used placeholders for Hopper checkpoints; once Flash FP8 was public, the generator needed to emit the real SGLang repackaging.
- Key implementation: replaces the H200 Flash placeholder with `sgl-project/DeepSeek-V4-Flash-FP8`, while leaving Pro temporarily as a placeholder in this PR.
- Key code excerpt:

```diff
-"h200|small": { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Flash-hopper>", tp: 4 },
+"h200|small": { slug: "sgl-project/DeepSeek-V4-Flash-FP8", tp: 4 },
```

- Reviewed files: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`.
- Validation implications: H200 Flash generated commands must use the `sgl-project` FP8 checkpoint and keep `SGLANG_DSV4_FP4_EXPERTS=0`.

### PR #23628 - Add H200 checkpoint warning to DeepSeek-V4 docs

- Link: https://github.com/sgl-project/sglang/pull/23628
- State: merged at `2026-04-24T07:06:31Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `24` lines, `1` file.
- Motivation: the command generator encodes the H200 checkpoint distinction, but readers may still copy the default DeepSeek slug manually.
- Key implementation: adds a cookbook note telling H200 users to use the `sgl-project` checkpoint.
- Key code excerpt:

```mdx
<Note>
For H200 GPU deployments, use the SGLang checkpoint under `sgl-project`, not the default DeepSeek checkpoint.
</Note>
```

- Reviewed files: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`.
- Validation implications: docs reviews should reject examples that point H200 at `deepseek-ai/DeepSeek-V4-*`.

### PR #23622 - Update DeepSeek-V4 Docker and recipe verification details

- Link: https://github.com/sgl-project/sglang/pull/23622
- State: merged at `2026-04-24T07:12:35Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `292` lines, `2` files.
- Motivation: the cookbook needed a copyable Docker launch skeleton and a more accurate verified-recipe set. The CP recipe also had duplicate `--max-running-requests` behavior that should be represented as a single effective value.
- Key implementation: adds a Docker `sglang serve <use args below>` block; expands `VERIFIED_RECIPES` to B200 Flash/Pro low-latency/balanced/max-throughput/CP and H200 Flash low-latency/balanced/max-throughput; removes recipe subtitles; and keeps a single CP `--max-running-requests` value with the Blackwell Pro override.
- Key code excerpts:

```diff
+docker run --gpus all \
+    --shm-size 32g \
+    -p 30000:30000 \
+    lmsysorg/sglang:deepseek-v4-blackwell \
+    sglang serve <use args below>
```

```jsx
const VERIFIED_RECIPES = new Set([
  "b200|small|low-latency",
  "b200|small|balanced",
  "b200|small|max-throughput",
  "b200|small|cp",
  "h200|small|max-throughput",
]);
```

- Reviewed files: `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx`, `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`.
- Validation implications: verification status is data. New recipe cells should not be uncommented without an end-to-end run against the latest checkpoint.

### PR #23634 - Update DeepSeek-V4 Pro FP8 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/23634
- State: merged at `2026-04-24T07:58:05Z`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `26` lines, `1` file.
- Motivation: H200 Pro moved from a placeholder to a published FP8 checkpoint.
- Key implementation: changes the H200 Pro slug to `sgl-project/DeepSeek-V4-Pro-FP8` and updates the comment to say both H200 repackagings are public.
- Key code excerpt:

```diff
-"h200|big": { slug: "<TO_BE_UPLOADED_DeepSeek-V4-Pro-FP8>", tp: 16, multinode: true, nnodes: 2 },
+"h200|big": { slug: "sgl-project/DeepSeek-V4-Pro-FP8", tp: 16, multinode: true, nnodes: 2 },
```

- Reviewed files: `docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx`.
- Validation implications: H200 Pro generation must emit multinode TP=16 and the SGLang FP8 checkpoint.

## Validation Notes

- DeepSeek-V4 support in current main is docs/recipe support; there is no new `python/sglang/srt/models/deepseek_v4.py` file in `bca3dd958`.
- The command generator is the source of truth for the day-0 deployment matrix. If runtime support lands later, add separate runtime PR cards instead of folding them into these docs-only cards.
