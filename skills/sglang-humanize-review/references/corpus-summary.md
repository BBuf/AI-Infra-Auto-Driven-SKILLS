# SGLang Human Review Corpus Summary

- Repo: `sgl-project/sglang`
- Source PR years: `2024` to `2025` inclusive
- Generated at: `2026-06-04T11:31:44+00:00`
- Corpus file: `sglang-review-corpus-2024-2025.jsonl.gz`
- Threads: `17914`
- Comments in corpus: `39511`
- Human reviewer comments: `39511`
- Agent reviewer comments: `0`

## Collection Policy

- Pull requests are selected by PR `created_at` in the requested year range.
- Review/comment events are also capped at the requested end date.
- Pull requests authored by GitHub bots or obvious coding-agent accounts are excluded.
- Inline review comments are GitHub pull-review comments grouped by thread.
- Top-level PR conversation comments are included from GitHub PR comments and grouped by PR.
- Review submission bodies are included for COMMENT / REQUEST_CHANGES style review summaries; empty approvals are skipped.
- Comment bodies are kept in their original language; the corpus does not translate or drop non-English text.
- `diff_hunk` stores the code context that produced each review thread.

## Pull Request Stats

| Metric | Count |
| --- | ---: |
| `excluded_agent_prs` | 8 |
| `included_human_prs` | 11573 |
| `included_human_prs_2024` | 1772 |
| `included_human_prs_2025` | 9801 |
| `seen_after_window` | 119 |
| `window_prs` | 11581 |

## Comment Stats

| Metric | Count |
| --- | ---: |
| `conversation_agent_pr_conversation_comments_on_target_prs` | 4937 |
| `conversation_all_pr_conversation_comments_seen` | 25732 |
| `conversation_human_pr_conversation_comments_on_target_prs` | 19236 |
| `conversation_pr_conversation_comments_after_window_skipped` | 1559 |
| `conversation_threads` | 5422 |
| `inline_agent_reviewer_comments_on_target_prs` | 11760 |
| `inline_all_review_comments_seen` | 29332 |
| `inline_human_reviewer_comments_on_target_prs` | 17558 |
| `inline_threads` | 10531 |
| `submission_agent_review_submissions_on_target_prs` | 8243 |
| `submission_all_review_submissions_seen` | 28428 |
| `submission_empty_review_submissions_skipped` | 16677 |
| `submission_human_review_submissions_on_target_prs` | 2717 |
| `submission_review_submissions_after_window_skipped` | 791 |
| `submission_threads` | 1961 |

## Episode Types

| Type | Threads |
| --- | ---: |
| `inline_review_thread` | 10531 |
| `pr_conversation` | 5422 |
| `review_submission` | 1961 |

## Event Kinds

| Kind | Events |
| --- | ---: |
| `pr_conversation` | 19236 |
| `inline_review_comment` | 17558 |
| `review_submission` | 2717 |

## Review States

| State | Events |
| --- | ---: |
| `APPROVED` | 1161 |
| `CHANGES_REQUESTED` | 925 |
| `COMMENTED` | 628 |
| `DISMISSED` | 3 |

## Top Categories

| Category | Threads |
| --- | ---: |
| `models-quant` | 7552 |
| `correctness` | 6582 |
| `tests-ci` | 6014 |
| `gpu-kernel` | 4894 |
| `api-compat` | 4200 |
| `performance` | 3821 |
| `docs-examples` | 3561 |
| `memory-cache` | 3205 |
| `style-maintainability` | 2933 |
| `build-deps` | 2664 |
| `distributed-concurrency` | 2487 |
| `observability` | 2083 |
| `general-review` | 1125 |

## Code Languages

| Language | Threads |
| --- | ---: |
| `python` | 8150 |
| `conversation` | 5422 |
| `review` | 1961 |
| `markdown` | 743 |
| `rust` | 390 |
| `cuda` | 282 |
| `cpp` | 213 |
| `yaml` | 184 |
| `notebook` | 177 |
| `toml` | 102 |
| `shell` | 78 |
| `extensionless` | 57 |
| `text` | 45 |
| `rst` | 16 |
| `unknown` | 14 |
| `json` | 11 |
| `rocm` | 10 |
| `gb200` | 10 |
| `proto` | 9 |
| `dockerfile` | 8 |
| `npu` | 5 |
| `jinja` | 4 |
| `jpeg` | 3 |
| `jpg` | 3 |
| `sycl` | 2 |
| `xeon` | 2 |
| `musa` | 2 |
| `gif` | 1 |
| `mp4` | 1 |
| `jsonconfig` | 1 |
| `po` | 1 |
| `dev` | 1 |
| `in` | 1 |
| `sagemaker` | 1 |
| `hip` | 1 |
| `blackwell` | 1 |
| `router` | 1 |
| `png` | 1 |

## Comment Language Hints

| Hint | Comments |
| --- | ---: |
| `en_or_ascii` | 37061 |
| `non_ascii_other` | 2237 |
| `zh_or_cjk` | 212 |
| `ja` | 1 |

## Top Paths

| Path | Threads |
| --- | ---: |
| `<conversation>` | 7383 |
| `python/sglang/srt/server_args.py` | 318 |
| `python/sglang/srt/models/deepseek_v2.py` | 250 |
| `python/sglang/srt/managers/scheduler.py` | 206 |
| `python/sglang/srt/model_executor/model_runner.py` | 175 |
| `python/sglang/srt/managers/schedule_batch.py` | 145 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 124 |
| `python/sglang/srt/utils.py` | 109 |
| `docs/backend/server_arguments.md` | 106 |
| `python/sglang/srt/layers/moe/ep_moe/layer.py` | 90 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | 86 |
| `python/pyproject.toml` | 80 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 80 |
| `test/srt/run_suite.py` | 73 |
| `python/sglang/srt/layers/quantization/modelopt_quant.py` | 68 |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | 65 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | 65 |
| `python/sglang/srt/managers/cache_controller.py` | 63 |
| `python/sglang/srt/managers/io_struct.py` | 62 |
| `python/sglang/srt/openai_api/adapter.py` | 61 |
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | 61 |
| `python/sglang/srt/layers/rotary_embedding.py` | 61 |
| `python/sglang/srt/server.py` | 59 |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | 58 |
| `python/sglang/srt/entrypoints/http_server.py` | 57 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | 55 |
| `python/sglang/bench_serving.py` | 51 |
| `python/sglang/srt/configs/model_config.py` | 50 |
| `python/sglang/srt/layers/quantization/fp8.py` | 50 |
| `python/sglang/srt/layers/moe/topk.py` | 49 |

## Top Human Reviewers

| Reviewer | Comments |
| --- | ---: |
| `zhyncs` | 2687 |
| `merrymercy` | 2100 |
| `zhaochenyang20` | 1716 |
| `Fridge003` | 1531 |
| `fzyzcjy` | 1448 |
| `mickqian` | 988 |
| `ShangmingCai` | 876 |
| `CatherineSue` | 862 |
| `BBuf` | 834 |
| `JustinTong0323` | 811 |
| `yuan-luo` | 770 |
| `hnyls2002` | 736 |
| `yizhang2077` | 660 |
| `ispobock` | 651 |
| `HaiShaw` | 611 |
| `mingfeima` | 537 |
| `ch-wan` | 509 |
| `Alcanderian` | 441 |
| `slin1237` | 431 |
| `hebiao064` | 410 |
| `b8zhong` | 340 |
| `xiezhq-hermann` | 313 |
| `FlamingoPg` | 286 |
| `Qiaolin-Yu` | 286 |
| `Ying1123` | 271 |
| `yhyang201` | 268 |
| `ByronHsu` | 249 |
| `iforgetmyname` | 249 |
| `key4ng` | 230 |
| `trevor-m` | 222 |

## Query Examples

```bash
python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --query cuda --limit 5
python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --path python/sglang/srt --category correctness --limit 8
python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --query 'server_args' --format jsonl --limit 3
```
