from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_open_pr_watch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_open_pr_watch", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_open_pr_watch"] = module
    spec.loader.exec_module(module)
    return module


class OpenPrWatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_gh_pagination_keeps_matches_from_later_pages(self) -> None:
        pages = [
            [{"number": 1, "title": "unrelated"}],
            [{"number": 2, "title": "MoE fusion", "updated_at": "2026-11-08"}],
        ]
        with mock.patch.object(self.mod, "run", return_value=json.dumps(pages)) as run:
            items = self.mod.collect_watch_items({"x": "owner/repo"}, ["MoE"], 1)
        self.assertEqual([item.number for item in items], [2])
        self.assertIn("--paginate", run.call_args.args[0])

    def test_anonymous_fallback_reads_until_the_final_page(self) -> None:
        first = mock.MagicMock()
        first.__enter__.return_value.read.return_value = b'[{"number":1}]'
        last = mock.MagicMock()
        last.__enter__.return_value.read.return_value = b'[]'
        with mock.patch.object(self.mod.request, "urlopen", side_effect=[first, last]) as fetch:
            rows = self.mod.http_fetch_open_prs("owner/repo", 1)
        self.assertEqual(rows, [{"number": 1}])
        self.assertIn("page=2", fetch.call_args.args[0].full_url)

    def test_rest_fallback_sends_token_and_explains_rate_limit(self) -> None:
        page = mock.MagicMock()
        page.__enter__.return_value.read.return_value = b"[]"
        with mock.patch.dict(self.mod.os.environ, {"GH_TOKEN": "t0ken"}, clear=False), \
             mock.patch.object(self.mod.request, "urlopen", return_value=page) as fetch:
            self.mod.http_fetch_open_prs("owner/repo", 100)
        self.assertEqual(fetch.call_args.args[0].get_header("Authorization"), "Bearer t0ken")

        limited = self.mod.error.HTTPError("https://api.github.com", 403, "rate limit", {}, None)
        env = {k: v for k, v in self.mod.os.environ.items() if k not in ("GH_TOKEN", "GITHUB_TOKEN")}
        with mock.patch.dict(self.mod.os.environ, env, clear=True), \
             mock.patch.object(self.mod.request, "urlopen", side_effect=limited):
            with self.assertRaisesRegex(RuntimeError, "anonymous rate limit"):
                self.mod.http_fetch_open_prs("owner/repo", 100)

    def test_one_failed_repository_cannot_produce_a_complete_report(self) -> None:
        with mock.patch.object(self.mod, "gh_search_open_prs", side_effect=[[], RuntimeError("offline")]), \
             mock.patch.object(self.mod, "http_fetch_open_prs", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "incomplete open PR inventory"):
                self.mod.collect_watch_items({"a": "a/repo", "b": "b/repo"}, ["MoE"])

    def test_new_terms_match_case_and_punctuation_once_per_pr(self) -> None:
        def fetch(repo: str, terms: list[str], per_page: int) -> list[dict]:
            return [
                {
                    "number": 7,
                    "title": "MINIMAX-M3 and kimi-k3 runtime",
                    "body": "Inkling plus Unlimited-OCR",
                    "html_url": f"https://github.com/{repo}/pull/7",
                    "updated_at": "2026-07-27T00:00:00Z",
                }
            ]

        self.mod.gh_search_open_prs = fetch
        items = self.mod.collect_watch_items(
            {"x": "owner/repo"},
            ["MiniMax M3", "Kimi K3", "Inkling", "Unlimited OCR"],
            10,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(
            set(items[0].terms),
            {"MiniMax M3", "Kimi K3", "Inkling", "Unlimited OCR"},
        )
        rendered = self.mod.render_markdown(items, generated_on="2026-07-27")
        self.assertEqual(rendered.count("[#7]("), 1)

    def test_render_markdown_groups_repos_and_terms(self) -> None:
        items = [
            self.mod.WatchItem(
                repo="sgl-project/sglang",
                number=29470,
                title="[GLM-5] Tune router GEMM",
                url="https://github.com/sgl-project/sglang/pull/29470",
                updated_at="2026-06-27",
                terms=("GLM-5", "MoE"),
            ),
            self.mod.WatchItem(
                repo="vllm-project/vllm",
                number=44835,
                title="MoE sum kernel | topk=5-8",
                url="https://github.com/vllm-project/vllm/pull/44835",
                updated_at="2026-06-26",
                terms=("MoE",),
            ),
        ]

        text = self.mod.render_markdown(items, generated_on="2026-06-27")

        self.assertIn("Generated: `2026-06-27`.", text)
        self.assertIn("## sgl-project/sglang", text)
        self.assertIn("[#29470](https://github.com/sgl-project/sglang/pull/29470)", text)
        self.assertIn("`GLM-5`, `MoE`", text)
        self.assertIn("MoE sum kernel \\| topk=5-8", text)

    def test_collect_uses_http_fallback_when_gh_fails(self) -> None:
        def fail_gh(repo: str, terms: list[str], per_page: int) -> list[dict]:
            raise RuntimeError("rate limited")

        def fallback(repo: str, per_page: int) -> list[dict]:
            return [
                {
                    "number": 1,
                    "title": "Qwen3.5 NVFP4 update",
                    "body": "",
                    "html_url": f"https://github.com/{repo}/pull/1",
                    "updated_at": "2026-06-27T00:00:00Z",
                }
            ]

        self.mod.gh_search_open_prs = fail_gh
        self.mod.http_fetch_open_prs = fallback

        items = self.mod.collect_watch_items({"x": "owner/repo"}, ["Qwen3.5"], 10)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].repo, "owner/repo")
        self.assertEqual(items[0].terms, ("Qwen3.5",))

    def test_collect_raises_when_all_repo_fetches_fail(self) -> None:
        def fail_gh(repo: str, terms: list[str], per_page: int) -> list[dict]:
            raise RuntimeError("rate limited")

        def fail_fallback(repo: str, per_page: int) -> list[dict]:
            raise RuntimeError("offline")

        self.mod.gh_search_open_prs = fail_gh
        self.mod.http_fetch_open_prs = fail_fallback

        with self.assertRaisesRegex(RuntimeError, "incomplete open PR inventory"):
            self.mod.collect_watch_items({"x": "owner/repo"}, ["Qwen3.5"], 10)


if __name__ == "__main__":
    unittest.main()
