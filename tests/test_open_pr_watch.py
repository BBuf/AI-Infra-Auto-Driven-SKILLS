from __future__ import annotations

import importlib.util
import sys
import unittest
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

    def test_search_query_is_open_pr_repo_scoped(self) -> None:
        query = self.mod.search_query("sgl-project/sglang", "Qwen3.5")

        self.assertIn("repo:sgl-project/sglang", query)
        self.assertIn("is:pr", query)
        self.assertIn("is:open", query)
        self.assertIn("Qwen3.5", query)

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

        with self.assertRaisesRegex(RuntimeError, "all open PR fetches failed"):
            self.mod.collect_watch_items({"x": "owner/repo"}, ["Qwen3.5"], 10)


if __name__ == "__main__":
    unittest.main()
