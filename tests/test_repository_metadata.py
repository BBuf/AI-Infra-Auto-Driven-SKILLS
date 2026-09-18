"""Plugin packaging invariants; versions and prose are not frozen."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_plugin_and_marketplace_agree():
    plugin = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    marketplace = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    entry = next(p for p in marketplace["plugins"] if p["name"] == plugin["name"])
    assert entry["version"] == plugin["version"]
