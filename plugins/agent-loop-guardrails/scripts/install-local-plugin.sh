#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PLUGIN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLUGIN_NAME="agent-loop-guardrails"
HOME_PLUGIN_PATH="${HOME}/plugins/${PLUGIN_NAME}"
HOME_MARKETPLACE="${HOME}/.agents/plugins/marketplace.json"
CODEX_CONFIG="${CODEX_HOME:-${HOME}/.codex}/config.toml"

mkdir -p "${HOME}/plugins" "${HOME}/.agents/plugins"
ln -sfn "$PLUGIN_ROOT" "$HOME_PLUGIN_PATH"

python3 - "$HOME_MARKETPLACE" "$PLUGIN_NAME" <<'PY'
import json, os, sys
marketplace_path, plugin_name = sys.argv[1], sys.argv[2]
entry = {
    "name": plugin_name,
    "source": {
        "source": "local",
        "path": f"./plugins/{plugin_name}"
    },
    "policy": {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL"
    },
    "category": "Productivity"
}

if os.path.exists(marketplace_path):
    with open(marketplace_path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {
        "name": "local",
        "interface": {
            "displayName": "Local Plugins"
        },
        "plugins": []
    }

plugins = [p for p in data.get("plugins", []) if p.get("name") != plugin_name]
plugins.append(entry)
data["plugins"] = plugins
data.setdefault("name", "local")
data.setdefault("interface", {}).setdefault("displayName", "Local Plugins")

with open(marketplace_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
PY

python3 - "$CODEX_CONFIG" "$PLUGIN_NAME" <<'PY'
import pathlib, re, sys
config_path = pathlib.Path(sys.argv[1]).expanduser()
plugin_name = sys.argv[2]
key = f'[plugins."{plugin_name}@local"]'
if config_path.exists():
    text = config_path.read_text(encoding="utf-8")
else:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = ""

pattern = re.compile(rf'^\[plugins\."{re.escape(plugin_name)}@local"\]\n(?:.*\n)*?(?=^\[|\Z)', re.M)
block = f'{key}\nenabled = true\n'
if pattern.search(text):
    new_text = pattern.sub(block + "\n", text).rstrip() + "\n"
else:
    new_text = text.rstrip()
    if new_text:
        new_text += "\n\n"
    new_text += block
config_path.write_text(new_text, encoding="utf-8")
PY

echo "Installed local Codex plugin:"
echo "- symlink: $HOME_PLUGIN_PATH -> $PLUGIN_ROOT"
echo "- marketplace: $HOME_MARKETPLACE"
echo "- config enabled: $CODEX_CONFIG"
