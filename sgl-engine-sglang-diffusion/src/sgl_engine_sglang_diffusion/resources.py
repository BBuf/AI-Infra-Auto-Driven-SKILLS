"""Paths to immutable runtime resources shipped inside the wheel."""

from pathlib import Path


RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
TECHNIQUE_REGISTRY = RESOURCE_ROOT / "techniques" / "registry.toml"
KNOWLEDGE_REGISTRY = RESOURCE_ROOT / "knowledge" / "registry.toml"
