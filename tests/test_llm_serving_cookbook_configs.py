from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "llm-serving-auto-benchmark"
CONFIG_ROOT = SKILL_ROOT / "configs" / "cookbook-llm"
VALIDATOR = SKILL_ROOT / "scripts" / "validate_cookbook_configs.py"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_cookbook_configs", VALIDATOR
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LlmServingCookbookConfigsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_validator()

    def config_paths(self) -> list[Path]:
        return sorted(CONFIG_ROOT.glob("*.yaml"))

    def load_config(self, path: Path) -> dict:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_all_cookbook_configs_validate(self) -> None:
        paths = self.config_paths()
        self.assertGreaterEqual(len(paths), 38)

        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(self.mod.validate_config(path), [])

    def test_every_config_has_three_framework_sections(self) -> None:
        for path in self.config_paths():
            with self.subTest(path=path.name):
                config = self.load_config(path)
                self.assertEqual(
                    set(config["frameworks"]),
                    {"sglang", "vllm", "tensorrt_llm"},
                )

                trt = config["frameworks"]["tensorrt_llm"]
                self.assertEqual(trt["backend_policy"], "fixed_pytorch")
                self.assertEqual(trt["base_server_flags"]["backend"], "pytorch")
                self.assertNotIn("backend", trt["search_space"])

    def test_rendered_commands_use_framework_native_flags(self) -> None:
        config = self.load_config(CONFIG_ROOT / "qwen3-235b-a22b.yaml")

        sglang = config["frameworks"]["sglang"]
        sglang_command = self.mod.render_command(
            "sglang", config, sglang["base_server_flags"]
        )
        self.assertIn("--model-path Qwen/Qwen3-235B-A22B", sglang_command)
        self.assertIn("--tp-size 8", sglang_command)

        vllm = config["frameworks"]["vllm"]
        vllm_command = self.mod.render_command(
            "vllm", config, vllm["base_server_flags"]
        )
        self.assertIn("vllm serve Qwen/Qwen3-235B-A22B", vllm_command)
        self.assertIn("--tensor-parallel-size 8", vllm_command)
        self.assertNotIn("--tp-size", vllm_command)

        trt = config["frameworks"]["tensorrt_llm"]
        trt_command = self.mod.render_command(
            "tensorrt_llm", config, trt["base_server_flags"]
        )
        self.assertIn("trtllm-serve serve Qwen/Qwen3-235B-A22B", trt_command)
        self.assertIn("--backend pytorch", trt_command)
        self.assertIn("--tp_size 8", trt_command)
        self.assertNotIn("--tp-size", trt_command)

    def test_glm_configs_explicitly_enable_trust_remote_code(self) -> None:
        glm_configs = (
            "glm-4.5.yaml",
            "glm-4.6.yaml",
            "glm-4.7.yaml",
            "glm-4.7-flash.yaml",
        )

        for config_name in glm_configs:
            config = self.load_config(CONFIG_ROOT / config_name)
            with self.subTest(path=config_name):
                for framework in self.mod.FRAMEWORKS:
                    flags = config["frameworks"][framework]["base_server_flags"]
                    self.assertIs(flags.get("trust_remote_code"), True)

    def test_help_snapshot_path_validates_all_config_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            help_dir = Path(tmp)
            for framework in self.mod.FRAMEWORKS:
                flag_text = "\n".join(
                    self.mod.flag_name(framework, key)
                    for key in sorted(self.mod.STATIC_SERVER_FLAGS[framework])
                )
                if framework == "sglang":
                    name = "sglang_launch_server.txt"
                elif framework == "vllm":
                    name = "vllm_serve_all.txt"
                else:
                    name = "trtllm_serve.txt"
                (help_dir / name).write_text(flag_text, encoding="utf-8")

            help_flags = self.mod.load_help_flags(help_dir)
            self.assertEqual(set(help_flags), set(self.mod.FRAMEWORKS))

            for path in self.config_paths():
                with self.subTest(path=path.name):
                    self.assertEqual(self.mod.validate_config(path, help_flags), [])

    def test_invalid_config_reports_errors_without_crashing(self) -> None:
        config = {
            "schema_version": 1,
            "source": {"kind": "llm_serving_cookbook"},
            "model": {"name": "example/model"},
            "dataset": {"input_len": [], "output_len": []},
            "frameworks": {
                "sglang": {
                    "enabled": True,
                    "base_server_flags": {},
                    "search_space": {},
                },
                "vllm": {"enabled": False},
                "tensorrt_llm": {"enabled": False},
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            errors = self.mod.validate_config(path)

        self.assertIn(
            "dataset.input_len and dataset.output_len must not be empty", errors
        )
        self.assertIn("search must be a mapping", errors)
        self.assertIn("sglang: server_command must be a string", errors)


if __name__ == "__main__":
    unittest.main()
