"""Tests for trusted SGLang incident-request replay."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "sglang-prod-incident-triage"
    / "scripts"
    / "replay_trusted_request_dump.py"
)


def load_replay_module():
    spec = importlib.util.spec_from_file_location("replay_trusted_request_dump", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestReplayArgumentValidation(unittest.TestCase):
    def setUp(self):
        self.mod = load_replay_module()

    def test_rejects_nonpositive_speed(self):
        args = SimpleNamespace(speed=0, parallel=1, timeout=120)
        with self.assertRaisesRegex(ValueError, "--speed must be greater than zero"):
            self.mod.validate_replay_args(args)

    def test_rejects_nonpositive_parallelism(self):
        args = SimpleNamespace(speed=1, parallel=0, timeout=120)
        with self.assertRaisesRegex(
            ValueError, "--parallel must be greater than zero"
        ):
            self.mod.validate_replay_args(args)

    def test_rejects_nonpositive_timeout(self):
        args = SimpleNamespace(speed=1, parallel=1, timeout=0)
        with self.assertRaisesRegex(ValueError, "--timeout must be greater than zero"):
            self.mod.validate_replay_args(args)

    def test_accepts_positive_values(self):
        args = SimpleNamespace(speed=0.5, parallel=2, timeout=1)
        self.mod.validate_replay_args(args)


class TestReplayHttpHandling(unittest.TestCase):
    def setUp(self):
        self.mod = load_replay_module()

    def test_http_error_is_raised_before_response_body_is_consumed(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("server error")
        args = SimpleNamespace(
            host="127.0.0.1",
            port=30000,
            speed=1,
            ignore_eos=False,
            timeout=10,
        )
        record = ({"text": "hello"}, {}, 100.0, 101.0)

        with patch.object(self.mod.requests, "post", return_value=response):
            with self.assertRaises(requests.HTTPError):
                self.mod.run_one_request(record, args, 0.0, 100.0, 0)

        response.raise_for_status.assert_called_once_with()
        response.json.assert_not_called()
        response.iter_lines.assert_not_called()


if __name__ == "__main__":
    unittest.main()
