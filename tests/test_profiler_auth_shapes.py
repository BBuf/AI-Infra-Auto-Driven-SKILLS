from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib import error, request

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/llm-torch-profiler-analysis/scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import profile_common as common

    spec = importlib.util.spec_from_file_location(
        "profiler_auth_triage", SCRIPTS / "analyze_llm_torch_profile.py"
    )
    triage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(triage)
finally:
    sys.path.pop(0)


class ProfilerAuthShapesTest(unittest.TestCase):
    def test_credentials_are_opt_in_validated_and_hidden_from_repr(self):
        with mock.patch.dict(os.environ, {"TEST_API": "data-key", "TEST_ADMIN": "control-key"}):
            auth = common.ProfilerAuth.from_env("TEST_API", "TEST_ADMIN")
            self.assertNotIn("data-key", repr(auth))
            self.assertNotIn("control-key", repr(auth))
            self.assertIsNone(common.ProfilerAuth.from_env().api_key)
        for value in (None, "", " ", "secret\nvalue", "secret\rvalue"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["TEST_API"] = value
                with self.assertRaises(ValueError) as raised:
                    common.ProfilerAuth.from_env("TEST_API")
                self.assertNotIn("secret", str(raised.exception))

    def test_control_key_falls_back_to_api_without_using_admin_for_inference(self):
        for auth, admin, expected in (
            (common.ProfilerAuth("api"), True, "Bearer api"),
            (common.ProfilerAuth("api", "admin"), True, "Bearer admin"),
            (common.ProfilerAuth("api", "admin"), False, "Bearer api"),
            (common.ProfilerAuth(None, "admin"), False, None),
            (common.ProfilerAuth(), True, None),
        ):
            with self.subTest(auth=auth, admin=admin):
                req = request.Request("http://server/start_profile")
                auth.authorize(req, admin=admin)
                self.assertEqual(req.get_header("Authorization"), expected)

    def test_redirect_does_not_forward_bearer_header(self):
        original = request.Request("http://server/v1/models")
        common.ProfilerAuth("api").authorize(original)
        redirected = request.HTTPRedirectHandler().redirect_request(
            original, None, 302, "Found", {}, "http://other-server/v1/models"
        )
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_shapes_use_framework_specific_payloads_and_keep_defaults(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                ts = common.build_remote_profiler_start_payload(
                    "tokenspeed", Path("/shared"), "run", "decode", record_shapes=enabled
                )
                self.assertIs(ts["record_shapes"], enabled)
                vllm = common.build_remote_profiler_start_payload(
                    "vllm", Path("/shared"), "run", "decode",
                    frozenset({"profiler_kwargs"}), record_shapes=enabled,
                )
                self.assertEqual(vllm, {"profiler_kwargs": {"record_shapes": enabled}})
        self.assertIsNone(common.build_remote_profiler_start_payload(
            "vllm", Path("/shared"), None, None
        ))
        self.assertFalse(common.build_remote_profiler_start_payload(
            "tokenspeed", Path("/shared"), None, None
        )["record_shapes"])

    def test_unsupported_shapes_fail_before_warmup_or_start(self):
        for framework, fields in (("trtllm", frozenset()), ("vllm", frozenset())):
            for enabled in (True, False):
                with self.subTest(framework=framework, enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                    with (
                        mock.patch.object(common, "discover_openai_model", return_value="model"),
                        mock.patch.object(common, "vllm_profile_session_fields", return_value=fields),
                        mock.patch.object(common, "send_probe_requests") as probe,
                        mock.patch.object(common, "start_remote_profiler") as start,
                    ):
                        with self.assertRaisesRegex(ValueError, "record_shapes|profiler_kwargs"):
                            common.run_remote_profiler(
                                "http://server", tmp, framework,
                                common.ProbePlan("x", 1, 1, 1, 1), 0,
                                record_shapes=enabled,
                            )
                        probe.assert_not_called()
                        start.assert_not_called()

    def test_cli_options_reach_mapping_and_formal_capture(self):
        args = triage.parse_triage_args([
            "--mapping-url", "http://mapping", "--formal-url", "http://formal",
            "--api-key-env", "TEST_API", "--admin-key-env", "TEST_ADMIN",
            "--record-shapes",
        ])
        with mock.patch.dict(os.environ, {"TEST_API": "api", "TEST_ADMIN": "admin"}):
            with (
                mock.patch.object(triage, "resolve_framework", return_value="sglang"),
                mock.patch.object(triage, "run_profiler", return_value=Path("/traces")) as capture,
                mock.patch.object(triage, "discover_trace_targets", return_value=([], None)),
            ):
                for label in ("mapping", "formal"):
                    triage.resolve_profile_targets(
                        label=label, input_path=None, url=f"http://{label}",
                        output_dir=None, profile_prefix=label, args=args,
                    )
                self.assertEqual(capture.call_count, 2)
                for call in capture.call_args_list:
                    self.assertTrue(call.kwargs["record_shapes"])
                    self.assertEqual(call.kwargs["auth"].api_key, "api")
                    self.assertEqual(call.kwargs["auth"].admin_key, "admin")
        self.assertIsNone(triage.parse_triage_args(["--input", "trace.json"]).record_shapes)
        self.assertFalse(triage.parse_triage_args([
            "--url", "http://server", "--no-record-shapes"
        ]).record_shapes)

    def test_authenticated_capture_discovery_probes_control_and_cleanup(self):
        # Exercise real HTTP requests without a GPU or a framework install.
        for framework in ("sglang", "vllm", "trtllm", "tokenspeed"):
            with self.subTest(framework=framework), tempfile.TemporaryDirectory() as tmp:
                seen = []
                active = {}

                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *_args):
                        pass

                    def respond(self, payload):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(payload).encode())

                    def authorized(self):
                        admin = self.path in {"/start_profile", "/stop_profile"}
                        expected = "Bearer control-key" if admin else "Bearer data-key"
                        seen.append((self.path, self.headers.get("Authorization")))
                        if self.headers.get("Authorization") != expected:
                            self.send_error(401)
                            return False
                        return True

                    def do_GET(self):
                        if not self.authorized():
                            return
                        if self.path == "/server_info":
                            self.respond({"internal_states": []} if framework == "sglang" else {})
                        elif self.path == "/openapi.json":
                            self.respond({"paths": {"/start_profile": {"post": {
                                "requestBody": {"content": {"application/json": {"schema": {
                                    "properties": {"profiler_kwargs": {}, "profile_prefix": {}, "max_iterations": {}}
                                }}}}
                            }}}})
                        else:
                            self.respond({"data": [{"id": "model", "owned_by": framework}]})

                    def do_POST(self):
                        if not self.authorized():
                            return
                        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                        payload = json.loads(raw) if raw else {}
                        if self.path == "/start_profile":
                            active.update(payload)
                        elif self.path == "/stop_profile":
                            output = Path(active.get("output_dir", tmp)) / "worker.trace.json"
                            output.write_text(json.dumps({"traceEvents": [
                                {"ph": "X", "cat": "kernel", "name": "kernel", "ts": 0, "dur": 1}
                            ]}))
                        self.respond({"text": "ok", "choices": [{"message": {"content": "ok"}}]})

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_port}"
                auth = common.ProfilerAuth("data-key", "control-key")
                try:
                    with mock.patch.object(common.time, "sleep"), contextlib.redirect_stderr(io.StringIO()):
                        result = common.run_profiler(
                            base, tmp, 1, False, False, "run", 1, "x", 1, 0,
                            framework="auto", profile_workload="legacy", warmup_steps=1,
                            auth=auth, record_shapes=None if framework == "trtllm" else True,
                        )
                    self.assertTrue(common.discover_trace_files(result, recursive=True))
                    self.assertIn(("/start_profile", "Bearer control-key"), seen)
                    self.assertIn(("/stop_profile", "Bearer control-key"), seen)
                    self.assertGreaterEqual(sum(path in {"/generate", "/v1/chat/completions"} for path, _ in seen), 2)
                    if framework != "trtllm":
                        options = active.get("profiler_kwargs", active)
                        self.assertTrue(options["record_shapes"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()

    def test_sglang_shapes_default_and_failed_probe_still_authenticates_stop(self):
        auth = common.ProfilerAuth("api", "admin")
        for shapes in (None, False, True):
            with self.subTest(shapes=shapes), tempfile.TemporaryDirectory() as tmp:
                with (
                    mock.patch.object(common, "try_get_json", return_value={}),
                    mock.patch.object(common.request, "urlopen") as http,
                    mock.patch.object(common, "send_probe_requests", side_effect=error.HTTPError("http://server", 500, "failed", {}, None)),
                    mock.patch.object(common, "stop_remote_profiler") as stop,
                ):
                    with self.assertRaises(error.HTTPError):
                        common.run_sglang_profiler(
                            "http://server", tmp, 1, False, False, "run",
                            common.ProbePlan("x", 1, 1, 1, 0), 0,
                            auth=auth, record_shapes=shapes,
                        )
                    req = http.call_args.args[0]
                    self.assertEqual(req.get_header("Authorization"), "Bearer admin")
                    payload = json.loads(req.data)
                    if shapes is None:
                        self.assertNotIn("record_shapes", payload)
                    else:
                        self.assertIs(payload["record_shapes"], shapes)
                    stop.assert_called_once_with("http://server", "sglang", auth=auth)
