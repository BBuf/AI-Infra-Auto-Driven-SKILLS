import importlib.util
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'skills/sglang-prod-incident-triage/scripts/incident_artifact_tool.py'
SPEC = importlib.util.spec_from_file_location('incident_artifact_tool', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bundle_uses_all_dp_ranks_and_distinguishes_ready(tmp_path):
    (tmp_path / 'health.txt').write_text('ok')
    (tmp_path / 'ready.txt.error.json').write_text(json.dumps({'status': 503}))
    (tmp_path / 'metrics.txt.error.json').write_text(json.dumps({'status': 404}))
    (tmp_path / 'server_info.json').write_text(json.dumps({'enable_metrics': False, 'speculative_algorithm': 'EAGLE', 'internal_states': [{'avg_spec_accept_length': 1}]}))
    (tmp_path / 'loads_all.json').write_text(json.dumps({'loads': [
        {'dp_rank': 0, 'num_running_reqs': 3, 'num_waiting_reqs': 2, 'token_usage': .2, 'cache_hit_rate': .9},
        {'dp_rank': 1, 'num_running_reqs': 7, 'num_waiting_reqs': 5, 'token_usage': .95, 'cache_hit_rate': .1},
    ]}))
    result = MODULE.build_bundle_summary(tmp_path)
    assert result['point_in_time_load']['running_reqs'] == 10
    assert result['point_in_time_load']['waiting_reqs'] == 7
    assert result['point_in_time_load']['token_usage'] == .95
    assert [row['cache_hit_rate'] for row in result['point_in_time_load']['per_rank']] == [.9, .1]
    assert any('paused/draining' in signal for signal in result['signals'])
    assert any('metrics disabled' in signal for signal in result['signals'])
    assert result['speculative_accept_lengths'] == [1]


def test_health_timeout_does_not_claim_server_unhealthy(tmp_path):
    (tmp_path / 'health.txt.error.json').write_text(json.dumps({'status': -1, 'error': 'TimeoutError'}))
    result = MODULE.build_bundle_summary(tmp_path)
    assert any('no server health verdict' in signal for signal in result['signals'])
    assert not any('global unhealthy' in signal for signal in result['signals'])


def test_collect_health_waits_for_server_verdict(tmp_path, monkeypatch):
    calls = {}
    def fake_request(base_url, path, token, parse_json, timeout):
        calls[path] = timeout
        return {'ok': True, 'status': 200, 'text': '', 'json': {}}
    monkeypatch.setattr(MODULE, 'request_endpoint', fake_request)
    MODULE.collect_bundle('http://localhost:30000', None, str(tmp_path), 10)
    assert calls['/health'] >= 25
    assert calls['/health_generate'] >= 25
    assert calls['/ready'] == 10


def test_dump_resolved_config_fallback(tmp_path):
    dump = tmp_path / 'dump.pkl'
    dump.write_bytes(pickle.dumps({'server_args': None, 'resolved_config': {'model_path': 'public/model', 'tp_size': 2}, 'config_updates': {'dtype': 'bf16'}, 'requests': []}))
    result = MODULE.summarize_dump_file(dump, 10, 40)
    assert 'public/model' in result
    assert 'Config updates:' in result
