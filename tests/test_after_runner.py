import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.contracts import write_json
from evaluation.run_after import collect, validate_health, prepare_output, snapshot_sources, measure_http


def plan():
    return [
        {"sequence": 1, "case_id": "C08", "turn": 1, "repetition": 1, "thread_id": "same", "message": "이름은 민수"},
        {"sequence": 2, "case_id": "C08", "turn": 2, "repetition": 1, "thread_id": "same", "message": "내 이름은?"},
        {"sequence": 3, "case_id": "C09", "turn": 1, "repetition": 1, "thread_id": "other", "message": "안녕"}]


def test_failed_request_stops_new_calls_but_preserves_full_plan(tmp_path):
    called = []
    def failed(item):
        called.append(item)
        return {**item, "status": "error", "error": "TimeoutError", "response": None}
    rows = collect(plan(), tmp_path, failed)
    assert len(called) == 1
    assert [r["status"] for r in rows] == ["error", "not_run", "not_run"]
    assert len((tmp_path / "records.jsonl").read_text().splitlines()) == 3
    assert rows[1]["model_usage"] is None


def test_successful_two_turn_requests_preserve_session_and_order(tmp_path):
    observed = []
    def measure(item):
        observed.append((item["thread_id"], item["message"]))
        return {**item, "status": "ok", "response": {"reply": "ok"}}
    collect(plan(), tmp_path, measure)
    assert observed == [("same", "이름은 민수"), ("same", "내 이름은?"), ("other", "안녕")]


def test_output_never_reuses_existing_directory_or_before_child(tmp_path):
    before = tmp_path / "before"
    before.mkdir()
    with pytest.raises(ValueError):
        prepare_output(before / "new", before)
    with pytest.raises(FileExistsError):
        prepare_output(before, tmp_path / "different")


def test_snapshot_excludes_env_database_and_uses_current_uncommitted_sources(tmp_path):
    root, dest = tmp_path / "repo", tmp_path / "snapshot"
    (root / "agent").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "agent/changed.py").write_text("# current uncommitted")
    (root / "server.py").write_text("# server")
    for name in ("mcp_server.py", "mcp_server_docs.py"):
        (root / name).write_text("# mcp")
    (root / ".env").write_text("SECRET=not-copied")
    (root / "session.db").write_text("private")
    hashes = snapshot_sources(root, dest)
    assert (dest / "agent/changed.py").read_text() == "# current uncommitted"
    assert not (dest / ".env").exists()
    assert not (dest / "session.db").exists()
    assert "agent/changed.py" in hashes


def test_health_rejects_wrong_backend_model_or_version():
    m = {"requested_fast_model": "gpt-4o-mini", "requested_quality_model": "gpt-4o"}
    good = {"llm_backend": "openai", "supervisor_version": "v2", "fast_llm": "openai:gpt-4o-mini",
            "quality_llm": "openai:gpt-4o"}
    validate_health(good, m)
    for k, v in [("llm_backend", "fake"), ("quality_llm", "openai:wrong"), ("supervisor_version", "v1")]:
        with pytest.raises(ValueError):
            validate_health({**good, k: v}, m)


def test_health_checks_separately_configured_validator():
    manifest = {'requested_fast_model': 'gpt-4o-mini', 'requested_quality_model': 'gpt-4o',
                'requested_validator_model': 'review-model'}
    health = {'llm_backend': 'openai', 'supervisor_version': 'v2',
              'fast_llm': 'openai:gpt-4o-mini', 'quality_llm': 'openai:gpt-4o',
              'validator_llm': 'openai:review-model'}
    validate_health(health, manifest)
    with pytest.raises(ValueError, match='model'):
        validate_health({**health, 'validator_llm': 'openai:gpt-4o'}, manifest)


def test_invalid_http_response_is_preserved_as_error_without_leaking_body(monkeypatch):
    import io
    monkeypatch.setattr('evaluation.run_after.request.urlopen', lambda *a, **kw: io.BytesIO(b'{"secret": "private"}'))
    r = measure_http(plan()[0], 'http://localhost:1', 1)
    assert r['status'] == 'error'
    assert r['response'] is None
    assert r['error'] == 'ValueError'
    assert 'private' not in str(r)
    assert r['http_elapsed_ms'] >= 0


def test_measure_http_preserves_response_and_source_trace(monkeypatch):
    import io
    payload = {'reply': 'ok', 'events': [], 'metadata': {'langsmith_run_id': 'trace'}}
    monkeypatch.setattr('evaluation.run_after.request.urlopen', lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    r = measure_http(plan()[0], 'http://localhost:1', 1)
    assert r['status'] == 'ok'
    assert r['source_run_id'] == 'trace'
    assert r['response'] == payload
