import asyncio
import json

import httpx
import pytest

from evaluation.recording_server import UsageRecorder, instrument_handler


def test_sync_and_async_provider_usage_are_separated_from_request_content(tmp_path):
    recorder = UsageRecorder(tmp_path)
    token = recorder.current.set({"case_id": "C01", "turn": 1, "repetition": 1})
    try:
        body = {"model": "gpt-4o-2024", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
        response = httpx.Response(200, json=body)
        def post(url, **kwargs):
            return response
        wrapped = recorder.wrap_sync(post)
        assert wrapped("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o"}, headers={"Authorization": "secret"}) is response
        async def apost(url, **kwargs):
            return httpx.Response(200, json={"model": "qwen3:8b", "prompt_eval_count": 20, "eval_count": 3})
        asyncio.run(recorder.wrap_async(apost)("http://localhost:11434/api/chat", json={"model": "qwen3:8b"}))
    finally:
        recorder.current.reset(token)
    text = (tmp_path / "model-usage.jsonl").read_text()
    rows = [json.loads(s) for s in text.splitlines()]
    assert [r["usage"]["total_tokens"] for r in rows] == [12, 23]
    assert [r["provider"] for r in rows] == ["openai", "ollama"]
    assert "secret" not in text and "Authorization" not in text


def test_network_failure_is_recorded_without_fabricated_zero_tokens(tmp_path):
    recorder = UsageRecorder(tmp_path)
    token = recorder.current.set({"case_id": "C01", "turn": 1, "repetition": 1})
    def fail(url, **kwargs):
        raise httpx.ConnectError("secret details")
    try:
        with pytest.raises(httpx.ConnectError):
            recorder.wrap_sync(fail)("https://api.openai.com/v1/chat/completions", json={"model": "gpt-4o"})
    finally:
        recorder.current.reset(token)
    row = json.loads((tmp_path / "model-usage.jsonl").read_text())
    assert row["usage"] is None
    assert row["error"] == "ConnectError"
    assert "secret" not in str(row)


def test_unrelated_http_and_index_startup_are_not_model_calls(tmp_path):
    recorder = UsageRecorder(tmp_path)
    recorder.wrap_sync(lambda *a, **kw: httpx.Response(200))("https://api.openai.com/v1/chat/completions")
    token = recorder.current.set({"case_id": "C01", "turn": 1, "repetition": 1})
    try:
        recorder.wrap_sync(lambda *a, **kw: httpx.Response(200))("https://example.com")
    finally:
        recorder.current.reset(token)
    assert not (tmp_path / "model-usage.jsonl").exists()


def test_recorded_handler_preserves_response_and_records_failure_state(tmp_path):
    recorder = UsageRecorder(tmp_path)
    plan = [{"case_id": "C01", "turn": 1, "repetition": 1, "thread_id": "s", "message": "question"}]
    async def original(app, message, thread_id, **kwargs):
        assert recorder.current.get()["case_id"] == "C01"
        raise RuntimeError("internal")
    handler = instrument_handler(original, recorder, plan, {"run_id": "r", "dataset_sha256": "h"})
    with pytest.raises(RuntimeError):
        asyncio.run(handler(None, "question", "s"))
    row = json.loads((tmp_path / "evidence.jsonl").read_text())
    assert row["error"] == "RuntimeError"
    assert recorder.current.get() is None


def test_real_http_app_records_state_and_trace_without_external_provider(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from langsmith import Client, tracing_context
    import server
    from test_app_runtime import make_test_app
    app, _, _ = make_test_app()
    recorder = UsageRecorder(tmp_path)
    plan = [{"case_id": "C07", "turn": 1, "repetition": 1, "thread_id": "recorded", "message": "안녕"}]
    monkeypatch.setattr(server, "_handle_chat", instrument_handler(server._handle_chat, recorder, plan,
                                                                 {"run_id": "offline", "dataset_sha256": "h"}))
    monkeypatch.setattr(Client, "create_run", lambda *a, **kw: None)
    monkeypatch.setattr(Client, "update_run", lambda *a, **kw: None)
    trace_client = Client(api_url="http://127.0.0.1:1", api_key="offline-test-key",
                          auto_batch_tracing=False, info={"version": "0"})
    try:
        with tracing_context(enabled="local", client=trace_client):
            with TestClient(app) as client:
                result = client.post('/chat', json={"message": "안녕", "thread_id": "recorded"})
    finally:
        trace_client.close()
    assert result.status_code == 200
    row = json.loads((tmp_path / "evidence.jsonl").read_text())
    assert row["state"]["final"] == result.json()["reply"]
    assert row["source_run_id"] == result.json()["metadata"]["langsmith_run_id"]
    assert row["trace_elapsed_ms"] > 0
