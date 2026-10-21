from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from evaluation.verify_traces import verify_records


class TraceClient:
    def __init__(self, reply="answer"):
        self.reply = reply

    def read_run(self, run_id):
        start = datetime(2026, 10, 21, tzinfo=timezone.utc)
        return SimpleNamespace(id=run_id, inputs={"message": "question", "thread_id": "s"},
                               outputs={"output": [{"final": self.reply}, run_id]},
                               start_time=start, end_time=start + timedelta(seconds=1),
                               total_tokens=12, total_cost="0.0001", error=None)

    def get_run_url(self, *, run):
        return "https://smith.test/r/" + run.id


def record():
    return {"case_id": "C01", "turn": 1, "repetition": 1, "source_run_id": "source", "status": "ok",
            "message": "question", "thread_id": "s", "response": {"reply": "answer"},
            "model_usage": [{"provider": "openai", "usage": {"total_tokens": 12}}]}


def test_readback_checks_inputs_outputs_and_preserves_estimate_semantics():
    rows, report = verify_records([record()], TraceClient())
    assert report['verified_requests'] == 1
    assert rows[0]['trace_elapsed_ms'] == 1000
    assert rows[0]['langsmith_estimated_cost_usd'] == "0.0001"
    assert rows[0]['source_run_url'].endswith('/source')
    assert report['actual_billed_usd'] is None


def test_remote_wrong_reply_or_usage_is_not_verified():
    with pytest.raises(ValueError, match='output'):
        verify_records([record()], TraceClient(reply="different"))
    r = record()
    r['model_usage'][0]['usage']['total_tokens'] = 13
    with pytest.raises(ValueError, match='tokens'):
        verify_records([r], TraceClient())


def test_missing_source_run_is_not_silently_omitted():
    r = record()
    r['source_run_id'] = None
    with pytest.raises(ValueError, match='source'):
        verify_records([r], TraceClient())


@pytest.mark.parametrize('remote_reply', ['answer', 'different'])
def test_cli_closes_sdk_client_without_context_manager(tmp_path, monkeypatch, remote_reply):
    import dotenv
    import langsmith
    from evaluation import verify_traces

    client = TraceClient(reply=remote_reply)
    closed = []
    client.close = lambda: closed.append(True)
    monkeypatch.setattr(langsmith, 'Client', lambda **kwargs: client)
    monkeypatch.setattr(dotenv, 'dotenv_values', lambda path: {})
    monkeypatch.setenv('LANGSMITH_API_KEY', 'offline-test-key')
    monkeypatch.setattr(verify_traces, 'load_bundle', lambda path: {
        'manifest': {'phase': 'after'}, 'records': [record()]})

    if remote_reply == 'answer':
        assert verify_traces.main(['--after', str(tmp_path)]) == 0
        assert (tmp_path / 'remote-verification.json').is_file()
    else:
        with pytest.raises(ValueError, match='output'):
            verify_traces.main(['--after', str(tmp_path)])
        assert not (tmp_path / 'remote-verification.json').exists()
    assert closed == [True]
