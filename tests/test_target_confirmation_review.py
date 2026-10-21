"""대상 확인 검수에서 재현한 오인·질문 유실·동시 완료·불필요한 파일 읽기 회귀 검사."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_app_runtime import make_test_app
from test_target_confirmation import monitored, send


@pytest.mark.parametrize("query", [
    "payments-service는 빼 주세요. cache-service 장애 진단",
    "payments-service 조사는 하지 말고 cache-service 장애 진단",
    "payments-service의 조사를 하지 말고 cache-service 장애 진단",
    "payments-service는 확인하지 말아 주세요. cache-service 장애 진단",
    "Don't investigate payments-service; cache-service 장애 진단",
])
def test_exclusion_variants_stop_before_any_monitoring(query):
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        result = send(client, query, stream=True)
        assert not monitored(result)
        assert any(e["type"] == "TargetConfirmationRequired" for e in result["events"])


@pytest.mark.parametrize("query", [
    "payments-service에서 ERR_CONNECTION_RESET 오류 발생 원인 진단",
    "payments-service의 p95-latency가 높아요. 원인 진단",
    "payments-service의 p99_latency_ms가 높아요. 원인 진단",
    "payments-service 장애 원인은 아직 not confirmed 상태입니다. 진단해주세요",
    "payments-service와 cache-service는 비교하지 말고 각각 장애 진단",
    "log-service 로그 서비스 장애 진단",
])
def test_diagnostic_terms_do_not_become_unknown_systems_or_exclusions(query):
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        result = send(client, query)
        expected = (["payments-service", "cache-service"] if "cache-service" in query
                    else ["log-service"] if query.startswith("log-service")
                    else ["payments-service"])
        assert monitored(result) == expected
        assert not any(e["type"] == "TargetConfirmationRequired" for e in result["events"])


@pytest.mark.parametrize("query", [
    "payments-service와 payment_service 장애 진단",
    "payments-service와 ERR_CONNECTION_RESET-service 장애 진단",
    "payments-service와 unknown-service 장애 진단",
    "payments-service-old 장애 진단",
])
def test_error_code_exceptions_do_not_hide_unknown_systems(query):
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        assert not monitored(send(client, query))


@pytest.mark.parametrize("reply", [
    "네, payments-service요",
    "payments-service로 해 주세요",
    "payments-service를 확인해 주세요",
    "네, payments-service로 진단해 주세요.",
])
@pytest.mark.parametrize("stream", [False, True])
def test_natural_selection_replies_keep_original_symptoms(reply, stream):
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 원인 진단해주세요", stream=not stream)
        result = send(client, reply, stream=stream)
        assert monitored(result) == ["payments-service"]
        assert "CPU 급증" in result["reply"]


@pytest.mark.parametrize("reply", [
    "아마 payments-service일 거예요",
    "네, payment_service요",
    "payments-service 말고요",
])
def test_uncertain_selection_keeps_original_question(reply):
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 원인 진단해주세요")
        result = send(client, reply)
        assert not monitored(result)
        assert any(e["type"] == "TargetConfirmationRequired" for e in result["events"])
        corrected = send(client, "payments-service")
        assert monitored(corrected) == ["payments-service"]
        assert "CPU 급증" in corrected["reply"]


def test_new_diagnosis_after_confirmation_does_not_reuse_old_symptom():
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 진단")
        result = send(client, "payments-service 메모리 장애를 새로 진단해주세요")
        assert monitored(result) == ["payments-service"]
        assert "메모리 장애" in result["reply"]
        assert "CPU 급증" not in result["reply"]


def test_same_text_new_confirmation_survives_old_completion():
    app, store, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 진단")
        original = app.state.deps.quality_llm.write

        async def replacement(*args, **kwargs):
            store.clear_pending_target("targets")
            store.set_pending_target("targets", "CPU 급증 진단")
            return await original(*args, **kwargs)

        app.state.deps.quality_llm.write = replacement
        send(client, "payments-service")
        assert store.pending_target("targets") is not None
        app.state.deps.quality_llm.write = original
        result = send(client, "cache-service")
        assert monitored(result) == ["cache-service"]
        assert "CPU 급증" in result["reply"]
        assert store.pending_target("targets") is None


def test_old_database_migration_preserves_pending_and_history(tmp_path):
    import sqlite3
    from agent.session_store import SqliteSessionStore

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE messages (thread_id TEXT, role TEXT, content TEXT, created TEXT)")
        conn.execute("INSERT INTO messages VALUES ('same', 'user', 'CPU 급증 진단', '2026-10-20')")
        conn.execute("CREATE TABLE pending_targets (thread_id TEXT PRIMARY KEY, query TEXT NOT NULL)")
        conn.execute("INSERT INTO pending_targets VALUES ('same', 'CPU 급증 진단')")
    store = SqliteSessionStore(path)
    try:
        original = store.pending_target("same")
        assert original.query == "CPU 급증 진단"
        assert original.request_id
        assert store.load("same") == [{"role": "user", "content": "CPU 급증 진단"}]
    finally:
        store.close()
    reopened = SqliteSessionStore(path)
    try:
        assert reopened.pending_target("same") == original
        reopened.set_pending_target("same", "CPU 급증 진단")
        replacement = reopened.pending_target("same")
        assert replacement.request_id != original.request_id
        reopened.clear_pending_target("same", expected_request_id=original.request_id)
        assert reopened.pending_target("same") == replacement
        reopened.clear_pending_target("same", expected_request_id=replacement.request_id)
        assert reopened.pending_target("same") is None
    finally:
        reopened.close()


def test_chat_requests_do_not_reread_runbook_files(monkeypatch):
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        original = Path.read_text
        reads = []

        def forbid_runbook_read(path, *args, **kwargs):
            if path.parent.name == "runbooks":
                reads.append(path)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", forbid_runbook_read)
        for query in ["안녕하세요", "CPU 급증 진단", "payments-service"]:
            send(client, query)
        assert reads == []


def test_pure_runbook_uses_same_loaded_titles_without_disk_read(monkeypatch):
    from agent.rag import Document, SearchResult

    class Retriever:
        documents = {"custom": Document("custom", "runbook://custom",
                                       "# Runbook: 사용자 지정 대응 절차\n1. 확인\n2. 조치")}

        async def search(self, query, observations):
            assert not observations
            return SearchResult(list(self.documents.values()), [], "조회", {})

    from agent.graph import build_graph
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        app.state.deps.runbook_retriever = Retriever()
        app.state.graph = build_graph(app.state.deps)
        original = Path.read_text
        reads = []
        def counted(path, *args, **kwargs):
            reads.append(path)
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", counted)
        result = send(client, "사용자 지정 대응 절차를 찾아줘", stream=True)
        assert "runbook://custom" in result["reply"]
        assert not monitored(result)
        assert reads == []
