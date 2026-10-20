"""주입한 실행 환경의 앱 격리, HTTP/SSE 계약, 자원 정리를 검증한다."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent import runtime
from agent.fake_llm import FastLLM, QualityLLM
from agent.session_store import SqliteSessionStore


class StubMCP:
    """외부 프로세스 대신 수명 기록과 계획용 프롬프트를 제공한다."""

    def __init__(self, *, stop_error=False):
        self.lifecycle = []
        self.stop_error = stop_error

    async def start(self):
        self.lifecycle.append(("start", asyncio.current_task()))

    async def stop(self):
        self.lifecycle.append(("stop", asyncio.current_task()))
        if self.stop_error:
            raise RuntimeError("종료 실패")

    async def find_prompt(self, name):
        assert name == "incident_analysis_prompt"
        return "ops"

    async def get_prompt(self, name, arguments, server):
        assert name == "incident_analysis_prompt"
        assert server == "ops"
        return "상태를 조회하고 표준 절차에 따라 진단하세요."


def fake_llms():
    return "fake", FastLLM("test-fast", 0.00015), QualityLLM("test-quality", 0.005)


def make_test_app(version="v2"):
    import server
    from agent.deps import SupervisorDeps
    from agent.evidence import CollectionResult, Evidence
    from agent.graph import build_graph

    class FixedCollector:
        async def collect(self, tasks):
            return CollectionResult([
                Evidence(
                    index=task.index, source="monitoring" if task.kind == "monitoring" else "runbook",
                    data={
                        "system_name": task.target, "status": "healthy",
                        "cpu_percent": 34.0, "memory_percent": 48.0,
                        "last_checked": "2026-10-20T00:00:00", "alerts": [],
                    } if task.kind == "monitoring" else "# 대응 절차\n1. 확인\n2. 조치\n3. 재확인",
                    retries=0,
                ) for task in tasks
            ], elapsed=0.25)

    store = SqliteSessionStore(":memory:")
    backend, fast, quality = fake_llms()
    deps = SupervisorDeps(
        fast_llm=fast, quality_llm=quality, mcp=StubMCP(),
        session_store=store, version=version, evidence_collector=FixedCollector(),
    )
    calls = []

    @asynccontextmanager
    async def factory():
        calls.append("start")
        try:
            yield runtime.AgentRuntime(
                deps=deps, graph=build_graph(deps) if version == "v2" else None,
                llm_backend=backend,
            )
        finally:
            calls.append("stop")
            store.close()

    return server.create_app(runtime_factory=factory), store, calls


def decode_sse(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    return [json.loads(frame) for frame in frames[:-1]]


def test_injected_apps_keep_history_and_health_isolated():
    first, _, first_calls = make_test_app()
    second, _, second_calls = make_test_app()
    with TestClient(first) as a, TestClient(second) as b:
        h1, h2 = a.get("/health").json(), b.get("/health").json()
        assert h1["creation_count"] == h2["creation_count"] == 1
        assert h1["instance_id"] != h2["instance_id"]
        assert h1["fast_llm"] == "test-fast"
        assert a.get("/version").json() == {"supervisor_version": "v2"}
        a.post("/chat", json={"message": "제 이름은 민수입니다", "thread_id": "same"})
        remembered = a.post("/chat", json={"message": "제 이름이 뭐예요?", "thread_id": "same"}).json()
        other = b.post("/chat", json={"message": "제 이름이 뭐예요?", "thread_id": "same"}).json()
        assert "민수님" in remembered["reply"]
        assert "민수" not in other["reply"]
        assert remembered["metadata"]["instance_id"] == h1["instance_id"]
        assert remembered["metadata"]["route"] == "chat"
        assert set(remembered) == {"reply", "events", "metadata"}
        assert first_calls == second_calls == ["start"]
    assert first_calls == second_calls == ["start", "stop"]


def test_incident_stream_preserves_events_final_answer_and_history():
    app, store, _ = make_test_app()
    with TestClient(app) as client:
        events = decode_sse(client.get("/chat/stream", params={
            "message": "web-01 장애 진단", "thread_id": "incident",
        }))
        types = [event["type"] for event in events]
        assert types == [
            "RouteDecided", "PlanCreated", "GateApplied", "TaskCompleted",
            "ReplanCheck", "TaskCompleted", "ReplanCheck", "AnalysisSkipped",
            "AnswerToken", "DraftWritten", "Validation",
            "AnswerToken", "DraftWritten", "Validation", "Finalize", "FinalAnswer",
        ]
        final = events[-1]
        assert set(final) == {"type", "reply", "prompt"}
        assert "web-01 상태=healthy" in final["reply"]
        assert "대응 절차" in final["reply"]
        assert [e["attempt"] for e in events if e["type"] == "AnswerToken"] == [1, 2]
        assert store.load("incident") == [
            {"role": "user", "content": "web-01 장애 진단"},
            {"role": "assistant", "content": final["reply"]},
        ]
        normal = client.post("/chat", json={"message": "web-01 장애 진단"}).json()
        assert normal["reply"] == final["reply"]
        assert normal["metadata"]["validation_attempts"] == 2
        assert normal["metadata"]["parallel_elapsed"] == 0.25
        assert normal["metadata"]["retry_count"] == 0


def test_direct_stream_saves_history_and_legacy_routes_keep_their_contract():
    app, store, _ = make_test_app()
    with TestClient(app) as client:
        events = decode_sse(client.get("/chat/stream", params={"message": "제 이름은 민수입니다"}))
        assert [e["type"] for e in events] == [
            "RouteDecided", "AnswerToken", "Finalize", "FinalAnswer",
        ]
        assert store.load("default")[-1]["content"] == events[-1]["reply"]
    legacy, store, _ = make_test_app("v1")
    with TestClient(legacy) as client:
        response = client.post("/chat", json={"message": "안녕하세요"}).json()
        assert response["metadata"]["supervisor_version"] == "v1"
        assert response["metadata"]["route"] == "legacy"
        events = decode_sse(client.get("/chat/stream", params={"message": "안녕하세요"}))
        assert [e["type"] for e in events] == ["Legacy", "Finalize", "FinalAnswer"]
        assert events[-1]["reply"] == response["reply"]
        # v1 SSE의 기존 저장 동작은 이번 구조 변경에서 바꾸지 않는다.
        assert len(store.load("default")) == 2


@pytest.mark.parametrize("failure", ["none", "models", "graph", "body", "stop"])
def test_runtime_cleans_resources_on_success_and_failure(monkeypatch, failure):
    mcp = StubMCP(stop_error=failure == "stop")
    closed = []
    store = SimpleNamespace(close=lambda: closed.append(True))

    def models():
        if failure == "models":
            raise RuntimeError("모델 조립 실패")
        return fake_llms()

    def graph(deps):
        if failure == "graph":
            raise RuntimeError("그래프 조립 실패")
        return object()

    monkeypatch.setenv("SUPERVISOR_VERSION", "v2")
    monkeypatch.setattr(runtime, "build_graph", graph)

    async def exercise():
        async with runtime.create_runtime(
            mcp_factory=lambda: mcp,
            llm_factory=models,
            session_store_factory=lambda: store,
            retriever_factory=None,
        ) as bundle:
            assert bundle.deps.mcp is mcp
            assert bundle.deps.session_store is store
            assert bundle.llm_backend == "fake"
            if failure == "body":
                raise RuntimeError("실행 실패")

    if failure == "none":
        asyncio.run(exercise())
    else:
        with pytest.raises(RuntimeError):
            asyncio.run(exercise())
    assert [event for event, task in mcp.lifecycle] == ["start", "stop"]
    assert mcp.lifecycle[0][1] is mcp.lifecycle[1][1]
    assert closed == ([] if failure == "models" else [True])


def test_chat_trace_keeps_original_inputs_and_node_parentage(monkeypatch):
    from langsmith import Client, get_current_run_tree, traceable, tracing_context
    import server

    # LangGraph의 tracer는 local 문맥에서도 SDK 전송 메서드를 호출한다.
    # 추적 트리 생성은 실제로 실행하고 외부 전송 지점만 대체한다.
    monkeypatch.setattr(Client, "create_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(Client, "update_run", lambda *args, **kwargs: None)
    app, _, _ = make_test_app()
    captured = []

    @traceable(name="test_root")
    async def invoke():
        parent = get_current_run_tree()
        result, run_id = await server._handle_chat(app, "안녕하세요", "trace-thread")
        captured.extend(parent.child_runs)
        return result, run_id

    client = Client(
        api_url="http://127.0.0.1:1", api_key="offline-test-key",
        auto_batch_tracing=False, info={"version": "0"},
    )
    try:
        with TestClient(app), tracing_context(enabled="local", client=client):
            result, run_id = asyncio.run(invoke())
        chat_run = next(run for run in captured if run.name == "chat_request")
        assert chat_run.inputs == {"message": "안녕하세요", "thread_id": "trace-thread"}
        assert str(chat_run.id) == run_id
        graph_run = next(run for run in chat_run.child_runs if run.name == "LangGraph")
        assert {"route", "chat_answer"} <= {run.name for run in graph_run.child_runs}
        assert result["final"] == "안녕하세요! 무엇을 도와드릴까요?"
    finally:
        client.close()


def test_real_runtime_serves_chat_and_stream_and_closes_mcp(tmp_path, monkeypatch):
    from functools import partial
    import server
    monkeypatch.setenv("SUPERVISOR_VERSION", "v2")
    app = server.create_app(runtime_factory=partial(
        runtime.create_runtime, llm_factory=fake_llms,
        session_store_factory=lambda: SqliteSessionStore(tmp_path / "session.db"),
        retriever_factory=None,
    ))
    with TestClient(app) as client:
        mcp = app.state.deps.mcp
        first_id = client.get("/health").json()["instance_id"]
        response = client.post("/chat", json={"message": "payments-service CPU 급증 진단"}).json()
        events = decode_sse(client.get("/chat/stream", params={
            "message": "payments-service CPU 급증 진단", "thread_id": "stream",
        }))
        assert "payments-service 상태=degraded" in response["reply"]
        assert response["reply"] == events[-1]["reply"]
        assert response["metadata"]["validation_attempts"] == 2
        assert response["metadata"]["retry_count"] == 0
        assert client.get("/health").json()["instance_id"] == first_id
        assert client.get("/health").json()["creation_count"] == 1
    assert mcp.sessions == {}


def test_rag_documents_reach_http_and_sse_with_sources():
    from agent.rag import Document, SearchResult
    class Retriever:
        async def search(self, query, observations):
            assert observations[0]["system_name"] == "web-01"
            return SearchResult([
                Document("cpu-spike", "runbook://incident-response/cpu-spike",
                         "# CPU\n1. CPU 확인\n2. 부하 분산"),
                Document("memory-leak", "runbook://incident-response/memory-leak",
                         "# 메모리\n1. 추세 확인\n2. 힙 조사"),
            ], [], "복합 증상 근거", {})
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        app.state.deps.runbook_retriever = Retriever()
        response = client.post("/chat", json={"message": "web-01 장애 진단"}).json()
        assert "runbook://incident-response/cpu-spike" in response["reply"]
        assert "runbook://incident-response/memory-leak" in response["reply"]
        events = decode_sse(client.get("/chat/stream", params={"message": "web-01 장애 진단"}))
        retrieved = next(e for e in events if e["type"] == "RunbookRetrieved")
        assert retrieved["document_ids"] == ["cpu-spike", "memory-leak"]
        assert events[-1]["reply"] == response["reply"]


def test_rag_empty_context_is_not_replaced_with_legacy_runbook():
    from agent.rag import SearchResult
    class Retriever:
        async def search(self, query, observations):
            return SearchResult([], [], "해당 절차 없음", {})
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        app.state.deps.runbook_retriever = Retriever()
        events = decode_sse(client.get("/chat/stream", params={"message": "web-01 장애 진단"}))
        assert "런북 근거를 찾지 못했습니다" in events[-1]["reply"]
        assert "재시작" not in events[-1]["reply"]
        assert not any(e["type"] == "TaskCompleted" and e["source"] == "runbook" for e in events)
        assert len([e for e in events if e["type"] == "DraftWritten"]) == 1


def test_rag_index_failure_cleans_mcp_and_store(monkeypatch):
    mcp, closed = StubMCP(), []
    store = SimpleNamespace(close=lambda: closed.append(True))
    async def broken_index(client):
        assert client is mcp
        raise TimeoutError("임베딩 연결 실패")
    monkeypatch.setenv("SUPERVISOR_VERSION", "v2")
    async def exercise():
        async with runtime.create_runtime(
            mcp_factory=lambda: mcp, llm_factory=fake_llms,
            session_store_factory=lambda: store, retriever_factory=broken_index,
        ):
            pytest.fail("색인 실패 후 실행되면 안 됩니다")
    with pytest.raises(TimeoutError):
        asyncio.run(exercise())
    assert closed == [True]
    assert [event for event, _ in mcp.lifecycle] == ["start", "stop"]
