"""대상이 확정되기 전에는 조사하지 않고, 확정 후 원래 증상을 이어서 진단한다."""
import json

import pytest
from fastapi.testclient import TestClient

from test_app_runtime import decode_sse, make_test_app


def send(client, message, thread="targets", stream=False):
    if stream:
        events = decode_sse(client.get("/chat/stream", params={
            "message": message, "thread_id": thread,
        }))
        return {"reply": events[-1]["reply"], "events": events[:-1]}
    return client.post("/chat", json={"message": message, "thread_id": thread}).json()


def monitored(response):
    return [e["target"] for e in response["events"]
            if e["type"] == "TaskCompleted" and e["source"] == "monitoring"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("question", [
    "CPU 급증 원인 진단해주세요",
    "payments-service-old 장애 진단",
    "payment_service 장애 진단",
    "payments-service와 unknown-service 장애 진단",
    "결제 서비스 장애 진단",
    "payments-service 말고 cache-service 진단",
    "그 서비스 장애 진단",
    "payments-service가 아닌 cache-service 장애 진단",
    "payments-service는 정상, 나머지는 장애 진단",
    "my.payments-service 장애 진단",
])
def test_uncertain_target_stops_before_planning_and_research(question, stream):
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        result = send(client, question, stream=stream)
        assert not monitored(result)
        assert not any(e["type"] in {"PlanCreated", "RunbookRetrieved", "DraftWritten"}
                       for e in result["events"])
        event = next(e for e in result["events"] if e["type"] == "TargetConfirmationRequired")
        assert "payments-service" in event["choices"]
        assert "시스템" in result["reply"]


@pytest.mark.parametrize("stream", [False, True])
def test_selection_resumes_original_symptom_and_does_not_add_other_systems(stream):
    app, store, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 원인 진단해주세요", stream=stream)
        result = send(client, "payments-service", stream=not stream)
        assert monitored(result) == ["payments-service"]
        assert "CPU 급증" in result["reply"]
        assert store.load("targets")[-2]["content"] == "payments-service"
        # 이전 대상을 다음의 대상 없는 질문에 암묵적으로 재사용하지 않는다.
        again = send(client, "메모리 장애 진단", stream=stream)
        assert not monitored(again)
        assert any(e["type"] == "TargetConfirmationRequired" for e in again["events"])


def test_unknown_original_name_does_not_override_explicit_correction():
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "payments-service-old CPU 급증 진단")
        result = send(client, "payments-service로 확인해줘")
        assert monitored(result) == ["payments-service"]
        assert "CPU 급증" in result["reply"]


def test_confirmed_multiple_targets_override_incomplete_or_unrequested_model_plan():
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        app.state.deps.fast_llm.plan = lambda *args: ({
            "needs_analysis": False,
            "tasks": [{"id": "wrong", "agent": "research", "kind": "monitoring",
                       "target": "auth-service", "status": "pending"}],
        }, {"usd": 0.0})
        result = send(client, "gateway-service와 log-service 장애 진단")
        assert monitored(result) == ["gateway-service", "log-service"]
        plan = next(e for e in result["events"] if e["type"] == "PlanCreated")
        assert plan["targets"] == ["gateway-service", "log-service"]


def test_exact_name_with_its_korean_description_does_not_need_confirmation():
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        result = send(client, "payments-service 결제 서비스의 CPU 급증 진단")
        assert monitored(result) == ["payments-service"]
        assert not any(e["type"] == "TargetConfirmationRequired" for e in result["events"])


def test_cancel_new_question_and_thread_isolation():
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 진단", thread="a")
        other = send(client, "payments-service", thread="b")
        assert not monitored(other)
        repeat = send(client, "네", thread="a")
        assert any(e["type"] == "TargetConfirmationRequired" for e in repeat["events"])
        cancelled = send(client, "취소", thread="a", stream=True)
        assert "취소" in cancelled["reply"]
        assert not monitored(send(client, "payments-service", thread="a"))
        send(client, "CPU 급증 진단", thread="a")
        fresh = send(client, "log-service 디스크 장애 진단", thread="a")
        assert monitored(fresh) == ["log-service"]
        assert "CPU 급증" not in fresh["reply"]
        send(client, "CPU 급증 진단", thread="a")
        chat = send(client, "안녕하세요", thread="a")
        assert "안녕하세요" in chat["reply"]
        assert not monitored(send(client, "payments-service", thread="a"))


def test_unrelated_new_question_does_not_get_stuck_in_target_confirmation():
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 진단")
        result = send(client, "오늘 점심 메뉴 추천해줘")
        assert not any(e["type"] == "TargetConfirmationRequired" for e in result["events"])
        assert not monitored(send(client, "payments-service"))


@pytest.mark.parametrize("thread", ["", "restart"])
def test_pending_question_survives_store_reopen_and_empty_thread(thread, tmp_path):
    from agent.session_store import SqliteSessionStore
    path = tmp_path / "session.db"
    app, _, _ = make_test_app(store=SqliteSessionStore(path))
    with TestClient(app) as client:
        send(client, "CPU 급증 진단", thread=thread)
    app, store, _ = make_test_app(store=SqliteSessionStore(path))
    with TestClient(app) as client:
        result = send(client, "gateway-service와 log-service", thread=thread, stream=True)
        assert monitored(result) == ["gateway-service", "log-service"]
        assert "CPU 급증" in result["reply"]
        assert store.pending_target(thread) is None


def test_failed_diagnosis_keeps_pending_question_for_retry():
    app, store, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 진단")
        collector = app.state.deps.evidence_collector

        async def broken(tasks):
            raise TimeoutError("모니터링 실패")

        original = collector.collect
        collector.collect = broken
        with pytest.raises(TimeoutError):
            send(client, "payments-service")
        assert store.pending_target("targets").query == "CPU 급증 진단"
        collector.collect = original
        result = send(client, "payments-service")
        assert monitored(result) == ["payments-service"]
        assert "CPU 급증" in result["reply"]


def test_confirmation_never_calls_downstream_dependencies():
    app, _, _ = make_test_app()

    def forbidden(*args, **kwargs):
        pytest.fail("대상 확인 전에 계획·MCP·RAG·답변 작성을 실행했습니다")

    with TestClient(app) as client:
        deps = app.state.deps
        deps.fast_llm.plan = forbidden
        deps.mcp.find_prompt = forbidden
        deps.evidence_collector.collect = forbidden
        deps.quality_llm.write = forbidden
        from types import SimpleNamespace
        deps.runbook_retriever = SimpleNamespace(search=forbidden)
        result = send(client, "CPU 급증 진단", stream=True)
        assert any(e["type"] == "TargetConfirmationRequired" for e in result["events"])


def test_finishing_old_diagnosis_does_not_clear_a_new_pending_question():
    app, store, _ = make_test_app()
    with TestClient(app) as client:
        send(client, "CPU 급증 진단")
        writer = app.state.deps.quality_llm
        original = writer.write

        async def new_question_arrived(*args, **kwargs):
            # 이전 요청이 모델을 기다리는 동안 같은 스레드에 새 확인 요청이 도착한 상태.
            store.set_pending_target("targets", "메모리 장애 진단")
            return await original(*args, **kwargs)

        writer.write = new_question_arrived
        send(client, "payments-service")
        assert store.pending_target("targets").query == "메모리 장애 진단"


def test_pure_runbook_request_does_not_require_a_system_or_monitor():
    from agent.rag import Document, SearchResult

    class Retriever:
        async def search(self, query, observations):
            assert not observations
            return SearchResult([Document(
                "cpu-spike", "runbook://cpu-spike", "# CPU 급증 대응 절차\n1. 확인\n2. 조치",
            )], [], "명시적 절차 조회", {})

    app, _, _ = make_test_app()
    with TestClient(app) as client:
        app.state.deps.runbook_retriever = Retriever()
        result = send(client, "CPU 급증 대응 절차를 찾아줘", stream=True)
        assert not monitored(result)
        assert not any(e["type"] == "TargetConfirmationRequired" for e in result["events"])
        assert "runbook://cpu-spike" in result["reply"]


@pytest.mark.parametrize("backend", ["openai", "ollama", "fake"])
def test_planners_do_not_invent_targets_or_drop_explicit_targets(backend):
    from agent.fake_llm import FastLLM
    from agent.openai_llm import OpenAIFastLLM
    from agent.ollama_llm import OllamaFastLLM

    cls = {"openai": OpenAIFastLLM, "ollama": OllamaFastLLM, "fake": FastLLM}[backend]
    llm = cls("test", 0)
    llm._chat = lambda *a, **k: (json.dumps({"tasks": [
        {"agent": "research", "target": "auth-service"}]}), {"usd": 0})
    for question, expected in [
        ("CPU 급증 진단", []),
        ("payments-service-old 장애 진단", []),
        ("gateway-service와 log-service 장애 진단", ["gateway-service", "log-service"]),
    ]:
        result, _ = llm.plan(question, "표준 절차")
        assert [t["target"] for t in result["tasks"] if t["agent"] == "research"] == expected
