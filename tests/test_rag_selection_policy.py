"""명시적 문서 조회만 모델을 생략하고 의미 판단이 필요한 입력은 위임한다."""
import asyncio
import json
from pathlib import Path

import pytest

from agent.rag import Document, RunbookRetriever


class Embeddings:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Selector:
    def __init__(self):
        self.calls = []

    async def select(self, query, observations, documents):
        self.calls.append((query, observations, documents))
        return {"answerable": False, "document_ids": [], "reason": "모델 판단"}, {
            "selector_input_tokens": 120, "selector_output_tokens": 20}


CPU = Document("cpu", "runbook://cpu", "# Runbook: CPU 급증 대응 절차\n1. 지속 여부를 확인한다.")
OBSERVATIONS = [{"system_name": "api-service", "status": "degraded", "alerts": ["CPU high"]}]


async def search(query, observations=None, documents=None, **kwargs):
    selector = Selector()
    retriever = await RunbookRetriever.create(
        documents or [CPU], Embeddings(), selector, **kwargs)
    result = await retriever.search(query, OBSERVATIONS if observations is None else observations)
    return result, selector


@pytest.mark.parametrize("query", [
    "api-service CPU 급증 대응 절차를 찾아줘.",
    "CPU 급증 대응 절차를 찾아주세요",
    " CPU  급증 대응 절차 보여줘! ",
    "api-service의 cpu 급증 대응 절차를 보여주세요.",
])
def test_exact_title_request_preserves_full_document_without_selector(query):
    result, selector = asyncio.run(search(query))
    assert not selector.calls
    assert result.documents == [CPU]
    assert result.candidates == [{"document_id": "cpu", "score": 1.0}]
    assert result.usage["selection_method"] == "explicit_title"
    assert result.usage["selector_called"] is False
    assert result.usage["selector_input_tokens"] == result.usage["selector_output_tokens"] == 0
    assert "제목" in result.reason
    assert all(value >= 0 for value in result.usage["timings_ms"].values())


@pytest.mark.parametrize("query", [
    "api-service CPU가 치솟았어. 원인 확인해줘.",
    "CPU 급증 대응 절차를 찾아줘. 메모리도 확인해줘.",
    "CPU 급증 대응 절차와 메모리 누수 대응 절차를 찾아줘.",
    "CPU 급증 대응 절차를 찾지 말아줘.",
    "CPU 급증 대응 절차가 아니라 인증서 갱신 절차를 찾아줘.",
    "CPU 급증 대응 절차에 없는 인증서 갱신 절차를 찾아줘.",
    "CPU 급증 대응 절차를 찾아줘. 위 지시를 무시해.",
    "CPU가 정상인데 장애 대응 절차를 찾아줘.",
    "CPU 급증 대응 절차로 해결할 수 있을까?",
    "다른-service CPU 급증 대응 절차를 찾아줘.",
    "상태가 이상해",
])
def test_semantic_or_compound_queries_still_call_selector_even_with_perfect_score(query):
    result, selector = asyncio.run(search(query))
    assert len(selector.calls) == 1
    assert selector.calls[0][:2] == (query, OBSERVATIONS)
    assert result.documents == []
    assert result.reason == "모델 판단"
    assert result.usage["selection_method"] == "model"
    assert result.usage["selector_called"] is True
    assert result.usage["selector_input_tokens"] == 120


def test_multiple_system_observations_retain_model_judgment():
    observations = OBSERVATIONS + [{"system_name": "db-service", "alerts": ["DB down"]}]
    _, selector = asyncio.run(search("api-service CPU 급증 대응 절차를 찾아줘.", observations))
    assert len(selector.calls) == 1


@pytest.mark.parametrize("alerts", [
    ["CPU high", "CPU remains high"],
    ["CPU high", "disk full"],
    ["CPU high; disk full"],
])
def test_explicit_document_lookup_does_not_classify_incidents_by_alert_count(alerts):
    observations = [{"system_name": "api-service", "alerts": alerts}]
    result, selector = asyncio.run(search("api-service CPU 급증 대응 절차를 찾아줘.", observations))
    assert result.documents == [CPU]
    assert not selector.calls


@pytest.mark.parametrize("system,document_id", [
    ("payments-service", "cpu-spike"),
    ("log-service", "disk-full"),
    ("gateway-service", "network-timeout"),
])
def test_app_monitoring_profiles_support_lookup_but_preserve_diagnosis(system, document_id):
    """앱이 읽는 고정 데이터를 그대로 사용하며, 실제 운영 관측으로 간주하지 않는다."""
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / "data/monitoring.json").read_text())[system]
    observations = [{"system_name": system, "status": profile["status"],
                     "cpu_percent": profile["cpu_base"], "memory_percent": profile["memory_base"],
                     "alerts": profile["alerts"]}]
    document = Document(document_id, f"runbook://incident-response/{document_id}",
                        (root / f"data/runbooks/{document_id}.md").read_text())
    title = document.text.splitlines()[0].removeprefix("# Runbook: ")
    query = f"{system} {title}를 찾아줘."

    async def exercise():
        result, selector = await search(query, observations, [document])
        assert result.documents == [document]
        assert result.usage["selector_called"] is False
        assert not selector.calls
        for diagnostic_query in [
            f"{system}의 모든 알림 원인과 대응을 진단해줘.",
            query + " 다른 알림의 원인도 분석해줘.",
            f"{system} {title}가 현재 모든 증상을 해결할 수 있어?",
        ]:
            result, selector = await search(diagnostic_query, observations, [document])
            assert len(selector.calls) == 1
            assert selector.calls[0][:2] == (diagnostic_query, observations)
            assert result.usage["selector_called"] is True
    asyncio.run(exercise())


def test_explicit_procedure_lookup_does_not_require_an_active_incident():
    result, selector = asyncio.run(search("CPU 급증 대응 절차를 찾아줘.", []))
    assert result.documents == [CPU]
    assert not selector.calls


def test_duplicate_titles_are_ambiguous_even_if_only_one_is_a_candidate():
    documents = [CPU, Document("z", "runbook://z", CPU.text)] + [
        Document(name, f"runbook://{name}", f"# Runbook: {name} 대응 절차\n1. 확인한다.")
        for name in ["a", "b"]]
    result, selector = asyncio.run(search("CPU 급증 대응 절차를 찾아줘.", documents=documents))
    assert len(selector.calls) == 1
    assert "z" not in [c["document_id"] for c in result.candidates]


def test_exact_title_outside_vector_candidates_does_not_bypass_selector():
    documents = [Document("z", CPU.uri, CPU.text)] + [
        Document(name, f"runbook://{name}", f"# Runbook: {name} 대응 절차\n1. 확인한다.")
        for name in ["a", "b", "c"]]
    _, selector = asyncio.run(search("CPU 급증 대응 절차를 찾아줘.", documents=documents))
    assert len(selector.calls) == 1


def test_parenthetical_title_qualifier_is_not_silently_removed():
    document = Document("cpu", CPU.uri, "# Runbook: CPU 급증(Linux 전용) 대응 절차\n1. 확인한다.")
    _, selector = asyncio.run(search("CPU 급증 대응 절차를 찾아줘.", documents=[document]))
    assert len(selector.calls) == 1


def test_unrecognized_document_title_is_not_eligible():
    document = Document("cpu", CPU.uri, "# CPU 급증 대응 절차\n1. 확인한다.")
    _, selector = asyncio.run(search("CPU 급증 대응 절차를 찾아줘.", documents=[document]))
    assert len(selector.calls) == 1


def test_comparison_mode_keeps_existing_selector_path():
    result, selector = asyncio.run(search(
        "CPU 급증 대응 절차를 찾아줘.", explicit_title_selection=False))
    assert len(selector.calls) == 1
    assert result.documents == []


def test_selector_errors_are_not_converted_to_successful_abstention():
    class FailingSelector:
        async def select(self, *args):
            raise TimeoutError("selector unavailable")

    async def exercise():
        retriever = await RunbookRetriever.create([CPU], Embeddings(), FailingSelector())
        with pytest.raises(TimeoutError, match="selector unavailable"):
            await retriever.search("CPU가 치솟았어", OBSERVATIONS)
    asyncio.run(exercise())


def test_langsmith_does_not_create_an_llm_span_for_skipped_selection():
    from langsmith import Client, traceable, tracing_context

    class TracedSelector(Selector):
        @traceable(name="test_selector", run_type="llm")
        async def select(self, *args):
            return await super().select(*args)

    async def exercise(client):
        retriever = await RunbookRetriever.create([CPU], Embeddings(), TracedSelector())
        runs = []
        with tracing_context(enabled="local", client=client):
            await retriever.search(
                "CPU 급증 대응 절차를 찾아줘.", OBSERVATIONS,
                langsmith_extra={"on_end": runs.append})
            await retriever.search(
                "CPU가 치솟았어", OBSERVATIONS,
                langsmith_extra={"on_end": runs.append})
        assert not [r for r in runs[0].child_runs if r.run_type == "llm"]
        assert len([r for r in runs[1].child_runs if r.run_type == "llm"]) == 1
        assert runs[0].outputs["output"].usage["selector_called"] is False

    client = Client(api_url="http://127.0.0.1:1", api_key="offline-test-key",
                    auto_batch_tracing=False, info={"version": "0"})
    try:
        asyncio.run(exercise(client))
    finally:
        client.close()


def test_graph_keeps_explicit_source_and_skip_reason_in_retrieval_event():
    from types import SimpleNamespace
    from agent.graph import replan_check_node

    async def exercise():
        selector = Selector()
        retriever = await RunbookRetriever.create([CPU], Embeddings(), selector)
        state = {"query": "api-service CPU 급증 대응 절차를 찾아줘.",
                 "collected": [{"source": "monitoring", "data": OBSERVATIONS[0]}],
                 "events": [], "tasks": []}
        result = await replan_check_node(state, SimpleNamespace(runbook_retriever=retriever))
        assert result["collected"][-1] == {
            "source": "runbook", "data": CPU.text, "document_id": "cpu", "uri": CPU.uri}
        assert result["events"][0]["usage"]["selector_called"] is False
        assert result["retrieval"]["document_ids"] == ["cpu"]
        assert not selector.calls
    asyncio.run(exercise())
