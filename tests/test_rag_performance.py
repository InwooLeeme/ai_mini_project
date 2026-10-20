"""검색 계측과 성능 개선이 근거 판단 계약을 보존하는지 검사한다."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agent import rag


def test_selector_reports_provider_durations_without_inventing_missing_values(monkeypatch):
    """단위 오변환·누락값의 0 채우기는 병목 판단을 왜곡한다."""
    def handler(request):
        return httpx.Response(200, json={
            "message": {"content": json.dumps({
                "answerable": False, "document_ids": [], "reason_code": "missing_procedure"})},
            "total_duration": 2_000_000_000, "load_duration": 250_000_000,
            "prompt_eval_duration": 500_000_000, "eval_duration": 1_000_000_000,
            "prompt_eval_count": 100, "eval_count": 20,
        })
    original = httpx.AsyncClient
    monkeypatch.setattr(rag.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    _, usage = asyncio.run(rag.OllamaModels().select("질문", [], [
        rag.Document("a", "runbook://a", "# A\n1. 확인")]))
    assert usage["selector_total_ms"] == 2000
    assert usage["selector_load_ms"] == 250
    assert usage["selector_prompt_eval_ms"] == 500
    assert usage["selector_eval_ms"] == 1000
    assert usage["selector_input_tokens"] == 100


def test_search_reports_separate_stage_times_and_preserves_selector_usage(monkeypatch):
    # 실제 대기 없이 구간 차이와 초→밀리초 변환을 각각 검증한다.
    clock = iter([10.0, 10.125, 10.25, 10.75, 11.0])
    monkeypatch.setattr(rag, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    class Embeddings:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Selector:
        async def select(self, query, observations, documents):
            return {"answerable": True, "document_ids": ["a"], "reason": "근거 있음"}, {
                "selector_input_tokens": 20}

    async def exercise():
        retriever = await rag.RunbookRetriever.create(
            [rag.Document("a", "runbook://a", "# A\n1. 확인")], Embeddings(), Selector())
        result = await retriever.search("질문", [])
        assert [d.document_id for d in result.documents] == ["a"]
        assert result.usage["selector_input_tokens"] == 20
        times = result.usage["timings_ms"]
        assert times == {"embedding": 125, "ranking": 125, "selection": 500, "total": 1000}
    asyncio.run(exercise())


@pytest.mark.parametrize("value", [None, -1, True, float("nan"), "123"])
def test_invalid_provider_timing_is_not_reported_as_measured_zero(value):
    assert rag.ollama_usage({"load_duration": value}, "selector")["selector_load_ms"] is None


def test_embedding_trace_keeps_dimensions_and_timing_without_vector_payload(monkeypatch):
    from langsmith import Client, tracing_context
    runs = []

    def handler(request):
        return httpx.Response(200, json={
            "embeddings": [[0.25, 0.75]], "load_duration": 1_000_000})
    original = httpx.AsyncClient
    monkeypatch.setattr(rag.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))

    async def exercise():
        client = Client(api_url="http://127.0.0.1:1", api_key="offline-test-key",
                        auto_batch_tracing=False, info={"version": "0"})
        try:
            with tracing_context(enabled="local", client=client):
                vectors = await rag.OllamaModels().embed(
                    ["입력"], langsmith_extra={"on_end": runs.append})
        finally:
            client.close()
        assert vectors == [[0.25, 0.75]]
        assert runs[0].outputs == {"vector_count": 1, "dimensions": 2}
        assert runs[0].extra["metadata"]["embedding_load_ms"] == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("answerable,ids,code,expected", [
    (True, ["a"], "supported", "요청 절차 또는 관측된 장애를 뒷받침하는 런북을 선택했습니다."),
    (False, [], "missing_procedure", "요청한 절차를 뒷받침하는 런북이 없습니다."),
    (False, [], "normal_observations", "관측이 정상이며 별도로 요청된 대응 절차가 없습니다."),
    (False, [], "unclear_symptoms", "증상이 구체적이지 않아 대응 런북을 확정할 수 없습니다."),
])
def test_compact_selector_decision_returns_readable_reason(
        monkeypatch, answerable, ids, code, expected):
    """짧은 모델 출력에서도 채택/미채택 판단과 사용자에게 보여줄 이유를 보존한다."""
    def handler(request):
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "answerable": answerable, "document_ids": ids, "reason_code": code})}})
    original = httpx.AsyncClient
    monkeypatch.setattr(rag.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    result, _ = asyncio.run(rag.OllamaModels().select(
        "질문", [], [rag.Document("a", "runbook://a", "# A\n1. 확인")]))
    assert result["reason"] == expected
    assert result["answerable"] is answerable
    assert result["document_ids"] == ids


@pytest.mark.parametrize("answerable,code", [
    (False, "invented"), (False, ["missing_procedure"]),
    (False, "supported"), (True, "normal_observations"),
])
def test_invalid_reason_is_an_error_not_a_valid_empty_result(monkeypatch, answerable, code):
    def handler(request):
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "answerable": answerable, "document_ids": [], "reason_code": code})}})
    original = httpx.AsyncClient
    monkeypatch.setattr(rag.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    with pytest.raises(ValueError, match="사유"):
        asyncio.run(rag.OllamaModels().select("질문", [], []))


@pytest.fixture
def selector_trace(monkeypatch):
    """HTTP만 대체하고 실제 선택 함수와 로컬 LangSmith 추적을 실행한다."""
    from langsmith import Client, tracing_context
    client = Client(api_url="http://127.0.0.1:1", api_key="offline-test-key",
                    auto_batch_tracing=False, info={"version": "0"})
    original = httpx.AsyncClient

    def exercise(body, expected_error=None):
        runs, requests = [], []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=body)

        monkeypatch.setattr(rag.httpx, "AsyncClient",
                            lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))

        async def call():
            with tracing_context(enabled="local", client=client):
                return await rag.OllamaModels().select(
                    "질문", [], [rag.Document("a", "runbook://a", "# A\n1. 확인")],
                    langsmith_extra={"on_end": runs.append})

        if expected_error:
            with pytest.raises(expected_error):
                asyncio.run(call())
            result = None
        else:
            result = asyncio.run(call())
        return runs[0], requests[0], result

    yield exercise
    client.close()


@pytest.mark.parametrize("content,error,done_reason", [
    ('{"answerable":true,"document_ids":["a"],"reason_code":"supported"}', None, "stop"),
    ('{"answerable":true,"document_ids":[', json.JSONDecodeError, "length"),
    ('{"answerable":true,"document_ids":["a"],"reason_code":"normal_observations"}',
     ValueError, "stop"),
])
def test_selector_trace_keeps_usage_even_when_decision_is_invalid(
        selector_trace, content, error, done_reason):
    """파싱·판단 검증 실패도 이미 발생한 모델 사용량을 잃지 않아야 한다."""
    run, _, result = selector_trace({
        "message": {"content": content}, "prompt_eval_count": 100, "eval_count": 20,
        "total_duration": 2_000_000_000, "done": True, "done_reason": done_reason,
    }, error)
    assert run.metadata["usage_metadata"] == {
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    assert run.metadata["ls_provider"] == "ollama"
    assert run.metadata["ls_model_name"] == rag.OllamaModels().selection_model
    assert run.metadata["selector_total_ms"] == 2000
    assert run.metadata["selector_done_reason"] == done_reason
    assert bool(run.error) is bool(error)
    if result:
        assert result[0]["document_ids"] == ["a"]
        assert result[1]["selector_input_tokens"] == 100


@pytest.mark.parametrize("count,expected", [
    (None, {"output_tokens": 20}), (-1, {"output_tokens": 20}),
    (True, {"output_tokens": 20}), (1.5, {"output_tokens": 20}),
    (0, {"input_tokens": 0, "output_tokens": 20, "total_tokens": 20}),
])
def test_selector_trace_does_not_invent_or_accept_invalid_token_counts(selector_trace, count, expected):
    run, _, _ = selector_trace({
        "message": {"content": '{"answerable":false,"document_ids":[],"reason_code":"missing_procedure"}'},
        "prompt_eval_count": count, "eval_count": 20,
    })
    assert run.metadata["usage_metadata"] == expected


def test_selector_requests_bounded_output_schema(selector_trace):
    _, request, _ = selector_trace({
        "message": {"content": '{"answerable":true,"document_ids":["a"],"reason_code":"supported"}'},
    })
    schema = request["format"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["document_ids"]["maxItems"] == 1
    assert schema["properties"]["document_ids"]["items"]["enum"] == ["a"]


def test_selector_rejects_unexpected_prose_fields(selector_trace):
    selector_trace({
        "message": {"content": json.dumps({
            "answerable": True, "document_ids": ["a"], "reason_code": "supported",
            "explanation": "허용하지 않은 추가 설명",
        })},
    }, ValueError)
