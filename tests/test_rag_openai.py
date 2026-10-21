"""OpenAI RAG 백엔드가 Ollama와 같은 임베딩·근거 선택 계약을 지키는지 검사한다."""
import asyncio
import json

import httpx
import pytest

from agent import rag

DOCS = [rag.Document("a", "runbook://a", "# A\n1. 확인"),
        rag.Document("b", "runbook://b", "# B\n1. 확인")]


@pytest.fixture
def openai_http(monkeypatch):
    """HTTP만 대체하고 실제 OpenAIModels 코드를 실행한다."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    original = httpx.AsyncClient
    requests = []

    def install(body):
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=body)
        monkeypatch.setattr(rag.httpx, "AsyncClient",
                            lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
        return requests
    return install


def test_embed_restores_input_order_from_response_index(openai_http):
    requests = openai_http({"data": [{"index": 1, "embedding": [0.0, 1.0]},
                                     {"index": 0, "embedding": [1.0, 0.0]}],
                            "usage": {"prompt_tokens": 4, "total_tokens": 4}})
    vectors = asyncio.run(rag.OpenAIModels().embed(["첫째", "둘째"]))
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/embeddings"
    assert body == {"model": "text-embedding-3-small", "input": ["첫째", "둘째"]}


def test_embed_count_mismatch_is_an_error(openai_http):
    openai_http({"data": [{"index": 0, "embedding": [1.0]}]})
    with pytest.raises(ValueError, match="개수"):
        asyncio.run(rag.OpenAIModels().embed(["첫째", "둘째"]))


def test_missing_api_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(rag.httpx, "AsyncClient", lambda **kw: pytest.fail("network call"))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(rag.OpenAIModels().embed(["입력"]))


def test_select_uses_strict_schema_and_reports_usage(openai_http):
    requests = openai_http({
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "answerable": True, "document_ids": ["b"], "reason_code": "supported"})}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
    selection, usage = asyncio.run(rag.OpenAIModels().select("질문", [], DOCS))
    assert selection["document_ids"] == ["b"]
    assert selection["reason"] == rag.SELECTION_REASONS["supported"]
    assert usage["selector_model"] == "gpt-4o-mini"
    assert usage["selector_input_tokens"] == 100
    assert usage["selector_output_tokens"] == 20
    assert usage["selector_done_reason"] == "stop"
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert body["temperature"] == 0
    schema = body["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["properties"]["document_ids"]["items"]["enum"] == ["a", "b"]


def test_select_rejects_inconsistent_decision(openai_http):
    openai_http({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
        "answerable": True, "document_ids": [], "reason_code": "normal_observations"})}}]})
    with pytest.raises(ValueError, match="사유"):
        asyncio.run(rag.OpenAIModels().select("질문", [], DOCS))


@pytest.mark.parametrize("backend,expected", [
    (None, rag.OllamaModels), ("ollama", rag.OllamaModels), ("openai", rag.OpenAIModels)])
def test_backend_selects_model_class(monkeypatch, backend, expected):
    if backend is None:
        monkeypatch.delenv("RAG_BACKEND", raising=False)
    else:
        monkeypatch.setenv("RAG_BACKEND", backend)
    assert type(rag.build_models()) is expected


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "opneai")
    with pytest.raises(ValueError, match="RAG_BACKEND"):
        rag.build_models()
