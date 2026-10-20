"""검색 경계의 오류·순위·근거 없는 선택과 그래프 연결을 검증한다."""
import asyncio
import importlib
from types import SimpleNamespace

import pytest


def rag():
    return importlib.import_module("agent.rag")


def test_chunking_preserves_numbered_conditions_and_source():
    module = rag()
    doc = module.Document("cpu", "runbook://cpu", "# CPU\n\n1. 90% 이상이면 확인한다.\n2. 지속되면 조치한다.\n3. 복구 확인.")
    chunks = module.chunk_document(doc)
    assert len(chunks) == 2
    assert "1. 90% 이상이면 확인한다." in chunks[0].text
    assert "2. 지속되면 조치한다." in chunks[0].text
    assert chunks[0].document_id == chunks[1].document_id == "cpu"


class Vectors:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Selector:
    def __init__(self, ids):
        self.ids = ids

    async def select(self, query, observations, documents):
        return {"answerable": bool(self.ids), "document_ids": self.ids, "reason": "문서 근거 확인"}, {}


def test_search_deduplicates_chunks_and_keeps_full_source_context():
    module = rag()
    async def exercise():
        documents = [module.Document("a", "runbook://a", "# 제목\n1. 확인\n2. 조치\n3. 복구")]
        search = await module.RunbookRetriever.create(documents, Vectors(), Selector(["a"]))
        result = await search.search("조치", [])
        assert len(result.candidates) == len(result.documents) == 1
        assert result.documents[0].text == documents[0].text
        assert result.documents[0].uri == "runbook://a"
    asyncio.run(exercise())


def test_empty_selection_does_not_fall_back_to_db():
    module = rag()
    async def exercise():
        search = await module.RunbookRetriever.create(
            [module.Document("db-down", "runbook://db", "# DB\n1. 확인")], Vectors(), Selector([]))
        result = await search.search("인증서 갱신", [])
        assert result.documents == []
        assert result.candidates
    asyncio.run(exercise())


def test_selector_cannot_invent_a_document_id():
    module = rag()
    async def exercise():
        search = await module.RunbookRetriever.create(
            [module.Document("a", "runbook://a", "# A\n1. 확인")], Vectors(), Selector(["invented"]))
        with pytest.raises(ValueError, match="후보"):
            await search.search("query", [])
    asyncio.run(exercise())


@pytest.mark.parametrize("vector", [[float("nan"), 1], [0, 0], []])
def test_invalid_embedding_is_an_error_not_no_result(vector):
    module = rag()
    with pytest.raises(ValueError):
        module.normalize(vector)


def test_graph_passes_retrieved_sources_to_writer_without_legacy_gap():
    module = rag()
    from agent.graph import replan_check_node
    class Retriever:
        async def search(self, query, observations):
            return module.SearchResult(
                [module.Document("cpu", "runbook://cpu", "# CPU\n1. 확인")], [], "근거", {})
    state = {"query": "장애", "collected": [], "events": [], "tasks": []}
    result = asyncio.run(replan_check_node(state, SimpleNamespace(runbook_retriever=Retriever())))
    assert result["collected"][0]["uri"] == "runbook://cpu"
    assert result["collected"][0]["source"] == "runbook"
    assert result["events"][-1]["is_sufficient"] is True
    assert "tasks" not in result


def test_no_evidence_answer_does_not_call_model_or_recommend_restart():
    module = rag()
    from agent.graph import write_node
    from unittest.mock import patch
    state = {"query": "인증서 갱신", "collected": [], "events": [],
             "retrieval": {"document_ids": [], "reason": "해당 문서 없음"}}
    with patch("agent.graph.get_stream_writer", return_value=lambda event: None):
        result = asyncio.run(write_node(state, SimpleNamespace()))
    assert "근거" in result["draft"] and "재시작" not in result["draft"]


def test_embedding_failure_propagates_instead_of_becoming_abstention():
    module = rag()
    class Failure:
        async def embed(self, texts):
            raise TimeoutError("embedding unavailable")
    async def exercise():
        search = await module.RunbookRetriever.create(
            [module.Document("a", "runbook://a", "# A\n1. 확인")], Vectors(), Selector([]))
        search.embeddings = Failure()
        with pytest.raises(TimeoutError):
            await search.search("질문", [])
    asyncio.run(exercise())


def test_query_dimension_mismatch_is_rejected():
    module = rag()
    class WrongDimension:
        async def embed(self, texts):
            return [[1, 0, 0]]
    async def exercise():
        search = await module.RunbookRetriever.create(
            [module.Document("a", "runbook://a", "# A\n1. 확인")], Vectors(), Selector([]))
        search.embeddings = WrongDimension()
        with pytest.raises(ValueError, match="차원"):
            await search.search("질문", [])
    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_real_writer_prompt_contains_source_uri_and_all_selected_documents(backend):
    module = importlib.import_module(f"agent.{backend}_llm")
    model_class = getattr(module, "OpenAIQualityLLM" if backend == "openai" else "OllamaQualityLLM")
    model = model_class("test", 0)
    captured = []
    def chat(system, user, **kwargs):
        captured.append((system, user))
        return "1) 근거 기반 결과", {"usd": 0}
    model._chat = chat
    async def stream(system, user, **kwargs):
        return chat(system, user, **kwargs)
    model._chat_stream = stream
    collected = [{"source": "runbook", "uri": f"runbook://{name}", "data": f"# {name}\n1. 확인"}
                 for name in ["cpu", "memory"]]
    asyncio.run(model.write("장애", collected, 1, None))
    assert all(f"runbook://{name}" in captured[0][1] for name in ["cpu", "memory"])
    assert "출처" in captured[0][0]


def test_fake_writer_shows_both_sources_for_multi_document_retrieval():
    from agent.fake_llm import QualityLLM
    collected = [{"source": "runbook", "uri": f"runbook://{name}", "data": f"# {name}\n1. 확인\n2. 조치"}
                 for name in ["cpu", "memory"]]
    text, _ = asyncio.run(QualityLLM("fake", 0).write("복합 증상", collected, 2, None))
    assert "runbook://cpu" in text and "runbook://memory" in text


def test_explicit_unsupported_decision_discards_model_candidate_ids():
    module = rag()
    class Unsupported:
        async def select(self, query, observations, documents):
            return {"answerable": False, "document_ids": ["a"],
                    "reason": "주제는 비슷하지만 요청 절차가 없음"}, {}
    async def exercise():
        search = await module.RunbookRetriever.create(
            [module.Document("a", "runbook://a", "# A\n1. 확인")], Vectors(), Unsupported())
        result = await search.search("다른 작업", [])
        assert result.documents == []
    asyncio.run(exercise())


def test_selector_schema_limits_ids_to_actual_candidates(monkeypatch):
    import json
    import httpx
    module = rag()
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "answerable": False, "document_ids": [], "reason_code": "missing_procedure"})}})
    original = httpx.AsyncClient
    monkeypatch.setattr(module.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    asyncio.run(module.OllamaModels().select("질문", [], [
        module.Document("a", "runbook://a", "# A\n1. 확인")]))
    assert seen[0]["format"]["properties"]["document_ids"]["items"]["enum"] == ["a"]
