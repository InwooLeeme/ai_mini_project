"""복수 시스템 검색에서 각 시스템의 후보가 합집합에 보존되는지 검증한다."""
import asyncio
import json

import pytest

from agent.rag import Document, RunbookRetriever


DOCUMENTS = [
    Document(name, f"runbook://{name}", f"# {name}\n1. 확인\n2. 조치")
    for name in "abcdef"
]
OBSERVATIONS = [
    {"system_name": "gateway", "alerts": ["timeout"]},
    {"system_name": "log", "alerts": ["disk full"]},
]


class Embeddings:
    def __init__(self):
        self.inputs = []

    async def embed(self, texts):
        if texts[0].startswith("title:"):
            return [[1, 0], [.8, .6], [.6, .8], [0, 1], [-1, 0], [0, -1]]
        self.inputs.append(texts)
        vectors = []
        for text in texts:
            observations = json.loads(text.split("\n관측: ", 1)[1])
            # 전체 입력에서는 a,b,c가 상위권이고 d가 탈락한다.
            vectors.append([0, 1] if observations and all(
                o.get("system_name") == "log" for o in observations) else [1, 0])
        return vectors


class Selector:
    def __init__(self):
        self.calls = []

    async def select(self, query, observations, documents):
        self.calls.append((query, observations, documents))
        ids = [d.document_id for d in documents if d.document_id in {"a", "d"}]
        return {"answerable": bool(ids), "document_ids": ids, "reason": "근거 확인"}, {}


def test_multi_system_union_preserves_each_top_three_and_full_source():
    async def exercise():
        embeddings, selector = Embeddings(), Selector()
        retriever = await RunbookRetriever.create(DOCUMENTS, embeddings, selector)
        result = await retriever.search("두 시스템의 장애", OBSERVATIONS)
        assert [d["document_id"] for d in result.candidates] == ["a", "d", "b", "c"]
        assert result.documents == [DOCUMENTS[0], DOCUMENTS[3]]
        assert result.usage["retrieval_query_count"] == 2
        assert result.usage["candidate_count"] == 4
        assert len(selector.calls) == 1
        assert selector.calls[0][:2] == ("두 시스템의 장애", OBSERVATIONS)
        assert selector.calls[0][2] == [DOCUMENTS[i] for i in [0, 3, 1, 2]]
        assert len(embeddings.inputs) == 1
        assert [json.loads(t.split("\n관측: ", 1)[1]) for t in embeddings.inputs[0]] == [
            [OBSERVATIONS[0]], [OBSERVATIONS[1]]]
        reversed_result = await retriever.search("두 시스템의 장애", OBSERVATIONS[::-1])
        assert reversed_result.candidates == result.candidates
    asyncio.run(exercise())


@pytest.mark.parametrize("observations", [[], OBSERVATIONS[:1], [OBSERVATIONS[0]] * 2])
def test_zero_or_one_system_preserves_existing_top_three(observations):
    async def exercise():
        embeddings = Embeddings()
        retriever = await RunbookRetriever.create(DOCUMENTS, embeddings, Selector())
        result = await retriever.search("질문", observations)
        assert [d["document_id"] for d in result.candidates] == ["a", "b", "c"]
        assert embeddings.inputs == [[
            "task: search result | query: 질문\n관측: " + json.dumps(observations, ensure_ascii=False)]]
    asyncio.run(exercise())


def test_multiple_observations_of_one_system_are_kept_together():
    async def exercise():
        embeddings = Embeddings()
        retriever = await RunbookRetriever.create(DOCUMENTS, embeddings, Selector())
        extra = {"system_name": "gateway", "alerts": ["packet loss"]}
        await retriever.search("질문", OBSERVATIONS + [extra])
        groups = [json.loads(t.split("\n관측: ", 1)[1]) for t in embeddings.inputs[0]]
        assert groups == [[OBSERVATIONS[0], extra], [OBSERVATIONS[1]]]
    asyncio.run(exercise())


@pytest.mark.parametrize("name", [None, "", " ", 17, []])
@pytest.mark.parametrize("known", [OBSERVATIONS, OBSERVATIONS[1:]])
def test_unknown_system_observation_preserves_known_candidates(name, known):
    async def exercise():
        unknown = {"system_name": name, "alerts": ["unknown target"]}
        observations = known + [unknown]
        embeddings = Embeddings()
        selector = Selector()
        retriever = await RunbookRetriever.create(DOCUMENTS, embeddings, selector)
        result = await retriever.search("질문", observations)
        assert "d" in [d.document_id for d in result.documents]
        assert len(embeddings.inputs[0]) == len(known) + 1
        assert json.loads(embeddings.inputs[0][-1].split("\n관측: ", 1)[1]) == [unknown]
        assert selector.calls[0][1] == observations
        assert result.usage["retrieval_groups"][-1]["system_name"] is None
    asyncio.run(exercise())


def test_only_unnamed_observations_keep_one_group_without_loss():
    async def exercise():
        observations = [{"alerts": ["first"]}, {"system_name": "", "alerts": ["second"]}]
        embeddings = Embeddings()
        retriever = await RunbookRetriever.create(DOCUMENTS, embeddings, Selector())
        result = await retriever.search("질문", observations)
        assert len(embeddings.inputs[0]) == 1
        assert json.loads(embeddings.inputs[0][0].split("\n관측: ", 1)[1]) == observations
        assert result.usage["retrieval_groups"][0]["system_name"] is None
        assert result.usage["retrieval_groups"][0]["observation_count"] == 2
    asyncio.run(exercise())


def test_retrieval_trace_explains_candidate_admission_without_changing_selector_input():
    async def exercise():
        selector = Selector()
        retriever = await RunbookRetriever.create(DOCUMENTS, Embeddings(), selector)
        result = await retriever.search("질문", OBSERVATIONS)
        assert retriever.metadata["top_k_documents_per_system"] == 3
        assert "top_k_documents" not in retriever.metadata
        groups = result.usage["retrieval_groups"]
        assert [g["system_name"] for g in groups] == ["gateway", "log"]
        assert [g["observation_count"] for g in groups] == [1, 1]
        assert groups[0]["candidates"] == [
            {"document_id": "a", "rank": 1, "score": 1.0},
            {"document_id": "b", "rank": 2, "score": .8},
            {"document_id": "c", "rank": 3, "score": .6},
        ]
        assert groups[1]["candidates"] == [
            {"document_id": "d", "rank": 1, "score": 1.0},
            {"document_id": "c", "rank": 2, "score": .8},
            {"document_id": "b", "rank": 3, "score": .6},
        ]
        assert selector.calls == [("질문", OBSERVATIONS, [DOCUMENTS[i] for i in [0, 3, 1, 2]])]
    asyncio.run(exercise())


def test_multi_system_selector_cannot_select_outside_union():
    class OutsideSelector:
        async def select(self, *args):
            return {"answerable": True, "document_ids": ["e"], "reason": "잘못된 선택"}, {}

    async def exercise():
        retriever = await RunbookRetriever.create(DOCUMENTS, Embeddings(), OutsideSelector())
        with pytest.raises(ValueError, match="후보"):
            await retriever.search("질문", OBSERVATIONS)
    asyncio.run(exercise())


@pytest.mark.parametrize("vectors, message", [
    ([[1, 0]], "개수"),
    ([[1, 0], [0, 1, 0]], "차원"),
    ([[1, 0], [float("nan"), 1]], "유효"),
])
def test_invalid_batch_embedding_never_produces_partial_candidates(vectors, message):
    class InvalidEmbeddings:
        async def embed(self, texts):
            return vectors

    async def exercise():
        selector = Selector()
        retriever = await RunbookRetriever.create(DOCUMENTS, Embeddings(), selector)
        retriever.embeddings = InvalidEmbeddings()
        with pytest.raises(ValueError, match=message):
            await retriever.search("질문", OBSERVATIONS)
        assert not selector.calls
    asyncio.run(exercise())
