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
        # 시스템마다 자기 관측과 자기 후보만으로 따로 판정한다.
        assert selector.calls == [
            ("두 시스템의 장애", [OBSERVATIONS[0]], [DOCUMENTS[i] for i in [0, 1, 2]]),
            ("두 시스템의 장애", [OBSERVATIONS[1]], [DOCUMENTS[i] for i in [3, 2, 1]]),
        ]
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
        assert [call[1] for call in selector.calls] == [[o] for o in observations]
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


def test_retrieval_trace_explains_candidate_admission_and_per_system_selection():
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
        assert [g["selected_document_ids"] for g in groups] == [["a"], ["d"]]
    asyncio.run(exercise())


class FocusSelector:
    """질문 주제에 끌려 첫 후보만 고르는 모델처럼 동작한다 (2026-10-21 C02 재현)."""
    def __init__(self, usages=None):
        self.usages = list(usages or [])

    async def select(self, query, observations, documents):
        usage = self.usages.pop(0) if self.usages else {}
        return ({"answerable": True, "document_ids": [documents[0].document_id],
                 "reason": "근거 확인"}, usage)


def test_one_system_focus_does_not_drop_another_systems_document():
    async def exercise():
        retriever = await RunbookRetriever.create(DOCUMENTS, Embeddings(), FocusSelector())
        result = await retriever.search("gateway 타임아웃", OBSERVATIONS)
        assert result.documents == [DOCUMENTS[0], DOCUMENTS[3]]
    asyncio.run(exercise())


def test_per_system_selector_usage_is_summed_without_inventing_missing_counts():
    async def exercise():
        usages = [{"selector_model": "m", "selector_input_tokens": 100, "selector_output_tokens": 20,
                   "selector_reason_code": "supported"},
                  {"selector_model": "m", "selector_input_tokens": 120, "selector_output_tokens": None,
                   "selector_reason_code": "supported"}]
        retriever = await RunbookRetriever.create(DOCUMENTS, Embeddings(), FocusSelector(usages))
        result = await retriever.search("질문", OBSERVATIONS)
        assert result.usage["selector_model"] == "m"
        assert result.usage["selector_input_tokens"] == 220
        assert result.usage["selector_output_tokens"] is None
        assert result.usage["selector_reason_code"] == "supported"
        assert result.usage["selector_call_count"] == 2
    asyncio.run(exercise())


def test_supported_system_is_kept_when_another_system_is_normal():
    class PartialSelector:
        async def select(self, query, observations, documents):
            if observations[0]["system_name"] == "log":
                return {"answerable": False, "document_ids": [],
                        "reason": "관측 정상"}, {"selector_reason_code": "normal_observations"}
            return {"answerable": True, "document_ids": ["a"],
                    "reason": "근거 확인"}, {"selector_reason_code": "supported"}

    async def exercise():
        retriever = await RunbookRetriever.create(DOCUMENTS, Embeddings(), PartialSelector())
        result = await retriever.search("질문", OBSERVATIONS[::-1])
        assert result.documents == [DOCUMENTS[0]]
        assert result.reason == "근거 확인"
        assert result.usage["selector_reason_code"] == "supported"
    asyncio.run(exercise())


def test_no_supported_system_returns_empty_with_first_reason():
    class RejectSelector:
        async def select(self, query, observations, documents):
            return {"answerable": False, "document_ids": [],
                    "reason": observations[0]["system_name"]}, {"selector_reason_code": "normal_observations"}

    async def exercise():
        retriever = await RunbookRetriever.create(DOCUMENTS, Embeddings(), RejectSelector())
        result = await retriever.search("질문", OBSERVATIONS)
        assert result.documents == []
        assert result.reason == "gateway"
        assert result.usage["selector_reason_code"] == "normal_observations"
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
