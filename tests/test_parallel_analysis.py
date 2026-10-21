"""명시적 비교 의도, 관측 수, 병렬 결과 합류와 실패 정리를 검증한다."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from agent.evidence import CollectionResult, Evidence
from agent.graph import analysis_node, build_graph, gate_node
from agent.rag import Document, SearchResult


COMPARISON = "payments-service랑 cache-service 둘 다 저하됐는데 공통 원인이 있는지 분석해줘"
LOOKUP = "web-01 CPU 급증, auth-service 커넥션 풀도 같이 확인해줘"


@pytest.mark.parametrize("query", [
    LOOKUP,
    "payments-service 결제가 느려요. 원인 확인해주세요",
    "payments-service와 cache-service 상태만 확인해줘",
    "payments-service와 cache-service는 비교하지 말고 각각 확인해줘",
    "두 서비스 공통 원인 분석은 생략해줘",
    "payments-service CPU와 메모리를 비교해줘",
    "payments-service와 cache-service는 비교적 정상인지 각각 확인해줘",
    "payments-service와 cache-service를 비교해줘. 아니, 비교는 생략해줘",
])
def test_gate_rejects_unrequested_analysis_despite_planner_suggestion(query):
    state = {"query": query, "needs_analysis": True, "events": [], "tasks": [
        {"id": "analysis", "agent": "analysis", "status": "pending"}]}
    original = copy.deepcopy(state)
    result = asyncio.run(gate_node(state, None))
    assert result["needs_analysis"] is False
    assert result["tasks"] == []
    assert state == original


@pytest.mark.parametrize("query", [
    COMPARISON,
    "payments-service와 cache-service 상태 차이를 비교해줘",
    "두 서비스 장애의 연관성을 확인해 주세요",
    "Compare payments-service and cache-service for a common cause",
    "두 서비스 비교는 생략해줘. 대신 공통 원인이 있는지 분석해줘",
])
def test_gate_adds_one_analysis_task_even_when_planner_omits_it(query):
    state = {"query": query, "needs_analysis": False, "events": [], "tasks": [
        {"id": "monitor", "agent": "research", "status": "pending"}]}
    result = asyncio.run(gate_node(state, None))
    assert result["needs_analysis"] is True
    assert result["tasks"][0] == state["tasks"][0]
    assert len([t for t in result["tasks"] if t["agent"] == "analysis"]) == 1


def test_gate_deduplicates_analysis_tasks_without_mutating_planner_output():
    state = {"query": COMPARISON, "events": [], "tasks": [
        {"id": name, "agent": "analysis", "status": "pending"} for name in ("a", "b")]}
    original = copy.deepcopy(state)
    result = asyncio.run(gate_node(state, None))
    assert len(result["tasks"]) == 1
    assert result["gate_removed"] == 1
    assert state == original


@pytest.mark.parametrize("names", [[], ["payments-service"], ["payments-service"] * 2])
def test_analysis_requires_two_distinct_observed_systems(names):
    def forbidden(*args):
        pytest.fail("관측이 부족하면 모델을 호출하면 안 됩니다")

    state = {"query": COMPARISON, "events": [], "cost_usd": 0.3, "tasks": [
        {"id": "analysis", "agent": "analysis", "status": "pending"}],
        "collected": [{"source": "monitoring", "data": {"system_name": name}} for name in names]}
    result = asyncio.run(analysis_node(state, SimpleNamespace(fast_llm=SimpleNamespace(analyze=forbidden))))
    assert result["events"][-1]["type"] == "AnalysisSkipped"
    assert result["events"][-1]["reason"] == "insufficient_system_observations"
    assert not any(t["status"] == "pending" for t in result["tasks"])
    assert "collected" not in result  # 분석하지 않은 내용을 사실로 추가하지 않는다.


def make_deps(analyze, retrieve, written):
    from agent.fake_llm import FastLLM

    class Fast(FastLLM):
        def route(self, query):
            return {"route": "incident", "needs_analysis": True}, {"usd": 0.1}

        def plan(self, query, procedure):
            return {"needs_analysis": True, "tasks": [
                {"id": "a", "agent": "research", "kind": "monitoring",
                 "target": "payments-service", "status": "pending"},
                {"id": "b", "agent": "research", "kind": "monitoring",
                 "target": "cache-service", "status": "pending"},
                {"id": "analysis", "agent": "analysis", "status": "pending"},
            ]}, {"usd": 0.2}

    fast = Fast("test", 0)
    fast.analyze = analyze

    class Collector:
        async def collect(self, tasks):
            return CollectionResult([
                Evidence(t.index, "monitoring", {
                    "system_name": t.target, "status": "degraded",
                    "cpu_percent": 90, "memory_percent": 45, "last_checked": "2026-10-21T09:00:00", "alerts": ["latency"],
                }, 0) for t in tasks
            ], 0.01)

    class Writer:
        async def write(self, query, collected, attempt, feedback, on_token=None):
            written.append(copy.deepcopy(collected))
            names = ", ".join(c["data"]["system_name"] for c in collected if c["source"] == "monitoring")
            return f"1) {names}의 모니터링과 런북을 확인했습니다. 2) 공통 원인은 가설입니다. 3) 추가 근거를 확인하세요.", {"usd": 0.8}

    async def find_prompt(name):
        return "ops"

    async def get_prompt(*args, **kwargs):
        return "관측을 확인하세요"

    from agent.validation import AnswerValidator
    return SimpleNamespace(
        answer_validator=AnswerValidator(fast),
        fast_llm=fast, quality_llm=Writer(), evidence_collector=Collector(),
        mcp=SimpleNamespace(find_prompt=find_prompt, get_prompt=get_prompt),
        runbook_retriever=SimpleNamespace(search=retrieve),
    )


def document_result():
    return SearchResult([Document("high-latency", "runbook://latency", "# 지연\n1. 확인")],
                        [], "관측 근거", {})


def test_rag_and_analysis_overlap_and_merge_without_duplicates_or_lost_cost():
    async def exercise():
        rag_started, analysis_started = asyncio.Event(), asyncio.Event()
        written = []

        async def retrieve(query, observations):
            rag_started.set()
            await analysis_started.wait()
            assert len(observations) == 2
            return document_result()

        async def analyze(collected):
            analysis_started.set()
            await rag_started.wait()
            assert [c["source"] for c in collected] == ["monitoring", "monitoring"]
            return "공통 원인 여부는 추가 확인이 필요합니다", {"usd": 0.4}

        graph = build_graph(make_deps(analyze, retrieve, written))
        result = await asyncio.wait_for(graph.ainvoke({
            "query": COMPARISON, "events": [], "cost_usd": 0.0}), 2)
        assert result["cost_usd"] == 1.5
        assert [c["source"] for c in written[0]] == ["monitoring", "monitoring", "runbook", "analysis"]
        types = [e["type"] for e in result["events"]]
        assert types.count("TaskCompleted") == 2
        assert types.count("RunbookRetrieved") == types.count("AnalysisCompleted") == 1
        assert result["tasks"][-1]["status"] == "done"

    asyncio.run(exercise())


def test_simple_lookup_uses_rag_without_analysis_call_or_analysis_cost():
    async def exercise():
        async def analyze(collected):
            pytest.fail("단순 복수 조회에서 분석 모델을 호출하면 안 됩니다")

        async def retrieve(query, observations):
            return document_result()

        written = []
        result = await build_graph(make_deps(analyze, retrieve, written)).ainvoke({
            "query": LOOKUP, "events": [], "cost_usd": 0.0})
        assert result["cost_usd"] == 1.1
        assert all(c["source"] != "analysis" for c in written[0])
        assert any(e["type"] == "AnalysisSkipped" for e in result["events"])

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["rag", "analysis", "request_cancel"])
def test_failure_cancels_and_joins_other_branch_before_returning(failure):
    async def exercise():
        started = {name: asyncio.Event() for name in ("rag", "analysis")}
        cleaned, written = [], []

        async def branch(name):
            started[name].set()
            try:
                await started["analysis" if name == "rag" else "rag"].wait()
                if name == failure:
                    raise RuntimeError("의도한 분기 오류")
                await asyncio.Event().wait()
            finally:
                cleaned.append(name)

        async def retrieve(*args):
            return await branch("rag")

        async def analyze(*args):
            return await branch("analysis")

        task = asyncio.create_task(build_graph(make_deps(analyze, retrieve, written)).ainvoke({
            "query": COMPARISON, "events": [], "cost_usd": 0.0}))
        if failure == "request_cancel":
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="의도한 분기 오류"):
                await asyncio.wait_for(task, 2)
        assert sorted(cleaned) == ["analysis", "rag"]
        assert written == []

    asyncio.run(exercise())
