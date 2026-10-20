"""자료 수집과 그래프 상태 반영의 독립적인 계약을 검증한다."""
import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.evidence import CollectionResult, Evidence, MCPEvidenceCollector, ResearchTask
from agent.graph import research_node


class ResearchNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_injected_collector_runs_without_mcp_and_preserves_state(self):
        state = {
            "tasks": [
                {"id": "analysis", "agent": "analysis", "status": "pending"},
                {"id": "old", "agent": "research", "kind": "monitoring",
                 "target": "old", "status": "done"},
                {"id": "monitor", "agent": "research", "kind": "monitoring",
                 "target": "web-01", "status": "pending"},
                {"id": "book", "agent": "research", "kind": "runbook",
                 "target": "db-down", "status": "pending"},
            ],
            "events": [{"type": "PlanCreated"}],
            "collected": [{"source": "runbook", "data": "기존 자료"}],
            "retry_count": 2,
        }
        original = copy.deepcopy(state)
        requests = []

        async def collect(tasks):
            requests.extend((t.index, t.kind, t.target) for t in tasks)
            return CollectionResult(items=[
                Evidence(2, "monitoring", monitoring("web-01"), 1),
                Evidence(3, "runbook", "복구 절차", 0),
            ], elapsed=0.42)

        deps = SimpleNamespace(evidence_collector=SimpleNamespace(collect=collect))
        result = await research_node(state, deps)

        self.assertEqual(requests, [(2, "monitoring", "web-01"), (3, "runbook", "db-down")])
        self.assertEqual(state, original)

        self.assertEqual([t["status"] for t in result["tasks"]],
                         ["pending", "done", "done", "done"])
        self.assertEqual(result["collected"], [
            {"source": "runbook", "data": "기존 자료"},
            {"source": "monitoring", "data": monitoring("web-01")},
            {"source": "runbook", "data": "복구 절차"},
        ])
        self.assertEqual(result["retry_count"], 3)
        self.assertEqual(result["parallel_elapsed"], 0.42)
        self.assertEqual(result["events"], [
            {"type": "PlanCreated"},
            {"type": "TaskCompleted", "id": "monitor", "source": "monitoring",
             "target": "web-01", "retries": 1},
            {"type": "TaskCompleted", "id": "book", "source": "runbook",
             "target": "db-down", "retries": 0},
        ])

    async def test_no_pending_research_skips_without_collector_access(self):
        state = {"tasks": [
            {"id": "done", "agent": "research", "status": "done"},
            {"id": "analysis", "agent": "analysis", "status": "pending"},
        ], "events": [{"type": "GateApplied"}]}
        result = await research_node(state, SimpleNamespace())
        self.assertEqual(result, {"events": [
            {"type": "GateApplied"},
            {"type": "ResearchSkipped", "reason": "pending research task 0개"},
        ]})

    async def test_collection_failure_does_not_apply_partial_state(self):
        state = {"tasks": [
            {"id": "monitor", "agent": "research", "kind": "monitoring",
             "target": "web-01", "status": "pending"},
        ], "events": [], "collected": [], "retry_count": 4}
        original = copy.deepcopy(state)

        async def collect(tasks):
            raise ValueError("수집 실패")

        deps = SimpleNamespace(evidence_collector=SimpleNamespace(collect=collect))
        with self.assertRaisesRegex(ValueError, "수집 실패"):
            await research_node(state, deps)
        self.assertEqual(state, original)


class CollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_collection_returns_input_order_with_original_indices(self):
        both_started = asyncio.Event()
        finish_first = asyncio.Event()
        started = []
        finished = []

        async def call_tool(name, arguments):
            self.assertEqual(name, "get_monitoring_status")
            target = arguments["system_name"]
            started.append(target)
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            if target == "web-01":
                await finish_first.wait()
            else:
                finish_first.set()
            finished.append(target)
            return monitoring(target)

        collector = MCPEvidenceCollector(SimpleNamespace(call_tool=call_tool))
        # 순차 실행으로 회귀하면 두 번째 작업이 시작되지 않아 이 제한에 걸린다.
        async with asyncio.timeout(1):
            result = await collector.collect([
                ResearchTask(7, "monitoring", "web-01"),
                ResearchTask(2, "monitoring", "auth-service"),
            ])
        self.assertEqual(finished, ["auth-service", "web-01"])
        self.assertEqual([(e.index, e.source, e.data["system_name"], e.retries)
                          for e in result.items],
                         [(7, "monitoring", "web-01", 0),
                          (2, "monitoring", "auth-service", 0)])

    async def test_discovers_once_and_excludes_discovery_time_from_elapsed(self):
        clock = [10.0]
        discoveries = []
        reads = []

        async def discover(name):
            discoveries.append(name)
            clock[0] = 15.0
            return "custom-docs", "custom://books/{incident_type}"

        async def read(uri, *, server):
            reads.append((uri, server))
            clock[0] = 15.376
            return f"절차: {uri}"

        collector = MCPEvidenceCollector(SimpleNamespace(
            find_resource_template=discover, read_resource=read))
        with patch("agent.evidence.time.perf_counter", side_effect=lambda: clock[0]):
            result = await collector.collect([
                ResearchTask(0, "runbook", "db-down"),
                ResearchTask(5, "legacy-resource", "cpu-spike"),
            ])
        self.assertEqual(discoveries, ["get_incident_runbook"])
        self.assertEqual(reads, [
            ("custom://books/db-down", "custom-docs"),
            ("custom://books/cpu-spike", "custom-docs"),
        ])
        self.assertEqual(result.elapsed, 0.38)
        self.assertEqual([(e.index, e.source, e.data, e.retries) for e in result.items], [
            (0, "runbook", "절차: custom://books/db-down", 0),
            (5, "runbook", "절차: custom://books/cpu-spike", 0),
        ])

    async def test_monitoring_timeout_retries_and_reports_count(self):
        attempts = []

        async def call_tool(name, arguments):
            attempts.append((name, arguments))
            if len(attempts) == 1:
                raise TimeoutError("일시적인 지연")
            return monitoring("web-01")

        collector = MCPEvidenceCollector(SimpleNamespace(call_tool=call_tool))
        with patch("agent.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await collector.collect([ResearchTask(3, "monitoring", "web-01")])
        self.assertEqual(attempts, [
            ("get_monitoring_status", {"system_name": "web-01"}),
            ("get_monitoring_status", {"system_name": "web-01"}),
        ])
        self.assertEqual(result.items[0].retries, 1)
        self.assertEqual(result.items[0].data, monitoring("web-01"))

    async def test_permanent_monitoring_error_is_propagated_without_retry(self):
        attempts = []
        error = ValueError("잘못된 응답")

        async def call_tool(name, arguments):
            attempts.append(arguments)
            raise error

        collector = MCPEvidenceCollector(SimpleNamespace(call_tool=call_tool))
        with self.assertRaises(ValueError) as raised:
            await collector.collect([ResearchTask(0, "monitoring", "web-01")])
        self.assertIs(raised.exception, error)
        self.assertEqual(attempts, [{"system_name": "web-01"}])

    async def test_runbook_timeout_is_propagated_without_retry(self):
        reads = []
        error = TimeoutError("런북 응답 지연")

        async def discover(name):
            return "docs", "runbook://{incident_type}"

        async def read(uri, *, server):
            reads.append((uri, server))
            raise error

        collector = MCPEvidenceCollector(SimpleNamespace(
            find_resource_template=discover, read_resource=read))
        with self.assertRaises(TimeoutError) as raised:
            await collector.collect([ResearchTask(1, "runbook", "db-down")])
        self.assertIs(raised.exception, error)
        self.assertEqual(reads, [("runbook://db-down", "docs")])

    async def test_discovery_failure_prevents_batch_calls(self):
        error = ValueError("런북 제공 서버 없음")

        async def discover(name):
            raise error

        # 도구 호출이 잘못 먼저 실행되면 존재하지 않는 메서드 접근으로 실패한다.
        collector = MCPEvidenceCollector(SimpleNamespace(find_resource_template=discover))
        with self.assertRaises(ValueError) as raised:
            await collector.collect([
                ResearchTask(0, "monitoring", "web-01"),
                ResearchTask(1, "runbook", "db-down"),
            ])
        self.assertIs(raised.exception, error)


def monitoring(system_name):
    """실제 MCP 모니터링 응답의 모든 필드를 갖춘 고정 데이터."""
    return {"system_name": system_name, "status": "healthy",
            "cpu_percent": 34.0, "memory_percent": 48.0,
            "last_checked": "2026-10-20T00:00:00+00:00", "alerts": []}
