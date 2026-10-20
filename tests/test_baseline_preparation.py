"""Baseline 준비 범위만 검증한다. 실모델·외부 추적 전송은 사용하지 않는다."""
import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from agent.deps import SupervisorDeps
from agent.fake_llm import FastLLM, QualityLLM
from agent.graph import build_graph, research_node
from agent.mcp_client import MCPClient
from agent.retry import with_retry


class MonitoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_monitoring_returns_fixed_values_over_real_mcp(self):
        client = MCPClient(flaky_first_tool_call=False)
        await client.start()
        try:
            expected = {
                "auth-service": (78.0, 61.0, "degraded"),
                "web-01": (34.0, 48.0, "healthy"),
                "payments-service": (91.0, 55.0, "degraded"),
                "cache-service": (42.0, 89.0, "degraded"),
                "gateway-service": (30.0, 45.0, "degraded"),
                "log-service": (20.0, 35.0, "degraded"),
            }
            for system, values in expected.items():
                for attempt in range(2):
                    with self.subTest(system=system, attempt=attempt):
                        result = await client.call_tool(
                            "get_monitoring_status", {"system_name": system})
                        self.assertEqual(
                            (result["cpu_percent"], result["memory_percent"],
                             result["status"]), values)
                        self.assertEqual(result["system_name"], system)
                        self.assertTrue(result["last_checked"])
            unknown = await client.call_tool(
                "get_monitoring_status", {"system_name": "billing-service"})
            self.assertEqual(unknown["status"], "unknown")
            self.assertEqual(unknown["cpu_percent"], 0.0)
            self.assertTrue(unknown["alerts"])
        finally:
            await client.stop()

    async def test_default_client_does_not_force_first_failure(self):
        client = MCPClient()
        await client.start()
        try:
            result, retries = await with_retry(
                client.call_tool, "get_monitoring_status",
                {"system_name": "web-01"}, base_delay=0)
            self.assertEqual(result["system_name"], "web-01")
            self.assertEqual(retries, 0)
        finally:
            await client.stop()

    async def test_opt_in_failure_still_recovers_once(self):
        client = MCPClient(flaky_first_tool_call=True)
        await client.start()
        try:
            for expected_retries in (1, 0):
                result, retries = await with_retry(
                    client.call_tool, "get_monitoring_status",
                    {"system_name": "web-01"}, base_delay=0)
                self.assertEqual(result["system_name"], "web-01")
                self.assertEqual(retries, expected_retries)
        finally:
            await client.stop()

    async def test_existing_v2_graph_runs_with_real_mcp_and_fake_llms(self):
        client = MCPClient()
        await client.start()
        try:
            deps = SupervisorDeps(
                fast_llm=FastLLM("baseline-fast", 0.00015),
                quality_llm=QualityLLM("baseline-quality", 0.005),
                mcp=client, session_store=None, version="v2")
            result = await build_graph(deps).ainvoke({
                "query": "payments-service CPU 급증 진단",
                "thread_id": "baseline-smoke",
                "history": [], "events": [], "cost_usd": 0.0})
            self.assertEqual(result["route"], "incident")
            self.assertIn("payments-service", result["final"])
            self.assertEqual(result["retry_count"], 0)
            self.assertEqual(result["validation_attempts"], 2)
            self.assertTrue(any(
                c["source"] == "runbook" for c in result["collected"]))
        finally:
            await client.stop()


class ResearchAndRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_research_completes_without_artificial_sleep(self):
        state = {
            "tasks": [
                {"id": "first", "agent": "research", "kind": "monitoring",
                 "target": "web-01", "status": "pending"},
                {"id": "second", "agent": "research", "kind": "monitoring",
                 "target": "auth-service", "status": "pending"},
            ],
            "events": [],
        }

        async def tool(name, arguments):
            self.assertEqual(name, "get_monitoring_status")
            return {"system_name": arguments["system_name"]}

        deps = SimpleNamespace(mcp=SimpleNamespace(call_tool=tool))
        with patch("agent.graph.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await research_node(state, deps)
        self.assertEqual([t["status"] for t in result["tasks"]], ["done", "done"])
        self.assertEqual(
            [c["data"]["system_name"] for c in result["collected"]],
            ["web-01", "auth-service"])
        self.assertEqual(result["retry_count"], 0)
        sleep.assert_not_awaited()

    async def test_transient_errors_keep_backoff_and_recover(self):
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise asyncio.TimeoutError("temporary")
            return "recovered"

        with patch("agent.retry.asyncio.sleep", new_callable=AsyncMock) as sleep:
            value, retries = await with_retry(operation)
        self.assertEqual((value, retries, attempts), ("recovered", 2, 3))
        self.assertEqual(sleep.await_count, 2)
        self.assertAlmostEqual(sleep.await_args_list[0].args[0], 0.3)
        self.assertAlmostEqual(sleep.await_args_list[1].args[0], 0.9)

    async def test_timeout_limit_still_raises(self):
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            raise asyncio.TimeoutError("still unavailable")

        with self.assertRaises(asyncio.TimeoutError):
            await with_retry(operation, base_delay=0)
        self.assertEqual(attempts, 3)

    async def test_permanent_error_is_not_retried(self):
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            raise ValueError("invalid")

        with self.assertRaises(ValueError):
            await with_retry(operation, base_delay=0)
        self.assertEqual(attempts, 1)


if __name__ == "__main__":
    unittest.main()
