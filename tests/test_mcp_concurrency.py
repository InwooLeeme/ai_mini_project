"""실제 stdio 병렬 실행, 작업 정리, 로컬 LangSmith 추적을 검증한다."""
import asyncio
import tempfile
import unittest
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace

from langsmith import Client, traceable, tracing_context

from agent.evidence import MCPEvidenceCollector, ResearchTask
from agent.mcp_client import MCPClient
from agent.retry import with_retry


SERVER = '''import asyncio
import time
from fastmcp import FastMCP
mcp = FastMCP("parallel-test")
groups = {}
active = 0
started_event = asyncio.Event()
@mcp.tool()
async def rendezvous(group: str, expected: int) -> dict:
    global active
    entry = groups.setdefault(group, {"count": 0, "event": asyncio.Event()})
    active += 1
    started_event.set()
    started = time.perf_counter()
    concurrent = active
    entry["count"] += 1
    if entry["count"] == expected:
        entry["event"].set()
    try:
        await asyncio.wait_for(entry["event"].wait(), 2)
        await asyncio.sleep(0.02)
        return {"started": started, "ended": time.perf_counter(), "concurrent": concurrent}
    finally:
        active -= 1
@mcp.tool()
async def wait_started() -> dict:
    await started_event.wait()
    return {"active": active}
if __name__ == "__main__":
    mcp.run(show_banner=False)
'''


def response():
    return SimpleNamespace(structured_content={"ok": True}, is_error=False)


class StdioConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_requests_overlap_inside_server(self):
        # 공용 잠금으로 순차화하면 첫 요청이 rendezvous를 통과하지 못한다.
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "server.py"
            script.write_text(SERVER)
            client = MCPClient(server_scripts={"ops": script})
            await client.start()
            try:
                results = await asyncio.gather(*(
                    client.call_tool("rendezvous", {"group": "pair", "expected": 2})
                    for _ in range(2)))
                self.assertEqual(max(r["concurrent"] for r in results), 2)
                self.assertLess(max(r["started"] for r in results),
                                min(r["ended"] for r in results))
                print("stdio MCP: server concurrency=2, overlap_ms="
                      f"{(min(r['ended'] for r in results) - max(r['started'] for r in results))*1000:.2f}")
            finally:
                await client.stop()

    async def test_real_stdio_stop_cancels_inflight_request_and_can_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "server.py"
            script.write_text(SERVER)
            client = MCPClient(server_scripts={"ops": script})
            await client.start()
            pending = asyncio.create_task(client.call_tool(
                "rendezvous", {"group": "unfinished", "expected": 2}))
            try:
                self.assertEqual(await client.call_tool("wait_started", {}), {"active": 1})
                await client.stop()
                self.assertTrue(pending.cancelled())
                await client.start()
                result = await client.call_tool("rendezvous", {"group": "new", "expected": 1})
                self.assertEqual(result["concurrent"], 1)
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                await client.stop()

    async def test_real_stdio_respects_server_concurrency_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "server.py"
            script.write_text(SERVER)
            client = MCPClient(server_scripts={"ops": script}, max_concurrency=2)
            await client.start()
            try:
                results = await asyncio.gather(*(
                    client.call_tool("rendezvous", {"group": f"pair-{i // 2}", "expected": 2})
                    for i in range(6)))
                self.assertEqual(max(r["concurrent"] for r in results), 2)
                for i in (0, 2, 4):
                    self.assertLess(max(r["started"] for r in results[i:i+2]),
                                    min(r["ended"] for r in results[i:i+2]))
            finally:
                await client.stop()


class RequestLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_queued_request_keeps_owner_and_releases_capacity(self):
        client = MCPClient(max_concurrency=1)
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def call(name, arguments):
            seen.append(name)
            if name == "owner":
                entered.set()
                await release.wait()
            return response()
        client.sessions = {"ops": SimpleNamespace(call_tool=call)}
        owner = asyncio.create_task(client.call_tool("owner", {}))
        await entered.wait()
        queued = asyncio.create_task(client.call_tool("queued", {}))
        await asyncio.sleep(0)
        queued.cancel()
        try:
            with self.assertRaises(asyncio.CancelledError):
                await queued
            self.assertFalse(owner.done())
        finally:
            release.set()
            await owner
        self.assertEqual(await client.call_tool("next", {}), {"ok": True})
        self.assertEqual(seen, ["owner", "next"])

    async def test_busy_server_does_not_block_another_server(self):
        client = MCPClient(max_concurrency=1)
        entered, release = asyncio.Event(), asyncio.Event()
        async def blocked(*args):
            entered.set()
            await release.wait()
            return response()
        async def immediate(*args):
            return response()
        client.sessions = {"ops": SimpleNamespace(call_tool=blocked),
                           "docs": SimpleNamespace(call_tool=immediate)}
        first = asyncio.create_task(client.call_tool("tool", {}))
        await entered.wait()
        try:
            async with asyncio.timeout(1):
                self.assertEqual(await client.call_tool("tool", {}, "docs"), {"ok": True})
        finally:
            release.set()
            await first

    async def test_stop_cancels_requests_before_closing_sessions(self):
        client = MCPClient(max_concurrency=1)
        entered, cleaned = asyncio.Event(), asyncio.Event()
        async def blocked(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
        client.sessions = {"ops": SimpleNamespace(call_tool=blocked)}
        stack = AsyncExitStack()
        stack.callback(lambda: self.assertTrue(cleaned.is_set()))
        client._stack = stack
        task = asyncio.create_task(client.call_tool("tool", {}))
        await entered.wait()
        queued = asyncio.create_task(client.call_tool("tool", {}))
        await asyncio.sleep(0)
        try:
            async with asyncio.timeout(1):
                await client.stop()
            self.assertTrue(task.cancelled())
            self.assertTrue(queued.cancelled())
            self.assertEqual(client.sessions, {})
        finally:
            task.cancel()
            queued.cancel()
            await asyncio.gather(task, queued, return_exceptions=True)

    def test_invalid_concurrency_limits_are_rejected(self):
        for value in (0, -1, True, 1.5, "2", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MCPClient(max_concurrency=value)


class CollectionCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_waits_for_sibling_cleanup_and_preserves_error(self):
        entered, cleaned, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        failure = ValueError("original failure")
        async def call_tool(name, arguments):
            if arguments["system_name"] == "fails":
                await entered.wait()
                raise failure
            entered.set()
            try:
                await release.wait()
                return {}
            finally:
                await asyncio.sleep(0)
                cleaned.set()
        collector = MCPEvidenceCollector(SimpleNamespace(call_tool=call_tool))
        try:
            with self.assertRaises(ValueError) as caught:
                await collector.collect([ResearchTask(0, "monitoring", "fails"),
                                         ResearchTask(1, "monitoring", "slow")])
            self.assertIs(caught.exception, failure)
            self.assertTrue(cleaned.is_set())
        finally:
            release.set()
            await asyncio.wait_for(cleaned.wait(), 1)

    async def test_request_cancellation_waits_for_all_collectors(self):
        entered, cleaned = set(), set()
        ready = asyncio.Event()
        async def call_tool(name, arguments):
            target = arguments["system_name"]
            entered.add(target)
            if len(entered) == 2:
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.add(target)
        collector = MCPEvidenceCollector(SimpleNamespace(call_tool=call_tool))
        task = asyncio.create_task(collector.collect([
            ResearchTask(0, "monitoring", "a"), ResearchTask(1, "monitoring", "b")]))
        await ready.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(cleaned, {"a", "b"})


class RequestTracingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tracing = Client(api_url="http://127.0.0.1:1", api_key="offline-test-key",
                              auto_batch_tracing=False, info={"version": "0"})
        self.addCleanup(self.tracing.close)

    async def test_transport_spans_and_retry_numbers_are_recorded_locally(self):
        calls = 0
        async def transient(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary")
            return response()
        client = MCPClient()
        client.sessions = {"ops": SimpleNamespace(call_tool=transient)}
        roots = []
        with tracing_context(enabled="local", client=self.tracing):
            result = await with_retry(client.call_tool, "tool", {}, base_delay=0,
                                      langsmith_extra={"on_end": roots.append})
        self.assertEqual(result, ({"ok": True}, 1))
        attempts = roots[0].child_runs
        self.assertEqual([r.metadata.get("retry_attempt") for r in attempts], [1, 2])
        for run in attempts:
            self.assertEqual(run.metadata["mcp_server"], "ops")
            self.assertEqual(run.metadata["operation"], "call_tool")
            self.assertGreaterEqual(run.metadata["queue_ms"], 0)
            self.assertGreaterEqual(run.metadata["transport_ms"], 0)
            self.assertEqual([r.name for r in run.child_runs], ["mcp.transport"])
            self.assertIsNotNone(run.child_runs[0].end_time)
        self.assertEqual(attempts[0].metadata["timeout_stage"], "transport")
        self.assertEqual(attempts[1].metadata["outcome"], "success")

    async def test_queue_timeout_has_no_transport_span_and_does_not_cancel_owner(self):
        client = MCPClient(max_concurrency=1)
        entered, release = asyncio.Event(), asyncio.Event()
        async def blocked(*args):
            entered.set()
            await release.wait()
            return response()
        client.sessions = {"ops": SimpleNamespace(call_tool=blocked)}
        owner = asyncio.create_task(client.call_tool("tool", {}))
        await entered.wait()
        client.timeout_seconds = 0.02
        runs = []
        try:
            with tracing_context(enabled="local", client=self.tracing):
                with self.assertRaises(TimeoutError):
                    await client.call_tool("tool", {}, langsmith_extra={"on_end": runs.append})
            self.assertFalse(owner.done())
            self.assertEqual(runs[0].metadata["timeout_stage"], "queue")
            self.assertGreater(runs[0].metadata["queue_ms"], 0)
            self.assertEqual(runs[0].metadata["transport_ms"], 0)
            self.assertEqual(runs[0].child_runs, [])
        finally:
            release.set()
            await owner

    async def test_queue_delay_is_outside_transport_span(self):
        client = MCPClient(max_concurrency=1)
        entered, release = asyncio.Event(), asyncio.Event()
        async def call(*args):
            if not entered.is_set():
                entered.set()
                await release.wait()
            return response()
        client.sessions = {"ops": SimpleNamespace(call_tool=call)}
        roots = []
        @traceable(name="requests")
        async def requests():
            first = asyncio.create_task(client.call_tool("tool", {}))
            await entered.wait()
            second = asyncio.create_task(client.call_tool("tool", {}))
            await asyncio.sleep(0.02)
            release.set()
            return await asyncio.gather(first, second)
        with tracing_context(enabled="local", client=self.tracing):
            await requests(langsmith_extra={"on_end": roots.append})
        first, second = roots[0].child_runs
        self.assertGreater(second.metadata["queue_ms"], 0)
        self.assertGreaterEqual(second.child_runs[0].start_time, first.child_runs[0].end_time)
        self.assertLess(second.start_time, first.child_runs[0].end_time)
