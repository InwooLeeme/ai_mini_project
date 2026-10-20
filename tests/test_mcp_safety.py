"""MCP transport boundary tests; no LLM or external tracing is used."""
import asyncio
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from agent.mcp_client import MCPClient
from agent.retry import with_retry

STATUS = {"system_name": "web-01", "status": "healthy", "cpu_percent": 34.0,
          "memory_percent": 48.0, "last_checked": "2026-10-20T10:20:30.123456",
          "alerts": []}


def result(value=None, **kwargs):
    return NS(content=[NS(type="text", text=json.dumps(STATUS if value is None else value))],
              is_error=False, **kwargs)


class BoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = MCPClient()
        self.session = NS(call_tool=AsyncMock(return_value=result()),
                          read_resource=AsyncMock(return_value=NS(contents=[NS(text="runbook")])),
                          get_prompt=AsyncMock(return_value=NS(messages=[NS(content=NS(type="text", text="prompt"))])),
                          list_prompts=AsyncMock(return_value=NS(prompts=[])),
                          list_resource_templates=AsyncMock(return_value=NS(resource_templates=[])))
        self.client.sessions = {"ops": self.session}

    async def assert_error(self, code, fn, *args, **kwargs):
        with self.assertRaises(ValueError) as caught:
            await fn(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertNotIsInstance(caught.exception, TimeoutError)
        return caught.exception

    async def test_invalid_monitoring_arguments_never_reach_server(self):
        for args in (None, [], {}, {"system_name": ""}, {"system_name": " "},
                     {"system_name": 1}, {"system_name": True},
                     {"system_name": "web-01\n"}, {"system_name": "web-01", "extra": 1}):
            with self.subTest(args=args):
                await self.assert_error("invalid_input", self.client.call_tool, "get_monitoring_status", args)
        self.session.call_tool.assert_not_awaited()

    async def test_invalid_identifiers_and_arguments_never_reach_server(self):
        cases = [(self.client.call_tool, ("", {})), (self.client.call_tool, ("bad name", {})),
                 (self.client.call_tool, ("tool", [])), (self.client.call_tool, ("tool", {}, " ")),
                 (self.client.read_resource, ("not-a-uri", "ops")),
                 (self.client.read_resource, (None, "ops")),
                 (self.client.get_prompt, ("", {})), (self.client.get_prompt, ("prompt", None)),
                 (self.client.find_prompt, ("",)), (self.client.find_resource_template, ("",)),
                 (self.client.list_prompts, ("",)), (self.client.list_resource_templates, (None,))]
        for fn, args in cases:
            with self.subTest(fn=fn.__name__, args=args):
                await self.assert_error("invalid_input", fn, *args)
        for method in vars(self.session).values(): method.assert_not_awaited()

    async def test_non_json_tool_and_non_string_prompt_arguments_are_not_sent(self):
        circular = {}
        circular["self"] = circular
        for arguments in ({1: "value"}, {"nested": {1: "value"}}, {"bad": object()},
                          {"bad": float("nan")}, circular):
            with self.subTest(arguments=arguments):
                await self.assert_error("invalid_input", self.client.call_tool, "tool", arguments)
        for arguments in ({1: "value"}, {"system_name": 123}, {"nested": {}},
                          {"values": []}, {"valid": "value", "bad": None}):
            with self.subTest(arguments=arguments):
                await self.assert_error("invalid_input", self.client.get_prompt, "prompt", arguments)
        for arguments in ({}, {"system_name": "web-01"},
                          {"system_name": "web-01", "symptom": " "},
                          {"system_name": "web-01", "symptom": "CPU", "extra": "value"}):
            await self.assert_error("invalid_input", self.client.get_prompt, "incident_analysis_prompt", arguments)
        self.session.call_tool.assert_not_awaited()
        self.session.get_prompt.assert_not_awaited()

    async def test_missing_and_unstarted_sessions_have_clear_errors(self):
        await self.assert_error("unknown_server", self.client.call_tool, "tool", {}, "missing")
        for fn, args in ((MCPClient().call_tool, ("tool", {})),
                         (MCPClient().find_prompt, ("prompt",)),
                         (MCPClient().find_resource_template, ("runbook",))):
            await self.assert_error("not_started", fn, *args)

    async def test_error_flag_precedes_content_parsing_and_is_not_retried(self):
        for field in ("is_error", "isError"):
            self.session.call_tool.return_value = NS(**{field: True}, content=[])
            error = await self.assert_error("server_error", with_retry,
                                           self.client.call_tool, "tool", {}, base_delay=0)
            self.assertNotIn("content", str(error))
        self.assertEqual(self.session.call_tool.await_count, 2)

    async def test_malformed_tool_results_are_permanent_and_safe(self):
        invalid = [NS(), NS(content=[]), NS(content=[NS(type="image", text="{}")]),
                   NS(content=[NS(type="text", text="")]),
                   NS(content=[NS(type="text", text="secret invalid JSON")]),
                   result([]), result("text"), result(12), result(True),
                   result(structured_content=[])]
        for response in invalid:
            with self.subTest(response=response):
                self.session.call_tool.return_value = response
                error = await self.assert_error("invalid_response", with_retry,
                                               self.client.call_tool, "tool", {}, base_delay=0)
                self.assertNotIn("secret", str(error))
        self.assertEqual(self.session.call_tool.await_count, len(invalid))

    async def test_monitoring_contract_rejects_mismatch_missing_and_invalid_fields(self):
        cases = [dict(STATUS, system_name="other"), dict(STATUS, status="ok"),
                 dict(STATUS, status=[]), dict(STATUS, last_checked="yesterday"),
                 dict(STATUS, last_checked="2026-10-20"), dict(STATUS, alerts="alarm"),
                 dict(STATUS, alerts=[1])]
        cases += [{key: value for key, value in STATUS.items() if key != missing} for missing in STATUS]
        cases += [dict(STATUS, **{key: value}) for key in ("cpu_percent", "memory_percent")
                  for value in (True, "34", -1, 101, float("nan"), float("inf"))]
        for value in cases:
            with self.subTest(value=value):
                self.session.call_tool.return_value = result(value)
                await self.assert_error("invalid_response", self.client.call_tool,
                                        "get_monitoring_status", {"system_name": "web-01"})

    async def test_structured_result_and_unknown_status_remain_valid(self):
        unknown = dict(STATUS, system_name="billing-service", status="unknown",
                       cpu_percent=0.0, memory_percent=0.0, alerts=["System not found"])
        for field in ("structured_content", "structuredContent"):
            self.session.call_tool.return_value = NS(content=[], **{field: unknown})
            self.assertEqual(await self.client.call_tool("get_monitoring_status",
                                                        {"system_name": "billing-service"}), unknown)

    async def test_resource_prompt_and_discovery_validate_shapes(self):
        cases = [("read_resource", ("runbook://incident/cpu", "ops"), [NS(), NS(contents=[]), NS(contents=[NS(blob="abc")])]),
                 ("get_prompt", ("prompt", {}), [NS(messages=[]), NS(messages=[NS(content=NS(type="image", text="bad"))])]),
                 ("list_prompts", ("ops",), [NS(prompts=None), NS(prompts=[NS()])]),
                 ("list_resource_templates", ("ops",), [NS(resource_templates="bad"), NS(resource_templates=[NS(name="x")])])]
        for name, args, values in cases:
            for value in values:
                with self.subTest(name=name, value=value):
                    getattr(self.session, name).return_value = value
                    await self.assert_error("invalid_response", getattr(self.client, name), *args)

    async def test_multiple_text_blocks_are_preserved(self):
        self.session.read_resource.return_value = NS(contents=[NS(text="first"), NS(text="second")])
        self.session.get_prompt.return_value = NS(messages=[NS(content=NS(type="text", text="first")), NS(content=NS(type="text", text="second"))])
        self.assertEqual(await self.client.read_resource("docs://api-reference/auth", "ops"), "first\nsecond")
        self.assertEqual(await self.client.get_prompt("prompt", {}), "first\nsecond")

    async def test_sdk_error_is_sanitized_and_not_retried(self):
        self.session.call_tool.side_effect = RuntimeError("secret server internals")
        error = await self.assert_error("request_failed", with_retry, self.client.call_tool, "tool", {}, base_delay=0)
        self.assertNotIn("secret", str(error))
        self.assertEqual(self.session.call_tool.await_count, 1)


class TimeoutTests(unittest.IsolatedAsyncioTestCase):
    def test_timeout_configuration_requires_positive_finite_number(self):
        for value in (0, -1, True, "10", None, float("inf"), float("nan"), 10 ** 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MCPClient(timeout_seconds=value)

    async def test_all_transport_operations_time_out_and_unlock(self):
        async def blocked(*args): await asyncio.Event().wait()
        for method, args in (("call_tool", ("tool", {})), ("read_resource", ("docs://api/auth", "ops")),
                             ("get_prompt", ("prompt", {})), ("list_prompts", ("ops",)),
                             ("list_resource_templates", ("ops",)), ("find_prompt", ("prompt",)),
                             ("find_resource_template", ("resource",))):
            with self.subTest(method=method):
                client = MCPClient(timeout_seconds=0.02)
                client.sessions = {"ops": NS(**{key: blocked for key in
                    ("call_tool", "read_resource", "get_prompt", "list_prompts", "list_resource_templates")})}
                with self.assertRaises(TimeoutError): await getattr(client, method)(*args)
                self.assertFalse(client._lock.locked())
                client.sessions["ops"].call_tool = AsyncMock(return_value=result({"recovered": True}))
                self.assertEqual(await client.call_tool("tool", {}), {"recovered": True})

    async def test_waiting_lock_timeout_does_not_cancel_owner(self):
        client = MCPClient(timeout_seconds=0.02)
        client.sessions = {"ops": NS(call_tool=AsyncMock(return_value=result({"ok": True})))}
        release, acquired = asyncio.Event(), asyncio.Event()
        async def owner():
            async with client._lock:
                acquired.set()
                await release.wait()
        task = asyncio.create_task(owner())
        await acquired.wait()
        try:
            with self.assertRaises(TimeoutError): await client.call_tool("tool", {})
            self.assertFalse(task.done())
            client.sessions["ops"].call_tool.assert_not_awaited()
        finally:
            release.set()
            await task
        self.assertEqual(await client.call_tool("tool", {}), {"ok": True})

    async def test_external_cancellation_is_preserved_and_releases_lock(self):
        entered = asyncio.Event()
        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()
        client = MCPClient(timeout_seconds=1)
        client.sessions = {"ops": NS(call_tool=blocked)}
        task = asyncio.create_task(client.call_tool("tool", {}))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertFalse(client._lock.locked())
        client.sessions["ops"].call_tool = AsyncMock(return_value=result({"ok": True}))
        self.assertEqual(await client.call_tool("tool", {}), {"ok": True})

    async def test_actual_timeout_retry_stops_after_three_calls(self):
        calls = 0
        async def blocked(*args):
            nonlocal calls
            calls += 1
            await asyncio.Event().wait()
        client = MCPClient(timeout_seconds=0.01)
        client.sessions = {"ops": NS(call_tool=blocked)}
        with self.assertRaises(TimeoutError):
            await with_retry(client.call_tool, "tool", {}, base_delay=0)
        self.assertEqual(calls, 3)

    async def test_discovery_has_one_budget_across_servers(self):
        async def slow_list():
            await asyncio.sleep(0.03)
            return NS(prompts=[])
        client = MCPClient(timeout_seconds=0.05)
        client.sessions = {name: NS(list_prompts=slow_list) for name in ("ops", "docs")}
        with self.assertRaises(TimeoutError): await client.find_prompt("missing")
        self.assertFalse(client._lock.locked())

    async def test_sdk_timeout_remains_retryable(self):
        from mcp.shared import exceptions
        from mcp.types import ErrorData, REQUEST_TIMEOUT
        if hasattr(exceptions, "MCPError"):
            error = exceptions.MCPError(REQUEST_TIMEOUT, "timed out")
        else:
            error = exceptions.McpError(ErrorData(code=REQUEST_TIMEOUT, message="timed out"))
        client = MCPClient()
        client.sessions = {"ops": NS(call_tool=AsyncMock(side_effect=[error, result({"ok": True})]))}
        self.assertEqual(await with_retry(client.call_tool, "tool", {}, base_delay=0), ({"ok": True}, 1))


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_start_closes_opened_resources_and_can_restart(self):
        active = []
        fail = True
        @asynccontextmanager
        async def transport(params):
            active.append(params.args[0])
            try: yield object(), object()
            finally: active.remove(params.args[0])
        @asynccontextmanager
        async def session(*args):
            async def initialize():
                if fail and len(active) == 2: raise RuntimeError("startup secret")
            yield NS(initialize=initialize)
        client = MCPClient(server_scripts={"ops": Path("ops.py"), "docs": Path("docs.py")})
        with patch("agent.mcp_client.stdio_client", transport), patch("agent.mcp_client.ClientSession", session):
            with self.assertRaises(ValueError) as caught: await client.start()
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(active, [])
            self.assertEqual(client.sessions, {})
            self.assertIsNone(client._stack)
            await client.stop()
            fail = False
            await client.start()
            await client.start()
            self.assertEqual(len(active), 2)
            await client.stop()
            await client.stop()
            self.assertEqual(active, [])

    async def test_start_timeout_cleans_resources_in_same_task(self):
        active = []
        @asynccontextmanager
        async def transport(params):
            active.append(asyncio.current_task())
            try: yield object(), object()
            finally:
                self.assertIs(active.pop(), asyncio.current_task())
        @asynccontextmanager
        async def session(*args):
            async def initialize(): await asyncio.Event().wait()
            yield NS(initialize=initialize)
        client = MCPClient(timeout_seconds=0.02)
        with patch("agent.mcp_client.stdio_client", transport), patch("agent.mcp_client.ClientSession", session):
            with self.assertRaises(TimeoutError): await client.start()
        self.assertEqual(active, [])
        self.assertEqual(client.sessions, {})
        await client.stop()

    async def test_cancelled_start_closes_resources_and_propagates_cancellation(self):
        entered, active = asyncio.Event(), []
        @asynccontextmanager
        async def transport(params):
            active.append(asyncio.current_task())
            try: yield object(), object()
            finally: self.assertIs(active.pop(), asyncio.current_task())
        @asynccontextmanager
        async def session(*args):
            async def initialize():
                entered.set()
                await asyncio.Event().wait()
            yield NS(initialize=initialize)
        client = MCPClient()
        with patch("agent.mcp_client.stdio_client", transport), patch("agent.mcp_client.ClientSession", session):
            task = asyncio.create_task(client.start())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(active, [])
        self.assertEqual(client.sessions, {})
        self.assertIsNone(client._stack)
        await client.stop()

    async def test_start_and_cleanup_failure_does_not_claim_connections_closed(self):
        @asynccontextmanager
        async def transport(params):
            try: yield object(), object()
            finally: raise RuntimeError("cleanup secret")
        @asynccontextmanager
        async def session(*args):
            yield NS(initialize=AsyncMock(side_effect=RuntimeError("startup secret")))
        client = MCPClient(server_scripts={"ops": Path("ops.py")})
        with patch("agent.mcp_client.stdio_client", transport), patch("agent.mcp_client.ClientSession", session):
            with self.assertRaises(ValueError) as caught: await client.start()
            self.assertEqual(caught.exception.code, "startup_failed")
            self.assertNotIn("secret", str(caught.exception))
            self.assertNotIn("were closed", str(caught.exception))
        self.assertEqual(client.sessions, {})
        self.assertIsNone(client._stack)
        await client.stop()

    async def test_failed_cleanup_is_sanitized_and_clears_state(self):
        @asynccontextmanager
        async def transport(params):
            try: yield object(), object()
            finally: raise RuntimeError("cleanup secret")
        @asynccontextmanager
        async def session(*args):
            yield NS(initialize=AsyncMock())
        client = MCPClient(server_scripts={"ops": Path("ops.py")})
        with patch("agent.mcp_client.stdio_client", transport), patch("agent.mcp_client.ClientSession", session):
            await client.start()
            with self.assertRaises(ValueError) as caught: await client.stop()
            self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(client.sessions, {})
        self.assertIsNone(client._stack)
        await client.stop()


class StdioTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_discovery_resource_prompt_and_unknown_contract(self):
        client = MCPClient()
        await client.start()
        try:
            self.assertEqual(await client.find_resource_template("get_api_doc"),
                             ("docs", "docs://api-reference/{section}"))
            self.assertIn("토큰", await client.read_resource("docs://api-reference/auth", "docs"))
            server = await client.find_prompt("incident_analysis_prompt")
            self.assertEqual(server, "ops")
            self.assertIn("web-01", await client.get_prompt("incident_analysis_prompt",
                          {"system_name": "web-01", "symptom": "CPU"}, server))
            status = await client.call_tool("get_monitoring_status", {"system_name": "not-registered"})
            self.assertEqual((status["status"], status["cpu_percent"], status["memory_percent"]), ("unknown", 0.0, 0.0))
        finally: await client.stop()

    async def test_real_partial_start_failure_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "exit.py"
            script.write_text("raise SystemExit(0)\n")
            client = MCPClient(server_scripts={"ops": Path(__file__).resolve().parents[1] / "mcp_server.py", "broken": script})
            with self.assertRaises(ValueError) as caught: await client.start()
            self.assertEqual(caught.exception.code, "startup_failed")
            self.assertEqual(client.sessions, {})
            self.assertIsNone(client._stack)
            await client.stop()

    async def test_real_start_timeout_terminates_children_and_can_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(__file__).resolve().parents[1] / "mcp_server.py"
            for name in ("ops", "docs"):
                body = (f"import os\nfrom pathlib import Path\nPath({str(root / (name + '.pid'))!r}).write_text(str(os.getpid()))\n")
                body += (f"import runpy\nrunpy.run_path({str(source)!r}, run_name='__main__')\n" if name == "ops"
                         else "import time\ntime.sleep(10)\n")
                (root / (name + ".py")).write_text(body)
            client = MCPClient(server_scripts={name: root / (name + ".py") for name in ("ops", "docs")}, timeout_seconds=1)
            with self.assertRaises(TimeoutError): await client.start()
            self.assertEqual(client.sessions, {})
            self.assertIsNone(client._stack)
            for name in ("ops", "docs"):
                pid = int((root / (name + ".pid")).read_text())
                with self.assertRaises(ProcessLookupError): os.kill(pid, 0)
            await client.stop()
            (root / "docs.py").write_text(f"import runpy\nrunpy.run_path({str(source)!r}, run_name='__main__')\n")
            client.timeout_seconds = 5
            await client.start()
            try:
                status = await client.call_tool("get_monitoring_status", {"system_name": "web-01"})
                self.assertEqual(status["cpu_percent"], 34.0)
            finally: await client.stop()

    async def test_real_stdio_timeout_then_next_request_succeeds(self):
        source = '''import asyncio\nfrom fastmcp import FastMCP\nmcp = FastMCP("timeout-test")\n@mcp.tool()\nasync def delayed(delay: float) -> dict:\n    await asyncio.sleep(delay)\n    return {"ok": True}\nif __name__ == "__main__":\n    mcp.run(show_banner=False)\n'''
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "server.py"
            script.write_text(source)
            client = MCPClient(server_scripts={"ops": script}, timeout_seconds=5)
            await client.start()
            try:
                client.timeout_seconds = 0.1
                with self.assertRaises(TimeoutError): await client.call_tool("delayed", {"delay": 2})
                self.assertEqual(await client.call_tool("delayed", {"delay": 0}), {"ok": True})
            finally: await client.stop()


if __name__ == "__main__": unittest.main()
