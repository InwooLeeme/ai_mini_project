"""Shared stdio MCP sessions with bounded calls and validated response contracts.

The lifecycle owner must start and stop the client in the same task because the
SDK's stdio/session contexts own AnyIO cancel scopes. Each request uses a shared
lock, and its timeout includes waiting for that lock. Retries remain in retry.py.
"""
import asyncio
import json
import math
import os
import re
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from numbers import Real
from pathlib import Path
from urllib.parse import urlsplit

from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared import exceptions as mcp_exceptions
from mcp.types import REQUEST_TIMEOUT

from agent.mcp_errors import MCPCallError

SERVER_DIR = Path(__file__).parent.parent
DEFAULT_SERVER_SCRIPTS = {
    "ops": SERVER_DIR / "mcp_server.py",
    "docs": SERVER_DIR / "mcp_server_docs.py",
}
_SDK_ERROR = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError
_MISSING = object()


def _attr(obj, *names, default=_MISSING):
    """Read SDK 1.x camelCase or 2.x snake_case fields without payload dumps."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not _MISSING:
        return default
    raise MCPCallError("invalid_response", "MCP response is missing a required field")


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[\w.-]+", value):
        raise MCPCallError("invalid_input", f"{label} must be a nonempty identifier")


def _arguments(arguments, *, prompt=False):
    if not isinstance(arguments, dict):
        raise MCPCallError("invalid_input", "arguments must be a dictionary")
    try:
        # Reject cycles, unsupported objects and NaN/Infinity before SDK I/O.
        json.dumps(arguments, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise MCPCallError("invalid_input", "arguments must contain valid JSON values") from None
    pending = [arguments]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise MCPCallError("invalid_input", "argument object keys must be strings")
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    if prompt and any(not isinstance(value, str) for value in arguments.values()):
        raise MCPCallError("invalid_input", "prompt argument values must be strings")


def _check_error(result):
    for name in ("is_error", "isError"):
        flag = _attr(result, name, default=False)
        if flag is True:
            raise MCPCallError("server_error", "MCP server reported an operation error")
        if not isinstance(flag, bool):
            raise MCPCallError("invalid_response", "MCP response has an invalid error flag")


def _list(result, *names, allow_empty=True):
    _check_error(result)
    values = _attr(result, *names)
    if not isinstance(values, list) or (not allow_empty and not values):
        raise MCPCallError("invalid_response", "MCP response must contain a valid list")
    return values


def _text(block, *, typed=False):
    if typed and _attr(block, "type", default=None) != "text":
        raise MCPCallError("invalid_response", "MCP response requires text content")
    text = _attr(block, "text")
    if not isinstance(text, str) or not text.strip():
        raise MCPCallError("invalid_response", "MCP response contains empty or invalid text")
    return text


def _tool_object(result):
    _check_error(result)
    structured = _attr(result, "structured_content", "structuredContent", default=None)
    if structured is not None:
        value = structured
    else:
        content = _list(result, "content", allow_empty=False)
        text = "\n".join(_text(block, typed=True) for block in content)
        try:
            def reject_constant(constant):
                raise ValueError("Non-finite JSON constant")
            value = json.loads(text, parse_constant=reject_constant)
        except (ValueError, RecursionError):
            raise MCPCallError("invalid_response", "MCP tool returned invalid JSON") from None
    if not isinstance(value, dict):
        raise MCPCallError("invalid_response", "MCP tool must return a JSON object")
    return value


def _monitoring_status(value, system_name):
    valid = (value.get("system_name") == system_name
             and isinstance(value.get("status"), str)
             and value["status"] in {"healthy", "degraded", "down", "unknown"})
    for key in ("cpu_percent", "memory_percent"):
        metric = value.get(key)
        valid = valid and (isinstance(metric, Real) and not isinstance(metric, bool)
                           and 0 <= metric <= 100 and math.isfinite(metric))
    timestamp = value.get("last_checked")
    try:
        if not isinstance(timestamp, str) or not any(sep in timestamp for sep in ("T", "t", " ")):
            raise ValueError
        datetime.fromisoformat(timestamp)
    except ValueError:
        valid = False
    alerts = value.get("alerts")
    valid = valid and isinstance(alerts, list) and all(isinstance(item, str) for item in alerts)
    if not valid:
        raise MCPCallError("invalid_response", "MCP monitoring result violates the requested system/status/metrics contract")
    return value


class MCPClient:
    def __init__(self, flaky_first_tool_call: bool = False,
                 server_scripts: dict[str, Path] | None = None, *,
                 timeout_seconds: float = 10.0):
        try:
            if not isinstance(timeout_seconds, Real) or isinstance(timeout_seconds, bool):
                raise ValueError
            timeout_seconds = float(timeout_seconds)
            if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
                raise ValueError
        except (ValueError, OverflowError):
            raise MCPCallError("invalid_input", "timeout_seconds must be a positive finite number") from None
        self.timeout_seconds = timeout_seconds
        self._server_scripts = dict(DEFAULT_SERVER_SCRIPTS if server_scripts is None else server_scripts)
        for server in self._server_scripts:
            _identifier(server, "server")
        self._stack: AsyncExitStack | None = None
        self.sessions: dict[str, ClientSession] = {}
        self._lock = asyncio.Lock()
        self._flaky_pending = flaky_first_tool_call

    async def start(self) -> None:
        # Keep SDK contexts and their cleanup in the caller's task (no wait_for).
        async with asyncio.timeout(self.timeout_seconds):
            async with self._lock:
                if self._stack is not None:
                    return
                stack = AsyncExitStack()
                sessions = {}
                try:
                    for server_name, script in self._server_scripts.items():
                        params = StdioServerParameters(
                            command=sys.executable, args=[str(script)],
                            env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"})
                        read, write = await stack.enter_async_context(stdio_client(params))
                        session = await stack.enter_async_context(ClientSession(read, write))
                        await session.initialize()
                        sessions[server_name] = session
                except BaseException as error:
                    try:
                        await stack.aclose()
                    except Exception:
                        # ExitStack still runs every exit callback; preserve the
                        # original failure/cancellation without exposing payloads.
                        pass
                    finally:
                        self.sessions = {}
                        self._stack = None
                    if isinstance(error, (asyncio.CancelledError, TimeoutError)):
                        raise
                    if isinstance(error, _SDK_ERROR) and error.error.code == REQUEST_TIMEOUT:
                        raise TimeoutError("MCP initialization timed out") from None
                    if not isinstance(error, Exception):
                        raise
                    raise MCPCallError("startup_failed", "MCP initialization failed; connection cleanup was attempted") from None
                self.sessions = sessions
                self._stack = stack

    async def stop(self) -> None:
        async with asyncio.timeout(self.timeout_seconds):
            async with self._lock:
                stack, self._stack = self._stack, None
                self.sessions = {}
                if stack is not None:
                    try:
                        await stack.aclose()
                    except TimeoutError:
                        raise
                    except Exception:
                        raise MCPCallError("shutdown_failed", "MCP connection cleanup failed") from None

    def _session(self, server):
        _identifier(server, "server")
        if not self.sessions:
            raise MCPCallError("not_started", "MCP client has no ready sessions; call start first")
        if server not in self.sessions:
            raise MCPCallError("unknown_server", "Requested MCP server is not connected")
        return self.sessions[server]

    async def _request(self, server, method, *args):
        self._session(server)
        async with asyncio.timeout(self.timeout_seconds):
            async with self._lock:
                session = self._session(server)
                try:
                    return await getattr(session, method)(*args)
                except (TimeoutError, MCPCallError):
                    raise
                except _SDK_ERROR as error:
                    if error.error.code == REQUEST_TIMEOUT:
                        raise TimeoutError("MCP request timed out") from None
                    raise MCPCallError("request_failed", "MCP server rejected or could not complete the request") from None
                except Exception:
                    raise MCPCallError("request_failed", "MCP request failed or returned an invalid protocol response") from None

    @traceable(run_type="tool", name="mcp.call_tool")
    async def call_tool(self, name: str, arguments: dict, server: str = "ops") -> dict:
        _identifier(name, "tool name")
        _arguments(arguments)
        if name == "get_monitoring_status":
            if set(arguments) != {"system_name"}:
                raise MCPCallError("invalid_input", "get_monitoring_status requires only system_name")
            _identifier(arguments["system_name"], "system_name")
        self._session(server)
        if self._flaky_pending and name == "get_monitoring_status":
            self._flaky_pending = False
            raise TimeoutError("simulated transient timeout (Wave 9 demo)")
        value = _tool_object(await self._request(server, "call_tool", name, arguments))
        if name == "get_monitoring_status":
            return _monitoring_status(value, arguments["system_name"])
        return value

    @traceable(run_type="tool", name="mcp.list_resource_templates")
    async def list_resource_templates(self, server: str) -> list:
        values = _list(await self._request(server, "list_resource_templates"),
                       "resourceTemplates", "resource_templates")
        for item in values:
            if not isinstance(_attr(item, "name"), str) or not item.name.strip():
                raise MCPCallError("invalid_response", "MCP resource template has an invalid name")
            template = _attr(item, "uriTemplate", "uri_template")
            if not isinstance(template, str) or not template.strip():
                raise MCPCallError("invalid_response", "MCP resource template has an invalid URI template")
        return values

    def _servers(self):
        if not self.sessions:
            raise MCPCallError("not_started", "MCP client has no ready sessions; call start first")
        return tuple(self.sessions)

    @traceable(run_type="tool", name="mcp.find_resource_template")
    async def find_resource_template(self, template_name: str) -> tuple[str, str]:
        _identifier(template_name, "resource template name")
        async with asyncio.timeout(self.timeout_seconds):
            for server_name in self._servers():
                templates = await self.list_resource_templates(server_name)
                matched = next((item for item in templates if item.name == template_name), None)
                if matched is not None:
                    return server_name, _attr(matched, "uriTemplate", "uri_template")
        raise MCPCallError("not_found", "MCP resource template not found on connected servers")

    @traceable(run_type="tool", name="mcp.read_resource")
    async def read_resource(self, uri: str, server: str) -> str:
        if not isinstance(uri, str) or not uri or any(char.isspace() or not char.isprintable() for char in uri):
            raise MCPCallError("invalid_input", "resource URI must be a nonempty absolute URI")
        try:
            if not urlsplit(uri).scheme:
                raise ValueError
        except ValueError:
            raise MCPCallError("invalid_input", "resource URI must be a valid absolute URI") from None
        values = _list(await self._request(server, "read_resource", uri), "contents", allow_empty=False)
        return "\n".join(_text(item) for item in values)

    @traceable(run_type="tool", name="mcp.list_prompts")
    async def list_prompts(self, server: str) -> list:
        values = _list(await self._request(server, "list_prompts"), "prompts")
        for item in values:
            if not isinstance(_attr(item, "name"), str) or not item.name.strip():
                raise MCPCallError("invalid_response", "MCP prompt has an invalid name")
        return values

    @traceable(run_type="tool", name="mcp.find_prompt")
    async def find_prompt(self, prompt_name: str) -> str:
        _identifier(prompt_name, "prompt name")
        async with asyncio.timeout(self.timeout_seconds):
            for server_name in self._servers():
                prompts = await self.list_prompts(server_name)
                if any(item.name == prompt_name for item in prompts):
                    return server_name
        raise MCPCallError("not_found", "MCP prompt not found on connected servers")

    @traceable(run_type="tool", name="mcp.get_prompt")
    async def get_prompt(self, name: str, arguments: dict, server: str = "ops") -> str:
        _identifier(name, "prompt name")
        _arguments(arguments, prompt=True)
        if name == "incident_analysis_prompt":
            if set(arguments) != {"system_name", "symptom"} or not arguments.get("symptom", "").strip():
                raise MCPCallError("invalid_input", "incident_analysis_prompt requires system_name and a nonempty symptom")
            _identifier(arguments["system_name"], "system_name")
        values = _list(await self._request(server, "get_prompt", name, arguments), "messages", allow_empty=False)
        return "\n".join(_text(_attr(item, "content"), typed=True) for item in values)
