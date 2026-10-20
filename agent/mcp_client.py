"""호출 시간과 응답 형식을 검증하는 공용 stdio MCP 클라이언트.

SDK의 stdio/세션 컨텍스트가 AnyIO 취소 범위를 소유하므로, 연결을 시작한
태스크에서 연결을 종료해야 한다. 요청은 공용 잠금을 사용하며 잠금 대기도
제한 시간에 포함한다. 재시도 정책은 retry.py에서 별도로 관리한다.
"""
import asyncio
import json
import logging
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
logger = logging.getLogger(__name__)


def _log_failure(server, operation, error, *, cleanup_failed=False):
    """진단용 메타데이터만 기록하고 요청·응답 원문과 예외 메시지는 남기지 않는다."""
    code = error.error.code if isinstance(error, _SDK_ERROR) else None
    metadata = {
        "mcp_server": server,
        "operation": operation,
        "error_type": type(error).__name__,
        "sdk_error_code": code if isinstance(code, int) and not isinstance(code, bool) else None,
        "cleanup_failed": cleanup_failed,
    }
    logger.warning(
        "MCP failure server=%(mcp_server)s operation=%(operation)s "
        "error_type=%(error_type)s sdk_error_code=%(sdk_error_code)s "
        "cleanup_failed=%(cleanup_failed)s", metadata, extra=metadata,
    )


def _attr(obj, *names, default=_MISSING):
    """SDK 1.x의 camelCase와 2.x의 snake_case 필드를 호환해서 읽는다."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not _MISSING:
        return default
    raise MCPCallError("invalid_response", "MCP response is missing a required field")


def _identifier(value, label):
    """서버·도구 이름 등 식별자가 비어 있지 않고 허용된 문자로 구성됐는지 검사한다."""
    if not isinstance(value, str) or not re.fullmatch(r"[\w.-]+", value):
        raise MCPCallError("invalid_input", f"{label} must be a nonempty identifier")


def _arguments(arguments, *, prompt=False):
    """입력을 검증하고 복사해 비동기 대기 중 원본이 바뀌어도 전송값을 유지한다."""
    if not isinstance(arguments, dict):
        raise MCPCallError("invalid_input", "arguments must be a dictionary")
    if prompt:
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in arguments.items()):
            raise MCPCallError("invalid_input", "prompt argument keys and values must be strings")
        # 키와 값이 모두 불변 문자열이므로 프롬프트 인자는 얕은 복사로 충분하다.
        return dict(arguments)
    try:
        # 서버 호출 전에 순환 참조, 직렬화 불가 객체, NaN·무한대를 차단한다.
        encoded = json.dumps(arguments, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise MCPCallError("invalid_input", "arguments must contain valid JSON values") from None
    # JSON 직렬화는 숫자 키를 문자열로 바꿀 수 있으므로 원본의 중첩 키도 검사한다.
    pending = [arguments]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise MCPCallError("invalid_input", "argument object keys must be strings")
            pending.extend(value.values())
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    # 다시 역직렬화하면 중첩 객체까지 복사되고 실제 전송 형식과도 일치한다.
    return json.loads(encoded)


def _check_error(result):
    """서버가 실패로 표시한 응답을 정상 데이터로 해석하지 않도록 차단한다."""
    for name in ("is_error", "isError"):
        flag = _attr(result, name, default=False)
        if flag is True:
            raise MCPCallError("server_error", "MCP server reported an operation error")
        if not isinstance(flag, bool):
            raise MCPCallError("invalid_response", "MCP response has an invalid error flag")


def _list(result, *names, allow_empty=True):
    """응답 필드가 목록인지 확인하고, 필요한 경우 빈 목록도 거부한다."""
    values = _attr(result, *names)
    if not isinstance(values, list) or (not allow_empty and not values):
        raise MCPCallError("invalid_response", "MCP response must contain a valid list")
    return values


def _text(block, *, typed=False):
    """내용 블록에서 비어 있지 않은 텍스트를 읽고 필요하면 블록 유형도 검사한다."""
    if typed and _attr(block, "type", default=None) != "text":
        raise MCPCallError("invalid_response", "MCP response requires text content")
    text = _attr(block, "text")
    if not isinstance(text, str) or not text.strip():
        raise MCPCallError("invalid_response", "MCP response contains empty or invalid text")
    return text


def _tool_object(result):
    """구조화된 응답을 우선 읽고, 없으면 텍스트를 JSON 객체로 해석한다."""
    structured = _attr(result, "structured_content", "structuredContent", default=None)
    if structured is not None:
        value = structured
    else:
        content = _list(result, "content", allow_empty=False)
        text = "\n".join(_text(block, typed=True) for block in content)
        try:
            def reject_constant(constant):
                """JSON 텍스트에 포함된 NaN·무한대 상수를 거부한다."""
                raise ValueError("Non-finite JSON constant")
            value = json.loads(text, parse_constant=reject_constant)
        except (ValueError, RecursionError):
            raise MCPCallError("invalid_response", "MCP tool returned invalid JSON") from None
    if not isinstance(value, dict):
        raise MCPCallError("invalid_response", "MCP tool must return a JSON object")
    return value


def _monitoring_status(value, system_name):
    """조회 대상과 상태·지표·확인 시각·알림이 모니터링 응답 규약에 맞는지 검사한다."""
    valid = (value.get("system_name") == system_name
             and isinstance(value.get("status"), str)
             and value["status"] in {"healthy", "degraded", "down", "unknown"})
    for key in ("cpu_percent", "memory_percent"):
        metric = value.get(key)
        # bool도 파이썬에서는 숫자로 취급되므로 별도로 제외한다.
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
    """서버 연결 수명과 MCP 요청의 입력·시간 제한·응답 검증을 관리한다."""

    def __init__(self, flaky_first_tool_call: bool = False,
                 server_scripts: dict[str, Path] | None = None, *,
                 timeout_seconds: float = 10.0):
        """서버 구성과 양수인 제한 시간을 설정하고 연결에 필요한 상태를 준비한다."""
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
        """모든 서버 초기화가 성공한 경우에만 세션을 공개하고, 실패하면 연결을 정리한다."""
        # wait_for는 별도 태스크를 만들 수 있으므로 호출자 태스크를 유지하는 timeout을 쓴다.
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
                    cleanup_failed = False
                    try:
                        await stack.aclose()
                    except Exception as cleanup_error:
                        # ExitStack은 나머지 종료 콜백도 실행한다. 정리 실패는 따로 기록하되
                        # 원래 실패나 취소를 덮어쓰지 않고 원문 데이터도 노출하지 않는다.
                        cleanup_failed = True
                        _log_failure("all", "startup_cleanup", cleanup_error, cleanup_failed=True)
                    finally:
                        self.sessions = {}
                        self._stack = None
                    _log_failure(server_name, "start", error, cleanup_failed=cleanup_failed)
                    if isinstance(error, (asyncio.CancelledError, TimeoutError)):
                        raise
                    if isinstance(error, _SDK_ERROR) and error.error.code == REQUEST_TIMEOUT:
                        raise TimeoutError("MCP initialization timed out") from None
                    if not isinstance(error, Exception):
                        raise
                    raise MCPCallError("startup_failed", "MCP initialization failed; connection cleanup was attempted") from None
                # 일부 서버만 연결된 상태에서 요청을 받지 않도록 성공 시 한 번에 공개한다.
                self.sessions = sessions
                self._stack = stack

    async def stop(self) -> None:
        """세션을 먼저 사용 불가로 전환한 뒤, 시작한 태스크에서 연결을 정리한다."""
        async with asyncio.timeout(self.timeout_seconds):
            async with self._lock:
                stack, self._stack = self._stack, None
                self.sessions = {}
                if stack is not None:
                    try:
                        await stack.aclose()
                    except TimeoutError as error:
                        _log_failure("all", "stop", error, cleanup_failed=True)
                        raise
                    except Exception as error:
                        _log_failure("all", "stop", error, cleanup_failed=True)
                        raise MCPCallError("shutdown_failed", "MCP connection cleanup failed") from None

    def _session(self, server):
        """요청한 서버가 준비됐는지 확인하고 사용 가능한 세션을 반환한다."""
        _identifier(server, "server")
        if not self.sessions:
            raise MCPCallError("not_started", "MCP client has no ready sessions; call start first")
        if server not in self.sessions:
            raise MCPCallError("unknown_server", "Requested MCP server is not connected")
        return self.sessions[server]

    async def _request(self, server, method, *args):
        """잠금 대기와 통신 시간을 제한하고, 응답 오류와 재시도할 시간 초과를 구분한다."""
        self._session(server)
        try:
            # timeout을 잠금 바깥에 두어 다른 요청 때문에 기다리는 시간도 제한한다.
            async with asyncio.timeout(self.timeout_seconds):
                async with self._lock:
                    # 잠금을 기다리는 동안 stop()이 실행됐을 수 있으므로 다시 확인한다.
                    session = self._session(server)
                    result = await getattr(session, method)(*args)
            # 기능별 응답 해석 전에 공통 경계에서 오류 표시를 한 번만 확인한다.
            _check_error(result)
            return result
        except MCPCallError:
            raise
        except TimeoutError as error:
            _log_failure(server, method, error)
            raise
        except _SDK_ERROR as error:
            _log_failure(server, method, error)
            if error.error.code == REQUEST_TIMEOUT:
                # SDK 시간 초과도 기존 retry.py가 재시도하는 TimeoutError로 통일한다.
                raise TimeoutError("MCP request timed out") from None
            raise MCPCallError("request_failed", "MCP server rejected or could not complete the request") from None
        except Exception as error:
            _log_failure(server, method, error)
            raise MCPCallError("request_failed", "MCP request failed or returned an invalid protocol response") from None

    @traceable(run_type="tool", name="mcp.call_tool")
    async def call_tool(self, name: str, arguments: dict, server: str = "ops") -> dict:
        """도구 입력을 검증·복사해 호출하고 모니터링 결과는 요청 대상까지 대조한다."""
        _identifier(name, "tool name")
        arguments = _arguments(arguments)
        if name == "get_monitoring_status":
            if set(arguments) != {"system_name"}:
                raise MCPCallError("invalid_input", "get_monitoring_status requires only system_name")
            _identifier(arguments["system_name"], "system_name")
        if self._flaky_pending and name == "get_monitoring_status":
            # 명시적으로 켠 데모 설정에서만 최초 실패를 재현한다. 기본값은 비활성이다.
            self._session(server)
            self._flaky_pending = False
            raise TimeoutError("simulated transient timeout (Wave 9 demo)")
        value = _tool_object(await self._request(server, "call_tool", name, arguments))
        if name == "get_monitoring_status":
            return _monitoring_status(value, arguments["system_name"])
        return value

    @traceable(run_type="tool", name="mcp.list_resource_templates")
    async def list_resource_templates(self, server: str) -> list:
        """서버의 리소스 템플릿을 조회하고 이름과 URI 템플릿 형식을 검사한다."""
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
        """연결 준비 상태를 확인하고 현재 서버 이름을 튜플로 반환한다."""
        if not self.sessions:
            raise MCPCallError("not_started", "MCP client has no ready sessions; call start first")
        return tuple(self.sessions)

    @traceable(run_type="tool", name="mcp.find_resource_template")
    async def find_resource_template(self, template_name: str) -> tuple[str, str]:
        """전체 탐색 시간 안에서 템플릿을 찾아 서버 이름과 URI 템플릿을 반환한다."""
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
        """절대 URI 형식을 검증한 뒤 리소스의 텍스트 내용을 읽는다."""
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
        """서버의 프롬프트 목록을 조회하고 각 프롬프트 이름을 검사한다."""
        values = _list(await self._request(server, "list_prompts"), "prompts")
        for item in values:
            if not isinstance(_attr(item, "name"), str) or not item.name.strip():
                raise MCPCallError("invalid_response", "MCP prompt has an invalid name")
        return values

    @traceable(run_type="tool", name="mcp.find_prompt")
    async def find_prompt(self, prompt_name: str) -> str:
        """전체 탐색 시간 안에서 지정한 프롬프트를 제공하는 서버를 찾는다."""
        _identifier(prompt_name, "prompt name")
        async with asyncio.timeout(self.timeout_seconds):
            for server_name in self._servers():
                prompts = await self.list_prompts(server_name)
                if any(item.name == prompt_name for item in prompts):
                    return server_name
        raise MCPCallError("not_found", "MCP prompt not found on connected servers")

    @traceable(run_type="tool", name="mcp.get_prompt")
    async def get_prompt(self, name: str, arguments: dict, server: str = "ops") -> str:
        """문자열 인자를 검증·복사해 프롬프트를 요청하고 텍스트 메시지를 합친다."""
        _identifier(name, "prompt name")
        arguments = _arguments(arguments, prompt=True)
        if name == "incident_analysis_prompt":
            if set(arguments) != {"system_name", "symptom"} or not arguments.get("symptom", "").strip():
                raise MCPCallError("invalid_input", "incident_analysis_prompt requires system_name and a nonempty symptom")
            _identifier(arguments["system_name"], "system_name")
        values = _list(await self._request(server, "get_prompt", name, arguments), "messages", allow_empty=False)
        return "\n".join(_text(_attr(item, "content"), typed=True) for item in values)
