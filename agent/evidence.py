"""조사 자료 수집 Module: MCP 탐색·병렬 호출·재시도를 하나의 Interface로 제공한다."""
import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from .retry import with_retry


@dataclass(frozen=True)
class ResearchTask:
    """수집할 작업. index는 호출자가 가진 전체 작업 목록의 위치다."""

    index: int
    kind: str
    target: str


@dataclass(frozen=True)
class Evidence:
    """한 작업에서 수집한 자료와 해당 작업의 재시도 횟수."""

    index: int
    source: str
    data: dict | str
    retries: int


@dataclass(frozen=True)
class CollectionResult:
    """입력 작업 순서의 수집 결과와 탐색을 제외한 수집 소요 시간."""

    items: list[Evidence]
    elapsed: float


class EvidenceCollector(Protocol):
    """그래프 상태를 수정하지 않고 자료를 수집하는 Interface.

    결과는 입력 순서와 원래 작업 index를 유지한다. 수집 실패는 호출자에게
    전파하며, 호출자는 전체 수집이 성공한 뒤에만 상태를 반영한다.
    """

    async def collect(self, tasks: list[ResearchTask]) -> CollectionResult:
        """작업별 자료·재시도 횟수와 수집 소요 시간을 반환한다."""
        ...


class EvidenceMCP(Protocol):
    """자료 수집 구현에 필요한 MCP 호출만 명시한다."""

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """모니터링 도구를 호출한다."""
        ...

    async def find_resource_template(self, template_name: str) -> tuple[str, str]:
        """리소스를 제공하는 서버와 URI 템플릿을 찾는다."""
        ...

    async def read_resource(self, uri: str, *, server: str) -> str:
        """탐색한 서버에서 런북 본문을 읽는다."""
        ...


class MCPEvidenceCollector:
    """기존 MCP 수집 정책을 구현하는 Adapter. 연결 수명은 호출자가 관리한다."""

    def __init__(self, mcp: EvidenceMCP):
        """이미 준비된 MCP 클라이언트를 전달받는다."""
        self._mcp = mcp

    async def collect(self, tasks: list[ResearchTask]) -> CollectionResult:
        """런북 위치를 한 번 탐색한 뒤 병렬 수집하고 입력 순서로 반환한다."""
        # monitoring 이외의 kind를 런북으로 해석하는 기존 규칙을 유지한다.
        runbook_server = runbook_uri_template = None
        if any(task.kind != "monitoring" for task in tasks):
            runbook_server, runbook_uri_template = await self._mcp.find_resource_template(
                "get_incident_runbook")

        async def run_one(task: ResearchTask) -> Evidence:
            """모니터링의 시간 초과만 재시도하고 런북 오류는 그대로 전파한다."""
            if task.kind == "monitoring":
                data, retries = await with_retry(
                    self._mcp.call_tool, "get_monitoring_status",
                    {"system_name": task.target})
                return Evidence(task.index, "monitoring", data, retries)
            uri = runbook_uri_template.format(incident_type=task.target)
            data = await self._mcp.read_resource(uri, server=runbook_server)
            return Evidence(task.index, "runbook", data, 0)

        # 기존 측정과 비교할 수 있도록 탐색 시간을 제외하고 소수 둘째 자리로 유지한다.
        started = time.perf_counter()
        results = await asyncio.gather(*(run_one(task) for task in tasks))
        return CollectionResult(results, round(time.perf_counter() - started, 2))
