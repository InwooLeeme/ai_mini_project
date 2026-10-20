"""실행 환경을 조립하고 MCP·저장소의 수명을 같은 태스크에서 관리한다."""
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .deps import SupervisorDeps
from .evidence import MCPEvidenceCollector
from .fake_llm import FastLLM, QualityLLM
from .graph import build_graph
from .mcp_client import MCPClient
from .session_store import SqliteSessionStore
from .rag import build_retriever


@dataclass
class AgentRuntime:
    """HTTP 요청이 공유하는 의존성과 실행 그래프를 묶는다."""

    deps: SupervisorDeps
    graph: Any
    llm_backend: str


def build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API)."""
    backend = os.environ.get("LLM_BACKEND", "fake")
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
        fast = OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"),
                             usd_per_1k_tokens=0.00015)
        quality = OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
    quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
    return backend, fast, quality


def _flush_traces():
    """기존 종료 동작처럼 추적 전송을 마무리하되 전송 실패로 종료를 막지 않는다."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass


@asynccontextmanager
async def create_runtime(
    *, mcp_factory: Callable = MCPClient,
    llm_factory: Callable = build_llms,
    session_store_factory: Callable = SqliteSessionStore,
    retriever_factory: Callable | None = build_retriever,
):
    """서버 수명당 한 번 조립하며, 조립·실행 실패 시에도 만든 자원을 정리한다.

    생성 함수를 전달하면 통신과 저장 방식을 교체할 수 있다. MCP의 AnyIO
    취소 범위를 지키기 위해 시작과 종료를 별도 태스크로 분리하지 않는다.
    """
    mcp = mcp_factory()
    store = None
    try:
        await mcp.start()
        backend, fast_llm, quality_llm = llm_factory()
        store = session_store_factory()
        deps = SupervisorDeps(
            fast_llm=fast_llm, quality_llm=quality_llm, mcp=mcp,
            session_store=store, version=os.environ.get("SUPERVISOR_VERSION", "v2"),
            evidence_collector=MCPEvidenceCollector(mcp),
        )
        if deps.version == "v2" and retriever_factory is not None:
            deps.runbook_retriever = await retriever_factory(mcp)
        graph = build_graph(deps) if deps.version == "v2" else None
        yield AgentRuntime(deps=deps, graph=graph, llm_backend=backend)
    finally:
        # MCP 종료가 실패해도 저장소 정리와 추적 전송은 각각 시도한다.
        try:
            await mcp.stop()
        finally:
            try:
                if store is not None:
                    store.close()
            finally:
                _flush_traces()
