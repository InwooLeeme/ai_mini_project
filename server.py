"""all_total 통합 실습 — FastAPI 백엔드 (Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : mcp_server.py를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : GET /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 legacy/신형 파이프라인 즉시 전환
- LangSmith: 요청당 chat_request 루트 run 아래로 노드·LLM·MCP 호출이 계층 기록됨
"""
import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from dotenv import load_dotenv

load_dotenv()   # LANGCHAIN_TRACING_V2 / API_KEY / PROJECT — import보다 먼저

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import StreamingResponse
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel

from agent.runtime import AgentRuntime, create_runtime
from agent.supervisor_v1 import run_v1

router = APIRouter()


class ChatRequest(BaseModel):   # Wave 1 — Pydantic 요청 검증
    message: str
    thread_id: str = "default"


def create_app(
    *, runtime_factory: Callable[[], AbstractAsyncContextManager[AgentRuntime]] = create_runtime,
) -> FastAPI:
    """실행 환경 생성 함수를 주입받아 서로 독립적인 앱을 만든다."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with runtime_factory() as runtime:
            application.state.deps = runtime.deps
            application.state.graph = runtime.graph
            application.state.llm_backend = runtime.llm_backend
            application.state.instance_id = id(runtime.deps)
            application.state.creation_count += 1
            yield

    application = FastAPI(lifespan=lifespan)
    application.state.creation_count = 0
    application.include_router(router)
    return application


@router.get("/health")
async def health(request: Request):
    app = request.app
    deps = app.state.deps
    validator_model = deps.answer_validator.model if deps.answer_validator else deps.quality_llm
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": app.state.creation_count,
            "supervisor_version": app.state.deps.version,
            "llm_backend": app.state.llm_backend,
            "fast_llm": app.state.deps.fast_llm.name,
            "quality_llm": app.state.deps.quality_llm.name,
            "validator_llm": getattr(validator_model, "name", None) if deps.version == "v2" else None}


@router.get("/version")
async def version(request: Request):
    app = request.app
    return {"supervisor_version": app.state.deps.version}


def _metadata(app: FastAPI, result: dict, run_id: str | None) -> dict:
    return {
        "route": result.get("route"),
        "task_count": len(result.get("tasks", [])),
        "gate_removed": result.get("gate_removed", 0),
        "replan_rounds": result.get("replan_rounds", 0),
        "retry_count": result.get("retry_count", 0),
        "parallel_elapsed": result.get("parallel_elapsed"),
        "validation_attempts": result.get("validation_attempts", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "llm_backend": app.state.llm_backend,
        "supervisor_version": app.state.deps.version,
        "instance_id": app.state.instance_id,
        "langsmith_run_id": run_id,
    }


@traceable(
    name="chat_request",
    # 앱은 실행 의존성이므로 기존 요청 추적 입력(message, thread_id)에 포함하지 않는다.
    process_inputs=lambda inputs: {key: value for key, value in inputs.items() if key != "app"},
)
async def _handle_chat(app: FastAPI, message: str, thread_id: str) -> tuple[dict, str | None]:
    deps = app.state.deps
    history = deps.session_store.load(thread_id)   # Wave 12
    if deps.version == "v2":
        result = await app.state.graph.ainvoke(
            {"query": message, "thread_id": thread_id, "history": history,
             "events": [], "cost_usd": 0.0})
    else:
        result = await run_v1(deps, message, thread_id, history)   # Wave 14
    deps.session_store.append(thread_id, "user", message)
    deps.session_store.append(thread_id, "assistant", result["final"])
    rt = get_current_run_tree()
    return result, (str(rt.id) if rt else None)


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    result, run_id = await _handle_chat(request.app, req.message, req.thread_id)
    return {"reply": result["final"], "events": result.get("events", []),
            "metadata": _metadata(request.app, result, run_id)}


@router.get("/chat/stream")
async def chat_stream(request: Request, message: str, thread_id: str = "default"):
    """Wave 3 — 그래프의 노드별 진행 상황을 SSE로 스트리밍.

    이벤트마다 'data: {json}\\n\\n' 형태로 내려보내고 마지막에 data: [DONE] 센티널을
    보낸다 — 프론트엔드는 \\n\\n 경계로 이벤트를 분리해 파싱해야 한다(Wave 3b).
    """
    app = request.app
    deps = app.state.deps

    async def gen():
        history = deps.session_store.load(thread_id)
        if deps.version != "v2":
            result = await run_v1(deps, message, thread_id, history)
            for ev in result["events"]:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'FinalAnswer', 'reply': result['final']}, ensure_ascii=False)}\n\n"
        else:
            seen = 0
            final = ""
            validation_status = None
            last_prompt = None   # finalize까지 살아남은 최종 write/chat 호출의 실제 프롬프트
            # stream_mode를 리스트로 주면 (mode, payload) 튜플이 나온다 — "updates"는
            # 노드가 끝날 때마다의 상태 델타, "custom"은 write_node/chat_answer_node가
            # get_stream_writer()로 실행 중에 즉시 흘려보내는 토큰(AnswerToken) 이벤트다.
            async for mode, payload in app.state.graph.astream(
                    {"query": message, "thread_id": thread_id, "history": history,
                     "events": [], "cost_usd": 0.0}, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    continue
                for node_name, delta in payload.items():
                    # 확인 대기가 없는 준비 노드는 상태를 바꾸지 않아 None 업데이트가 온다.
                    if not delta:
                        continue
                    for ev in (delta.get("events") or [])[seen:]:
                        ev = {**ev, "node": node_name}
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if delta.get("events"):
                        seen = len(delta["events"])
                    if delta.get("validation_status"):
                        validation_status = delta["validation_status"]
                    if delta.get("final"):
                        final = delta["final"]
                    if delta.get("last_prompt"):
                        last_prompt = delta["last_prompt"]
            answer = {'type': 'FinalAnswer', 'reply': final, 'prompt': last_prompt}
            if validation_status is not None:
                answer['validation_status'] = validation_status
            yield f"data: {json.dumps(answer, ensure_ascii=False)}\n\n"
            deps.session_store.append(thread_id, "user", message)
            deps.session_store.append(thread_id, "assistant", final)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# 기존 uvicorn server:app 진입점은 유지한다.
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
