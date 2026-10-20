"""v2 Supervisor — 실제 langgraph.StateGraph (Wave 4·5·6·7·8·10b·15 통합).

route → (chat 직행 | incident 파이프라인)
incident: plan(MCP Prompt 절차 반영) → gate(권위 게이트) → research(가드+병렬+재시도)
          → replan_check(RAG·필요한 분석 병렬 실행 및 합류) → write
          → validate(반려 시 재작성) → finalize
검색기를 주입하지 않은 기존 비교 경로는 재조사 → analysis 순서를 유지한다.

모든 노드는 SupervisorDeps를 functools.partial로 주입받는 순수 함수다(Wave 13).
각 노드는 @traceable로 LangSmith에 계층 트리로 기록된다.
"""
import asyncio
import json
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langsmith import traceable

from .deps import SupervisorDeps
from .evidence import ResearchTask
from .analysis_policy import comparison_requested


def _load_known_systems() -> set[str]:
    """ollama_llm.py/openai_llm.py와 동일한 소스(data/monitoring.json)에서
    시스템 목록을 읽는다 — plan_node가 MCP Prompt에 넘길 system_name을
    쿼리에서 멤버십 매칭으로 뽑아낼 때 쓴다."""
    path = Path(__file__).resolve().parent.parent / "data" / "monitoring.json"
    try:
        return set(json.loads(path.read_text()).keys())
    except (OSError, json.JSONDecodeError):
        return {"auth-service", "web-01"}


KNOWN_SYSTEMS = _load_known_systems()


def _extract_system_name(query: str) -> str:
    """정규식 대신 실제 KNOWN_SYSTEMS 멤버십 매칭 — MCP Prompt(incident_analysis_
    prompt)의 system_name은 "시스템: X" 한 줄짜리 컨텍스트라 항상 정해진 이름
    중 하나로만 고정한다(여러 개 언급돼도 join하지 않음 — 실제 조사 대상 목록은
    이후 plan()의 tasks가 별도로 결정). 쿼리에서 가장 먼저 등장하는 시스템을
    고르고, 하나도 안 걸리면 가장 흔한 사례인 auth-service로 폴백."""
    matches = [(query.index(s), s) for s in KNOWN_SYSTEMS if s in query]
    return min(matches)[1] if matches else "auth-service"


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    history: list[dict]
    route: str
    needs_analysis: bool
    tasks: list[dict]
    gate_removed: int
    collected: list[dict]
    replan_rounds: int
    retry_count: int
    parallel_elapsed: float
    draft: str
    validation_attempts: int
    validation_feedback: str
    final: str
    last_prompt: str
    cost_usd: float
    events: list[dict]
    retrieval: dict


def _cost(state: AgentState, usage: dict) -> float:
    return round(state.get("cost_usd", 0.0) + usage["usd"], 6)


@traceable(name="route_node")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 — 모든 쿼리가 무거운 파이프라인을 탈 필요는 없다."""
    result, usage = deps.fast_llm.route(state["query"])
    event = {"type": "RouteDecided", "route": result["route"]}
    return {"route": result["route"], "needs_analysis": result["needs_analysis"],
            "cost_usd": _cost(state, usage),
            "events": state.get("events", []) + [event]}


@traceable(name="chat_answer_node")
async def chat_answer_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 direct 경로 — 히스토리 기반 응답 (Wave 12 세션 증명 지점).

    토큰 단위 스트리밍 — get_stream_writer()는 /chat/stream처럼 custom 모드로 소비될
    때만 실제로 전송되고, /chat(ainvoke)처럼 스트리밍하지 않는 호출에서는 자동으로
    no-op이 된다."""
    writer = get_stream_writer()
    reply, usage = await deps.fast_llm.chat(
        state["query"], state.get("history", []),
        on_token=lambda piece: writer({"type": "AnswerToken", "delta": piece}))
    return {"final": reply, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "Finalize", "pipeline": "chat-direct"}]}


@traceable(name="plan_node")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 4 — Plan/Task 분해. Wave 2의 MCP Prompt가 표준 절차를 깔아준다.

    resource와 마찬가지로 이름을 어느 서버가 들고 있는지 미리 가정하지 않고,
    find_prompt()로 연결된 서버들을 먼저 조회해 찾은 뒤에만 get_prompt를 보낸다."""
    prompt_server = await deps.mcp.find_prompt("incident_analysis_prompt")
    procedure = await deps.mcp.get_prompt(
        "incident_analysis_prompt",
        {"system_name": _extract_system_name(state["query"]), "symptom": state["query"]},
        server=prompt_server,
    )
    result, usage = deps.fast_llm.plan(state["query"], procedure)
    targets = [t["target"] for t in result["tasks"]
               if t["agent"] == "research" and t.get("target")]
    event = {"type": "PlanCreated", "task_count": len(result["tasks"]),
             "targets": targets}
    return {"tasks": result["tasks"], "needs_analysis": result["needs_analysis"],
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="gate_node")
async def gate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """사용자의 명시적 비교 요청으로 분석 계획을 정규화한다. 관측 수는 수집 후 검사한다."""
    tasks = state["tasks"]
    requested = comparison_requested(state["query"], KNOWN_SYSTEMS)
    kept = [t for t in tasks if t["agent"] != "analysis"]
    analysis_tasks = [t for t in tasks if t["agent"] == "analysis"]
    removed = len(analysis_tasks) - (1 if requested and analysis_tasks else 0)
    if requested:
        # 모델이 필요한 작업을 빠뜨리거나 중복 생성해도 정확히 한 번 실행한다.
        kept.append({**(analysis_tasks[0] if analysis_tasks else {"id": "task-analysis"}),
                     "agent": "analysis", "kind": "pattern-analysis", "status": "pending"})
    event = {"type": "GateApplied", "removed_analysis_tasks": removed}
    return {"tasks": kept, "gate_removed": removed, "needs_analysis": requested,
            "events": state["events"] + [event]}


@traceable(name="research_node")
async def research_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """대기 중 조사 작업을 수집기에 맡기고 성공한 결과를 상태·이벤트에 반영한다."""
    tasks = state["tasks"]
    research_indices = [i for i, t in enumerate(tasks)
                        if t["agent"] == "research" and t["status"] == "pending"]
    # Wave 15 — 객체가 아니라 '실행할 목록'을 검사한다 (not research_agent 버그의 수정판)
    if not research_indices:
        return {"events": state["events"] +
                [{"type": "ResearchSkipped", "reason": "pending research task 0개"}]}

    batch = await deps.evidence_collector.collect([
        ResearchTask(i, tasks[i]["kind"], tasks[i]["target"]) for i in research_indices
    ])

    # 전체 수집 성공 후에만 복사본을 갱신하므로 실패 시 기존 상태가 유지된다.
    retry_total = state.get("retry_count", 0)
    new_tasks = [dict(t) for t in tasks]
    collected = list(state.get("collected", []))
    events = list(state["events"])
    for item in batch.items:
        new_tasks[item.index]["status"] = "done"
        collected.append({"source": item.source, "data": item.data})
        retry_total += item.retries
        events.append({"type": "TaskCompleted", "id": new_tasks[item.index]["id"],
                       "source": item.source, "target": new_tasks[item.index]["target"],
                       "retries": item.retries})
    return {"tasks": new_tasks, "collected": collected, "retry_count": retry_total,
            "parallel_elapsed": batch.elapsed, "events": events}


# 수집된 모니터링 결과(상태·알림 문구)를 보고 어떤 runbook이 맞는지 고른다. LLM에게
# 다시 묻는 대신 코드 레벨 규칙으로 판단한다 — gate(Wave 6)·replan(Wave 7)과 같은 설계
# 원칙: 모델의 판단을 그대로 믿지 않고, 실측 데이터로 검증 가능한 결정은 코드가 내린다.
_INCIDENT_TYPE_RULES = [
    ("latency", "high-latency"),
    ("pool", "high-latency"),
    ("cpu", "cpu-spike"),
    ("memory", "memory-leak"),
    ("heap", "memory-leak"),
    ("timeout", "network-timeout"),
    ("network", "network-timeout"),
    ("disk", "disk-full"),
    ("down", "db-down"),
]


def _classify_incident_type(collected: list[dict]) -> str:
    text = " ".join(
        f"{c['data'].get('status', '')} " + " ".join(c["data"].get("alerts", []))
        for c in collected if c.get("source") == "monitoring"
    ).lower()
    for keyword, incident_type in _INCIDENT_TYPE_RULES:
        if keyword in text:
            return incident_type
    return "db-down"   # 매칭되는 증상이 없으면 가장 일반적인 장애 대응 절차로 폴백


@traceable(name="replan_check_node")
async def replan_check_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """수집 관측으로 RAG 근거를 조회한다. 검색기 미주입 시 기존 비교 경로를 사용한다."""
    retriever = getattr(deps, "runbook_retriever", None)
    if retriever is not None:
        observations = [c["data"] for c in state.get("collected", [])
                        if c["source"] == "monitoring"]
        result = await retriever.search(state["query"], observations)
        documents = [{"source": "runbook", "data": doc.text,
                      "document_id": doc.document_id, "uri": doc.uri}
                     for doc in result.documents]
        retrieval = {"document_ids": [d.document_id for d in result.documents],
                     "candidates": result.candidates, "reason": result.reason,
                     "usage": result.usage}
        events = state["events"] + [{"type": "RunbookRetrieved", **retrieval}]
        events.append({"type": "ReplanCheck", "round": 1, "is_sufficient": True,
                       "reason": "retrieval_complete" if documents else "no_relevant_runbook"})
        return {"collected": state.get("collected", []) + documents, "retrieval": retrieval,
                "replan_rounds": 1, "events": events}
    has_runbook = any(c["source"] == "runbook" for c in state.get("collected", []))
    rounds = state.get("replan_rounds", 0)
    if has_runbook or rounds >= 1:
        verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": True}
        return {"events": state["events"] + [verdict]}
    incident_type = _classify_incident_type(state.get("collected", []))
    gap_task = {"id": "task-gap-runbook", "agent": "research", "kind": "runbook",
                "target": incident_type, "status": "pending"}
    verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": False,
               "gap_task": gap_task["id"], "target": incident_type}
    return {"tasks": state["tasks"] + [gap_task], "replan_rounds": rounds + 1,
            "events": state["events"] + [verdict]}


@traceable(name="analysis_node")
async def analysis_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """명시적 비교 요청과 서로 다른 두 시스템의 관측이 모두 있을 때만 분석한다."""
    tasks = state["tasks"]
    pending = [t for t in tasks if t["agent"] == "analysis" and t["status"] == "pending"]
    observations = {}
    for item in state.get("collected", []):
        if item["source"] == "monitoring":
            name = item["data"].get("system_name")
            if isinstance(name, str) and name.strip():
                observations[name] = item
    reason = None
    if not comparison_requested(state["query"], KNOWN_SYSTEMS):
        reason = "explicit_comparison_not_requested"
    elif len(observations) < 2:
        reason = "insufficient_system_observations"
    elif not pending:
        reason = "no_pending_analysis"
    if reason:
        return {"tasks": [{**t, "status": "skipped"} if t in pending else dict(t) for t in tasks],
                "needs_analysis": False,
                "events": state["events"] + [{"type": "AnalysisSkipped", "reason": reason}]}
    summary, usage = await deps.fast_llm.analyze(list(observations.values()))
    new_tasks = [dict(t) for t in tasks]
    for t in new_tasks:
        if t["agent"] == "analysis" and t["status"] == "pending":
            t["status"] = "done"
    collected = state.get("collected", []) + [{"source": "analysis", "data": summary}]
    event = {"type": "AnalysisCompleted", "length": len(summary)}
    return {"tasks": new_tasks, "collected": collected,
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="prepare_evidence_node")
async def prepare_evidence_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """두 읽기 작업을 겹쳐 실행하고, 성공한 결과만 한 번 상태에 합친다."""
    if getattr(deps, "runbook_retriever", None) is None:
        return await replan_check_node(state, deps)
    jobs = [asyncio.create_task(replan_check_node(state, deps)),
            asyncio.create_task(analysis_node(state, deps))]
    try:
        retrieval, analysis = await asyncio.gather(*jobs)
    except BaseException:
        # gather는 한쪽 실패 시 다른 쪽을 자동 정리하지 않는다. 요청 취소도 같은 정책이다.
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        raise
    # 노드는 원본 상태를 읽고 새 목록을 반환한다. 공통 접두부는 한 번만 보존한다.
    collected = state.get("collected", [])
    events = state["events"]
    return {
        **retrieval, **analysis,
        "collected": retrieval["collected"] + analysis.get("collected", collected)[len(collected):],
        "events": retrieval["events"] + analysis["events"][len(events):],
    }


@traceable(name="write_node")
async def write_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8/10a — 고품질 모델로 초안 작성 (1차는 의도적 저품질).

    토큰마다 attempt 번호를 같이 실어 보낸다 — validate가 반려해 재작성(attempt 증가)
    될 때 프론트가 새 말풍선으로 끊어서 보여줄 수 있게 하기 위함(Wave 8 재작성 루프를
    그대로 노출)."""
    attempt = state.get("validation_attempts", 0) + 1
    writer = get_stream_writer()
    if "retrieval" in state and not state["retrieval"]["document_ids"]:
        # 미조회는 생성 모델이 빈 근거를 임의 조치로 채우지 못하도록 명시적으로 처리한다.
        draft = ("1) 현재 질문과 관측에 맞는 런북 근거를 찾지 못했습니다. "
                 "2) 구체적인 증상·오류 메시지·발생 시점을 확인하고 해당 대응 문서를 확보해 주세요. "
                 "3) 근거 확인 전에는 특정 원인이나 조치를 확정할 수 없습니다.")
        writer({"type": "AnswerToken", "attempt": attempt, "delta": draft})
        return {"draft": draft, "last_prompt": None, "events": state["events"] + [
            {"type": "DraftWritten", "attempt": attempt, "length": len(draft)}]}
    draft, usage = await deps.quality_llm.write(
        state["query"], state.get("collected", []),
        attempt, state.get("validation_feedback"),
        on_token=lambda piece: writer(
            {"type": "AnswerToken", "attempt": attempt, "delta": piece}))
    return {"draft": draft, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "DraftWritten", "attempt": attempt,
                                          "length": len(draft)}]}


@traceable(name="validate_node")
async def validate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8 — 코드 기준 검증: 30자 이상 + 절차 번호 포함이어야 합격."""
    draft = state["draft"]
    attempts = state.get("validation_attempts", 0) + 1
    ok = len(draft) >= 30 and "1)" in draft
    verdict = "합격" if ok else "불합격"
    if not ok and attempts >= 3:   # 실모델 무한 재작성 방지 — 최대 3회 후 통과 처리
        ok, verdict = True, "합격(최대 시도 도달 — 통과 처리)"
    feedback = None if ok else (
        f"응답이 너무 짧습니다({len(draft)}자, 최소 30자). 절차를 1) 2) 3) 번호로 "
        f"구체적으로 설명하세요.")
    event = {"type": "Validation", "attempt": attempts, "verdict": verdict}
    return {"validation_attempts": attempts, "validation_feedback": feedback,
            "events": state["events"] + [event]}


@traceable(name="finalize_node")
async def finalize_node(state: AgentState, deps: SupervisorDeps) -> dict:
    return {"final": state["draft"],
            "events": state["events"] + [{"type": "Finalize", "pipeline": "incident-v2"}]}


def _after_route(state: AgentState) -> str:
    return "chat_answer" if state["route"] == "chat" else "plan"


def _after_replan_check(state: AgentState) -> str:
    if "retrieval" in state:
        return "write"  # RAG 경로는 분석까지 합류했으므로 다시 실행하지 않는다.
    last = state["events"][-1]
    return "analysis" if last.get("is_sufficient") else "research"


def _after_validate(state: AgentState) -> str:
    return "finalize" if state.get("validation_feedback") is None else "write"


def build_graph(deps: SupervisorDeps):
    g = StateGraph(AgentState)
    node = lambda fn: partial(fn, deps=deps)   # Wave 13 — 명시적 의존성 주입
    g.add_node("route", node(route_node))
    g.add_node("chat_answer", node(chat_answer_node))
    g.add_node("plan", node(plan_node))
    g.add_node("gate", node(gate_node))
    g.add_node("research", node(research_node))
    g.add_node("replan_check", node(prepare_evidence_node))
    g.add_node("analysis", node(analysis_node))
    g.add_node("write", node(write_node))
    g.add_node("validate", node(validate_node))
    g.add_node("finalize", node(finalize_node))

    g.set_entry_point("route")
    g.add_conditional_edges("route", _after_route,
                            {"chat_answer": "chat_answer", "plan": "plan"})
    g.add_edge("chat_answer", END)
    g.add_edge("plan", "gate")
    g.add_edge("gate", "research")
    g.add_edge("research", "replan_check")
    g.add_conditional_edges("replan_check", _after_replan_check,
                            {"analysis": "analysis", "research": "research", "write": "write"})
    g.add_edge("analysis", "write")
    g.add_edge("write", "validate")
    g.add_conditional_edges("validate", _after_validate,
                            {"finalize": "finalize", "write": "write"})
    g.add_edge("finalize", END)
    return g.compile()
