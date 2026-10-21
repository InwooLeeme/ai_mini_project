"""fast·quality 모델 역할의 계약과 공급자 공통 정책.

무엇을 묻고 응답을 어떻게 해석하는지는 여기서 한 번만 정의하고, 각 백엔드
(fake_llm·ollama_llm·openai_llm)는 전송 방식만 담당한다. writer.py(작성 프롬프트)·
analysis_policy.py(분석 지시)와 같은 분리 원칙이다.
"""
import json
import re
from collections.abc import Callable
from typing import Protocol

from .targets import KNOWN_SYSTEMS, normalize_plan, resolve_targets

OnToken = Callable[[str], None] | None


class FastModel(Protocol):
    """분류·계획·분석·단순 대화 전담 (저비용 모델)."""
    name: str

    def route(self, query: str) -> tuple[dict, dict]: ...
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]: ...
    async def analyze(self, collected: list[dict]) -> tuple[str, dict]: ...
    async def chat(self, query: str, history: list[dict],
                   on_token: OnToken = None) -> tuple[str, dict]: ...


class ReviewModel(Protocol):
    async def review(self, query: str, draft: str, collected: list[dict],
                     targets: list[str]) -> tuple[str, dict]: ...


class QualityModel(ReviewModel, Protocol):
    """최종 답변 작성 전담 (고품질 모델). 기본 답변 검증기도 같은 모델로 검토한다."""
    name: str

    async def write(self, query: str, collected: list[dict], attempt: int,
                    feedback: str | None, on_token: OnToken = None) -> tuple[str, dict]: ...


INCIDENT_PATTERN = re.compile(r"장애|진단|다운|지연|급증|incident|티켓")

ROUTE_SYSTEM = ("당신은 ICT 운영 지원 라우터다. 사용자 요청을 분류하라. "
                "장애·성능 저하·지연·진단·모니터링·티켓 등 운영 이슈면 route=incident, "
                "인사·잡담·일반 질문이면 route=chat. "
                "needs_analysis는 패턴 비교/통계 분석이 꼭 필요할 때만 true.")

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["chat", "incident"]},
        "needs_analysis": {"type": "boolean"},
    },
    "required": ["route", "needs_analysis"],
}

PLAN_SYSTEM = ("당신은 ICT 운영 Supervisor의 planning 노드다. 표준 진단 절차에 따라 "
               "실행 계획(tasks)을 JSON으로 세워라. "
               "사용자 요청에 등장한 시스템마다 agent=research task를 하나씩 세워라 "
               f"(알려진 시스템: {', '.join(sorted(KNOWN_SYSTEMS))} — 특정 시스템이 "
               "언급되지 않았으면 조사 대상을 임의로 만들지 마라). "
               "패턴 분석이 꼭 필요하다고 판단되면 agent=analysis task를 추가해도 된다.")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_analysis": {"type": "boolean"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string",
                              "enum": ["research", "analysis"]},
                    "target": {"type": "string"},
                },
                "required": ["agent"],
            },
        },
    },
    "required": ["needs_analysis", "tasks"],
}

CHAT_SYSTEM = ("당신은 간결한 한국어 비서다. 한두 문장으로 답하라. "
               "대화 이력에서 사용자가 밝힌 이름이 있으면 정확히 기억해 답하라.")


def is_incident(query: str) -> bool:
    return bool(INCIDENT_PATTERN.search(query))


def parse_route(content: str, query: str) -> dict:
    try:
        raw = json.loads(content)
        route = raw.get("route") if raw.get("route") in ("chat", "incident") else None
        return {"route": route or "chat",
                "needs_analysis": bool(raw.get("needs_analysis", False))}
    except (json.JSONDecodeError, AttributeError):
        # 구조화 출력이 형식만 보장하거나 붕괴해도 코드 레벨 fallback으로 분류한다 (W6 정신).
        return {"route": "incident" if is_incident(query) else "chat",
                "needs_analysis": False}


def plan_user(query: str, procedure_prompt: str) -> str:
    return f"## 표준 절차\n{procedure_prompt}\n\n## 사용자 요청\n{query}"


def parse_plan(content: str, query: str) -> dict:
    """형식이 지켜져도 내용은 흔들릴 수 있으므로(대상 시스템 오타, task 누락/중복)
    실행 가능한 형태로 보정한다."""
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        raw = {"needs_analysis": False, "tasks": []}
    return normalize_plan(raw, resolve_targets(query).targets)


def build_chat_prompt(query: str, history: list[dict]) -> tuple[str, str]:
    rendered = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
    user = (f"## 지금까지의 대화\n{rendered or '(없음)'}\n\n"
            f"## 사용자의 새 메시지\n{query}")
    return CHAT_SYSTEM, user


def rendered_prompt(system: str, user: str) -> str:
    """화면에 보여줄 실제 프롬프트 표기."""
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"


def token_usage(model_name: str, tokens: int, usd_per_1k: float) -> dict:
    """세 백엔드가 같은 잣대로 비교되도록 토큰 수 × 가상 단가로 비용을 계산한다."""
    return {"model": model_name, "tokens": tokens,
            "usd": round(tokens / 1000 * usd_per_1k, 6)}
