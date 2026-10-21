"""공급자 공통 응답 해석·프롬프트·비용 계산 정책을 모델 없이 검사한다."""
import json

import pytest

from agent.llm_roles import (
    CHAT_SYSTEM, build_chat_prompt, is_incident, parse_plan, parse_route, rendered_prompt,
    token_usage,
)


@pytest.mark.parametrize("query,expected", [
    ("payments-service 지연이 심해요", True),
    ("장애 티켓 확인", True),
    ("incident report", True),
    ("안녕하세요", False),
])
def test_is_incident(query, expected):
    assert is_incident(query) is expected


def test_parse_route_keeps_valid_model_output():
    content = json.dumps({"route": "incident", "needs_analysis": True})
    assert parse_route(content, "안녕") == {"route": "incident", "needs_analysis": True}


def test_parse_route_defaults_unknown_route_to_chat():
    assert parse_route('{"route": "escalate"}', "장애 진단") == {
        "route": "chat", "needs_analysis": False}


@pytest.mark.parametrize("content", ["not json", "[1, 2]", "null"])
@pytest.mark.parametrize("query,route", [("장애 진단", "incident"), ("안녕", "chat")])
def test_parse_route_falls_back_to_keywords_on_broken_output(content, query, route):
    assert parse_route(content, query) == {"route": route, "needs_analysis": False}


def test_parse_plan_limits_tasks_to_targets_in_query():
    content = json.dumps({"needs_analysis": False, "tasks": [
        {"agent": "research", "target": "log-service"},
    ]})
    plan = parse_plan(content, "cache-service 장애 진단")
    assert [t["target"] for t in plan["tasks"]] == ["cache-service"]


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_parse_plan_survives_broken_output(content):
    plan = parse_plan(content, "payments-service 장애 진단")
    assert [t["target"] for t in plan["tasks"]] == ["payments-service"]
    assert plan["needs_analysis"] is False


def test_parse_plan_creates_no_task_for_unconfirmed_target():
    assert parse_plan("{}", "unknown-service 장애 진단")["tasks"] == []


def test_build_chat_prompt_keeps_last_ten_messages():
    history = [{"role": "user", "content": f"m{i}"} for i in range(12)]
    system, user = build_chat_prompt("새 질문", history)
    assert system == CHAT_SYSTEM
    assert "user: m1\n" not in user
    assert "user: m2\n" in user and "user: m11" in user
    assert user.endswith("## 사용자의 새 메시지\n새 질문")


def test_build_chat_prompt_marks_empty_history():
    assert "## 지금까지의 대화\n(없음)" in build_chat_prompt("안녕", [])[1]


def test_rendered_prompt():
    assert rendered_prompt("S", "U") == "[SYSTEM]\nS\n\n[USER]\nU"


def test_token_usage_uses_virtual_unit_price():
    assert token_usage("m", 1500, 0.002) == {"model": "m", "tokens": 1500, "usd": 0.003}
    assert token_usage("m", 1, 0.00015)["usd"] == 0.0
