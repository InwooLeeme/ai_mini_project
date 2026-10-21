"""대상 시스템 판별·후속 응답 분류·계획 보정 규칙을 그래프 없이 직접 검사한다."""
from types import SimpleNamespace

import pytest

from agent.targets import (
    classify_followup, is_runbook_request, load_runbook_titles, normalize_plan, resolve_targets,
)


@pytest.mark.parametrize("query,targets", [
    ("payments-service 장애 진단", ("payments-service",)),
    ("payments-service와 cache-service 장애 진단", ("payments-service", "cache-service")),
    ("payments-service payments-service 상태", ("payments-service",)),
    ("ERR_TIMEOUT 발생한 payments-service 진단", ("payments-service",)),
    ("p99_latency 높은 payments-service 진단", ("payments-service",)),
    ("status not confirmed for payments-service", ("payments-service",)),
])
def test_resolve_targets_accepts_exact_known_systems(query, targets):
    result = resolve_targets(query)
    assert result.targets == targets
    assert result.reason is None


@pytest.mark.parametrize("query,reason", [
    ("CPU 급증 원인 진단해주세요", "missing_system"),
    ("unknown-service 장애 진단", "unknown_system"),
    ("payments-service와 unknown-service 장애 진단", "unknown_system"),
    ("payments-service 말고 cache-service 진단", "ambiguous_scope"),
    ("payments-service가 아닌 cache-service 장애 진단", "ambiguous_scope"),
    ("except payments-service, check cache-service", "ambiguous_scope"),
    ("그 서비스 장애 진단", "ambiguous_scope"),
    ("payments-service는 정상, 나머지는 장애 진단", "ambiguous_scope"),
    ("결제 서비스 장애 진단", "alias_requires_confirmation"),
])
def test_resolve_targets_defers_uncertain_targets(query, reason):
    result = resolve_targets(query)
    assert result.targets == ()
    assert result.reason == reason


def test_resolve_targets_suggests_but_does_not_select_typos_and_aliases():
    assert resolve_targets("paymnts-service 장애").suggestions == ("payments-service",)
    assert resolve_targets("결제 서비스 장애").suggestions == ("payments-service",)


@pytest.mark.parametrize("query,kind,targets", [
    ("취소", "cancel", ()),
    ("취소해 주세요", "cancel", ()),
    ("cancel", "cancel", ()),
    ("payments-service", "select", ("payments-service",)),
    ("payments-service로 진단해 주세요", "select", ("payments-service",)),
    ("네, payments-service와 cache-service요", "select", ("payments-service", "cache-service")),
    ("paymnts-service", "clarify", ()),
    ("아마 payments-service일 거예요", "clarify", ()),
    ("결제 서비스요", "clarify", ()),
    ("CPU 사용률이 높아요", "new_request", ()),
    ("auth-service 장애 진단해줘", "new_request", ()),
    ("안녕하세요", "new_request", ()),
])
def test_classify_followup(query, kind, targets):
    decision = classify_followup(query)
    assert (decision.kind, decision.targets) == (kind, targets)


def test_normalize_plan_creates_one_research_task_per_confirmed_target():
    raw = {"needs_analysis": True, "tasks": [
        {"agent": "research", "target": "log-service"},
        {"agent": "analysis"},
    ]}
    plan = normalize_plan(raw, ["payments-service", "payments-service", "cache-service"])
    assert [(t["id"], t["agent"], t.get("target")) for t in plan["tasks"]] == [
        ("task-0", "research", "payments-service"),
        ("task-1", "research", "cache-service"),
        ("task-analysis", "analysis", None),
    ]
    assert all(t["status"] == "pending" for t in plan["tasks"])
    assert plan["needs_analysis"] is True


@pytest.mark.parametrize("raw", [[], None, {"tasks": "broken"}, {"tasks": ["research"]}])
def test_normalize_plan_ignores_malformed_model_output(raw):
    plan = normalize_plan(raw, ("payments-service",))
    assert [t["target"] for t in plan["tasks"]] == ["payments-service"]
    assert plan["needs_analysis"] is False


@pytest.mark.parametrize("query,expected", [
    ("CPU 급증 대응 찾아줘", True),
    ("CPU  급증 대응을 보여주세요.", True),
    ("cpu 급증 대응 보여줘", True),
    ("CPU 급증 대응 찾아줘 그리고 진단", False),
    ("payments-service CPU 급증 대응 찾아줘", False),
])
def test_is_runbook_request_matches_only_whole_title_requests(query, expected):
    assert is_runbook_request(query, ("CPU 급증 대응",)) is expected


def test_load_runbook_titles_reads_retriever_documents():
    documents = {
        "a": SimpleNamespace(text="# Runbook:  CPU   급증 대응\n본문"),
        "b": SimpleNamespace(text="# 다른 문서\n본문"),
        "c": SimpleNamespace(text=""),
    }
    assert load_runbook_titles(SimpleNamespace(documents=documents)) == ("CPU 급증 대응",)
