"""그래프 분기 함수와 증상 분류 규칙을 노드 실행 없이 검사한다."""
import pytest

from agent.graph import _after_replan_check, _after_route, _classify_incident_type


@pytest.mark.parametrize("route,expected", [
    ("chat", "chat_answer"), ("incident", "resolve_targets"),
])
def test_after_route(route, expected):
    assert _after_route({"route": route}) == expected


def test_after_replan_check_skips_analysis_on_rag_path():
    # RAG 경로는 prepare_evidence_node에서 분석까지 합류했으므로 다시 실행하지 않는다.
    state = {"retrieval": {}, "events": [{"type": "ReplanCheck", "is_sufficient": False}]}
    assert _after_replan_check(state) == "write"


@pytest.mark.parametrize("sufficient,expected", [(True, "analysis"), (False, "research")])
def test_after_replan_check_without_rag_follows_last_verdict(sufficient, expected):
    state = {"events": [{"type": "Other"},
                        {"type": "ReplanCheck", "is_sufficient": sufficient}]}
    assert _after_replan_check(state) == expected


def monitoring(status="ok", alerts=()):
    return {"source": "monitoring", "data": {"status": status, "alerts": list(alerts)}}


@pytest.mark.parametrize("collected,expected", [
    ([monitoring(alerts=["p99 latency high"])], "high-latency"),
    ([monitoring(alerts=["DB connection pool exhausted"])], "high-latency"),
    ([monitoring(alerts=["CPU 95%"])], "cpu-spike"),
    ([monitoring(alerts=["heap usage rising"])], "memory-leak"),
    ([monitoring(alerts=["upstream timeout"])], "network-timeout"),
    ([monitoring(alerts=["disk 98%"])], "disk-full"),
    ([monitoring(status="down")], "db-down"),
    # 앞 규칙이 우선한다: latency와 cpu가 함께 있으면 latency로 분류한다.
    ([monitoring(alerts=["CPU 95%"]), monitoring(alerts=["latency 2s"])], "high-latency"),
])
def test_classify_incident_type(collected, expected):
    assert _classify_incident_type(collected) == expected


def test_classify_incident_type_ignores_other_sources_and_falls_back():
    collected = [{"source": "runbook", "data": {"alerts": ["cpu"]}}, monitoring()]
    assert _classify_incident_type(collected) == "db-down"
    assert _classify_incident_type([]) == "db-down"
