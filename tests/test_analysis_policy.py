"""명시적 시스템 비교 요청 판별과 분석 입력 요약을 그래프 없이 검사한다."""
import pytest

from agent.analysis_policy import analysis_facts, comparison_requested
from agent.targets import KNOWN_SYSTEMS


@pytest.mark.parametrize("query", [
    "payments-service와 cache-service 비교해줘",
    "두 시스템의 공통 원인 찾아줘",
    "payments-service와 cache-service의 상관관계 확인",
    "compare payments-service and cache-service",
    "payments-service와 cache-service 비교는 생략. 아니, 비교해줘",
])
def test_comparison_requested(query):
    assert comparison_requested(query, set(KNOWN_SYSTEMS)) is True


@pytest.mark.parametrize("query", [
    "payments-service CPU와 메모리 비교",
    "payments-service와 cache-service 장애 진단",
    "payments-service와 cache-service 비교는 하지 마",
    "payments-service와 cache-service 비교하지 말고 상태만",
    "비교적 높은 payments-service와 cache-service CPU",
    "skip comparison of both services",
])
def test_comparison_not_requested(query):
    assert comparison_requested(query, set(KNOWN_SYSTEMS)) is False


def test_analysis_facts_summarizes_only_monitoring():
    collected = [
        {"source": "monitoring", "data": {"system_name": "payments-service", "status": "degraded",
                                          "cpu_percent": 91, "alerts": ["cpu"]}},
        {"source": "runbook", "data": "본문"},
    ]
    assert analysis_facts(collected) == "- payments-service: 상태=degraded, CPU 91%, 알림 ['cpu']"


def test_analysis_facts_marks_missing_data():
    assert analysis_facts([{"source": "runbook", "data": "본문"}]) == "(모니터링 데이터 없음)"
