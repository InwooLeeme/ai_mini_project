"""비교 평가가 검증 불가를 정답으로 세거나 이전 호출의 토큰을 재사용하지 않는지 검사한다."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.validation import AnswerValidator
from evaluation import validator_latency
from test_answer_validation import ReviewStub


def test_case_set_preserves_recorded_context_and_distinguishes_controls():
    cases = {c['id']: c for c in validator_latency.comparison_cases()}
    assert len(cases) == 12
    assert sum(c['expected'] == 'passed' for c in cases.values()) == 3
    original = cases['recorded_context']
    assert '`cache-service`: 메모리 사용률이 89.0%' in original['draft']
    assert '모니터링 알림에 따르면 메모리 사용량이 6시간' in original['draft']
    assert cases['explicit_context']['collected'] == original['collected']
    assert cases['wrong_system_attribution']['expected'] == 'temporal'
    assert cases['wrong_period']['expected'] == 'temporal'
    assert cases['fabricated_logs']['expected'] == 'grounding'


def test_unavailable_review_is_failure_and_rule_rejection_has_no_model_usage(monkeypatch):
    all_cases = {c['id']: c for c in validator_latency.comparison_cases()}
    monkeypatch.setattr(validator_latency, 'comparison_cases',
                        lambda: [all_cases['recorded_context'], all_cases['wrong_numeric']])
    model = SimpleNamespace(last_usage={'total_tokens': 999}, last_model='previous')
    rows = []
    asyncio.run(validator_latency.evaluate(
        AnswerValidator(ReviewStub('{}')), model, 2, rows.append))
    assert len(rows) == 4
    for row in rows:
        assert row['provider_usage'] is None and row['model'] is None
        if row['id'] == 'recorded_context':
            assert row['result']['status'] == 'unavailable'
            assert not row['correct']
        else:
            assert row['correct'] and not row['result']['review_called']


def test_preflight_never_calls_model(monkeypatch, capsys):
    monkeypatch.setattr(AnswerValidator, 'validate', lambda *a: pytest.fail('network call'))
    assert validator_latency.main([]) == 0
    assert '모델 호출 없음' in capsys.readouterr().out


def test_invalid_repetition_is_rejected_without_a_call():
    with pytest.raises(SystemExit):
        validator_latency.main(['--repeats', '0'])


def test_extended_cases_keep_live_failures_visible_without_model_calls(capsys):
    cases = validator_latency.comparison_cases(True)
    assert len(cases) == 20 and len({c['id'] for c in cases}) == 20
    by_id = {c['id']: c for c in cases}
    assert by_id['duplicate_attributed_summary']['expected'] == 'temporal'
    assert by_id['near_55']['expected'] == 'numeric'
    assert validator_latency.main(['--include-context-edges']) == 0
    assert '20개 사례' in capsys.readouterr().out
