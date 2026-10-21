"""인용된 서비스명은 문단 주제를 바꾸지 않으며 잘못된 수치는 계속 반려한다."""
import asyncio
import json
from pathlib import Path

import pytest

from agent.validation import AnswerValidator, rule_issues
from test_answer_validation import ReviewStub, review_reply


@pytest.fixture
def c02():
    # 2026-10-21 실제 C02 첫 Writer 초안과 해당 요청의 수집 근거.
    return json.loads((Path(__file__).parent / 'fixtures/c02-numeric-attribution.json').read_text())


def numeric_issues(draft, context):
    return [i for i in rule_issues(draft, context['collected'], context['targets'])
            if i.code == 'numeric']


def test_recorded_c02_correct_values_reach_semantic_review(c02):
    result = asyncio.run(AnswerValidator(ReviewStub(review_reply())).validate(
        c02['query'], c02['draft'], c02['collected'], c02['targets']))
    assert result.status == 'passed'
    assert result.review_called
    assert not result.issues


@pytest.mark.parametrize('cpu,memory', [(91, 45), (30, 55), (91, 55)])
def test_recorded_c02_wrong_system_values_are_rejected(c02, cpu, memory):
    draft = c02['draft'].replace('30.0%', f'{cpu}.0%').replace('45.0%', f'{memory}.0%')
    result = asyncio.run(AnswerValidator(ReviewStub(review_reply())).validate(
        c02['query'], draft, c02['collected'], c02['targets']))
    assert result.status == 'rejected'
    assert not result.review_called
    issues = [i for i in result.issues if i.code == 'numeric']
    assert len(issues) == (cpu != 30) + (memory != 45)
    assert all(i.evidence.startswith('gateway-service ') for i in issues)


@pytest.mark.parametrize('opening,closing', [("'", "'"), ('"', '"'), ('“', '”'), ('‘', '’'), ('`', '`')])
@pytest.mark.parametrize('cpu,expected_errors', [(30, 0), (91, 1)])
def test_alert_quote_does_not_change_following_metric_owner(c02, opening, closing, cpu, expected_errors):
    draft = (f'1) gateway-service의 알림은 {opening}upstream timeout to payments-service{closing}입니다. '
             f'CPU {cpu}%, 메모리 45%입니다.\n\n'
             '2) payments-service CPU 91%, 메모리 55%입니다.')
    issues = numeric_issues(draft, c02)
    assert len(issues) == expected_errors
    if issues:
        assert issues[0].evidence == 'gateway-service cpu_percent=30.0'


@pytest.mark.parametrize('cpu,expected_errors', [(91, 0), (30, 1)])
def test_quoted_explicit_numeric_claim_is_still_checked_without_leaking_owner(c02, cpu, expected_errors):
    draft = (f'1) gateway-service CPU 30%입니다. 참고 문구는 "payments-service CPU {cpu}%"입니다. '
             '메모리 45%입니다.\n\n2) payments-service 메모리 55%입니다.')
    issues = numeric_issues(draft, c02)
    assert len(issues) == expected_errors
    if issues:
        assert issues[0].evidence == 'payments-service cpu_percent=91.0'


@pytest.mark.parametrize('wrapper', ['', '`', '**'])
@pytest.mark.parametrize('gateway,payments,expected_errors', [(30, 91, 0), (91, 30, 2)])
def test_explicit_subject_switch_in_same_sentence(c02, wrapper, gateway, payments, expected_errors):
    draft = (f'1) {wrapper}gateway-service{wrapper}는 CPU {gateway}%, '
             f'{wrapper}payments-service{wrapper}는 CPU {payments}%, 메모리 55%입니다.')
    assert len(numeric_issues(draft, c02)) == expected_errors


def test_quote_does_not_hide_wrong_value_for_its_surrounding_subject(c02):
    draft = ('1) gateway-service의 알림은 "CPU 91%"입니다. '
             '2) payments-service CPU 91%입니다. 추가 조사가 필요합니다.')
    issues = numeric_issues(draft, c02)
    assert len(issues) == 1
    assert issues[0].evidence == 'gateway-service cpu_percent=30.0'


@pytest.mark.parametrize('wrapper', ["'", '"', '`'])
def test_quoted_destination_name_alone_is_not_a_subject_switch(c02, wrapper):
    draft = (f'1) gateway-service의 호출 대상은 {wrapper}payments-service{wrapper}입니다. '
             'CPU 30%, 메모리 45%입니다. 2) payments-service CPU 91%입니다.')
    assert not numeric_issues(draft, c02)


@pytest.mark.parametrize('cpu,expected_errors', [(91, 0), (30, 1)])
def test_formatted_subject_inside_quote_keeps_its_local_scope(c02, cpu, expected_errors):
    draft = (f'1) gateway-service CPU 30%입니다. 인용문은 "`payments-service` CPU {cpu}%"입니다. '
             '메모리 45%입니다. 2) payments-service 메모리 55%입니다.')
    issues = numeric_issues(draft, c02)
    assert len(issues) == expected_errors
    if issues:
        assert issues[0].evidence == 'payments-service cpu_percent=91.0'
