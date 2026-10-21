"""축약된 통과 판정만 확장하고 실패·누락·잘못된 근거는 통과시키지 않는다."""

import asyncio
import copy
import json

import pytest

from agent.validation import CRITERIA, AnswerValidator, parse_review
from test_answer_validation import ReviewStub, evidence, review_reply


def compact_checks():
    return [
        {"criterion": name, "status": "pass", "evidence_ids": [0]} for name in CRITERIA
    ]


def test_compact_passes_preserve_all_six_checks_and_public_fields():
    checks, issues = parse_review(
        json.dumps({"checks": compact_checks()}), "초안", evidence()
    )
    assert not issues
    assert [c["criterion"] for c in checks] == list(CRITERIA)
    assert all(
        c["status"] == "pass" and c["quote"] == "" and c["correction"] == ""
        for c in checks
    )
    assert all(c["reason"] and c["evidence_ids"] == [0] for c in checks)


@pytest.mark.parametrize("status", ["fail", "unknown"])
def test_compact_passes_do_not_hide_detailed_failure_or_unknown(status):
    checks = compact_checks()
    failure = json.loads(review_reply(status))["checks"][3]
    checks[3] = failure
    result = asyncio.run(
        AnswerValidator(ReviewStub(json.dumps({"checks": checks}))).validate(
            "payments-service 진단",
            "1) payments-service CPU 91%입니다. 2) 로그에서 오류 원인이 발견되었습니다.",
            evidence(),
            ["payments-service"],
        )
    )
    assert result.status == ("rejected" if status == "fail" else "unavailable")
    assert result.review_called
    assert result.issues[0].code == "grounding"
    assert failure["correction"] in result.feedback()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_criterion",
        "duplicate",
        "missing_status",
        "invalid_status",
        "missing_evidence",
        "bad_evidence",
        "boolean_evidence",
        "extra_field",
        "bad_quote",
        "bad_reason",
        "bad_correction",
        "invented_quote",
        "compact_failure",
        "compact_unknown",
    ],
)
def test_malformed_compact_reviews_never_pass(mutation):
    checks = compact_checks()
    if mutation == "missing_criterion":
        checks.pop()
    elif mutation == "duplicate":
        checks[-1] = copy.deepcopy(checks[0])
    elif mutation == "missing_status":
        checks[0].pop("status")
    elif mutation == "invalid_status":
        checks[0]["status"] = "okay"
    elif mutation == "missing_evidence":
        checks[0].pop("evidence_ids")
    elif mutation in ("bad_evidence", "boolean_evidence"):
        checks[0]["evidence_ids"] = [999 if mutation == "bad_evidence" else True]
    elif mutation == "extra_field":
        checks[0]["override"] = True
    elif mutation in ("bad_quote", "bad_reason", "bad_correction"):
        checks[0][mutation[4:]] = None
    elif mutation == "invented_quote":
        checks[0]["quote"] = "존재하지 않는 인용"
    else:
        checks[0]["status"] = "fail" if mutation == "compact_failure" else "unknown"
    with pytest.raises(ValueError):
        parse_review(json.dumps({"checks": checks}), "초안", evidence())


def test_legacy_detailed_reviews_remain_readable():
    checks, issues = parse_review(review_reply(), "초안", evidence())
    assert len(checks) == 6 and not issues
    assert checks[3]["reason"] == "로그 수집 근거 없음"


def test_temporal_pass_preserves_its_quote_and_reason():
    checks = compact_checks()
    draft = 'cache-service: 모니터링 알림에 따르면 메모리가 증가했습니다.'
    checks[2].update(quote=draft, reason='SUPPORTED: 같은 항목에서 시스템과 알림을 명시했습니다.')
    parsed, issues = parse_review(json.dumps({'checks': checks}), draft, evidence())
    assert not issues
    assert parsed[2]['quote'] == draft
    assert parsed[2]['reason'].startswith('SUPPORTED:')
    assert parsed[2]['correction'] == ''


def test_failure_cannot_omit_evidence_ids_even_with_explanation():
    checks = compact_checks()
    checks[3] = json.loads(review_reply('fail'))['checks'][3]
    checks[3].pop('evidence_ids')
    with pytest.raises(ValueError, match='invalid check fields'):
        parse_review(json.dumps({'checks': checks}), '초안', evidence())


@pytest.mark.parametrize('phrase', ['90%에 근접합니다', '90%에 가까운 상태입니다'])
def test_near_threshold_reference_is_left_to_semantic_review(phrase):
    draft = f'1) payments-service 메모리 사용률은 89%입니다. 2) 메모리 사용률이 {phrase}.'
    checks = compact_checks()
    result = asyncio.run(AnswerValidator(ReviewStub(json.dumps({'checks': checks}))).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.review_called
    assert result.status == 'passed'


def test_near_threshold_does_not_hide_a_wrong_observed_value():
    draft = '1) payments-service 메모리 사용률은 90%입니다. 2) 임계값에 근접한 상태입니다.'
    result = asyncio.run(AnswerValidator(ReviewStub()).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.status == 'rejected' and not result.review_called
    assert result.issues[0].code == 'numeric'


def test_distant_value_cannot_bypass_numeric_rules_by_saying_near():
    draft = '1) payments-service 메모리 사용률은 55%입니다. 2) 메모리 사용률이 90%에 근접합니다.'
    result = asyncio.run(AnswerValidator(ReviewStub(json.dumps({'checks': compact_checks()}))).validate(
        'payments-service 진단', draft, evidence(memory=55), ['payments-service']))
    assert result.status == 'rejected' and not result.review_called
    assert result.issues[0].code == 'numeric'


@pytest.mark.parametrize('claim', ['지속되는 상황이므로', '지속되는 상태입니다'])
def test_factual_cpu_duration_is_not_hidden_by_topic_case_word(claim):
    draft = (f'1) payments-service의 경우, CPU 사용률이 90%를 넘고 {claim}, '
             '로그 확인이 필요합니다. 2) 원인은 미확인입니다.')
    result = asyncio.run(AnswerValidator(ReviewStub()).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.status == 'rejected' and not result.review_called
    assert any(i.code == 'temporal' for i in result.issues)


def test_duration_question_is_still_reviewed_as_a_question():
    draft = ('1) payments-service의 경우, CPU 사용률이 90%를 넘고 지속되는 상황인지 '
             '확인이 필요합니다. 2) 원인은 미확인입니다.')
    result = asyncio.run(AnswerValidator(ReviewStub(json.dumps({'checks': compact_checks()}))).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.status == 'passed' and result.review_called


def test_verified_duration_condition_before_cpu_is_not_an_assertion():
    draft = ('1) payments-service: 지속 여부가 확인된다면 CPU가 90%를 넘고 지속되는 상황이므로 '
             '인스턴스 증설을 검토하세요. 2) 발생 시각을 확인해 주세요.')
    result = asyncio.run(AnswerValidator(ReviewStub(json.dumps({'checks': compact_checks()}))).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.status == 'passed' and result.review_called


@pytest.mark.parametrize('qualification', ['증설해야 한다고 단정할 수 없습니다', '증설해야 한다는 주장은 미확인입니다'])
def test_duration_rule_preserves_negation_and_uncertainty(qualification):
    draft = f'1) payments-service CPU가 90%를 넘고 지속되는 상황이므로 {qualification}. 2) 로그를 추가 확인하세요.'
    result = asyncio.run(AnswerValidator(ReviewStub(json.dumps({'checks': compact_checks()}))).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.status == 'passed' and result.review_called


def test_asserted_simultaneity_is_not_cured_by_an_unknown_common_cause():
    draft = ('1) payments-service CPU 91%입니다. 2) 두 서비스의 성능 저하가 동시에 발생한 점을 '
             '고려할 때 로그 분석이 필요합니다. 현재 공통 원인은 미확인입니다.')
    result = asyncio.run(AnswerValidator(ReviewStub()).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.status == 'rejected' and not result.review_called
    assert any(i.code == 'temporal' for i in result.issues)


def test_memory_duration_after_cpu_snapshot_is_not_cpu_duration():
    data = evidence(cpu=42)
    data[0]['data']['alerts'] = ['memory usage climbing steadily over 6h']
    draft = ('1) payments-service CPU는 42%이며, 모니터링 알림에 따르면 메모리 증가가 '
             '6시간 동안 지속되는 상태입니다. 2) 원인 확인이 필요합니다.')
    result = asyncio.run(AnswerValidator(ReviewStub(json.dumps({'checks': compact_checks()}))).validate(
        'payments-service 진단', draft, data, ['payments-service']))
    assert result.status == 'passed' and result.review_called


def test_conditional_simultaneity_confirmation_is_not_an_assertion():
    draft = ('1) payments-service CPU 91%입니다. 2) 두 서비스의 성능 저하가 동시에 '
             '발생한 점이 확인되면 공통 이벤트를 조사하세요.')
    result = asyncio.run(AnswerValidator(ReviewStub(json.dumps({'checks': compact_checks()}))).validate(
        'payments-service 진단', draft, evidence(), ['payments-service']))
    assert result.status == 'passed' and result.review_called
