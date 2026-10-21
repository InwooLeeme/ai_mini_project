"""미수행 점검 완료 주장을 막되 정상 권고·실제 조회 결과는 의미 검토로 보낸다."""
import asyncio
import pytest

from agent.validation import AnswerValidator
from evaluation.inspection_grounding import inspection_cases
from test_answer_validation import ReviewStub, review_reply


@pytest.mark.parametrize('case', inspection_cases(), ids=lambda case: case['id'])
def test_inspection_completion_requires_evidence_without_rejecting_advice(case):
    # 의미 모델이 잘못 pass하더라도 실제 C02 완료 주장이 최종 통과하면 안 된다.
    result = asyncio.run(AnswerValidator(ReviewStub(review_reply())).validate(
        case['query'], case['draft'], case['collected'], case['targets']))
    if case['expected'] == 'grounding':
        assert result.status == 'rejected'
        assert any(i.code == 'grounding' for i in result.issues)
        assert not result.review_called
    else:
        assert result.status == 'passed'
        assert result.review_called


def test_monitoring_alert_is_not_direct_independent_inspection():
    case = inspection_cases()[0]
    case['collected'][0]['data']['alerts'].append('DNS 변경 사항을 점검했습니다.')
    case['draft'] = ('1) gateway-service의 모니터링 알림에 따르면 DNS 변경 사항을 점검했습니다. '
                     '2) payments-service CPU 91%입니다. 직접 점검한 사실은 없습니다.')
    result = asyncio.run(AnswerValidator(ReviewStub(review_reply())).validate(
        case['query'], case['draft'], case['collected'], case['targets']))
    assert result.status == 'passed'


def test_completed_inspection_is_rewritten_as_advice_before_saving():
    from fastapi.testclient import TestClient
    from test_app_runtime import make_test_app
    app, store, _ = make_test_app()
    feedbacks = []

    async def write(query, collected, attempt, feedback, on_token=None):
        feedbacks.append(feedback)
        text = ('DNS 변경 사항을 점검하였습니다.' if attempt == 1 else
                'DNS 변경 여부는 미확인입니다. 추가 점검이 필요합니다.')
        return '1) gateway-service와 payments-service의 관측을 확인합니다. 2) ' + text, {'usd': 0}

    with TestClient(app) as client:
        app.state.deps.quality_llm.write = write
        app.state.deps.answer_validator = AnswerValidator(ReviewStub(review_reply()))
        result = client.post('/chat', json={
            'message': 'gateway-service에서 payments-service 요청이 느린 장애입니다',
            'thread_id': 'inspection-regression'}).json()
        saved = store.load('inspection-regression')
    checks = [e for e in result['events'] if e['type'] == 'Validation']
    assert result['metadata']['validation_attempts'] == 2
    assert checks[0]['issues'][0]['code'] == 'grounding'
    assert '[grounding]' in feedbacks[1]
    assert checks[-1]['status'] == 'passed'
    assert '추가 점검이 필요합니다' in result['reply']
    assert '점검하였습니다' not in result['reply']
    assert saved[-1]['content'] == result['reply']


@pytest.mark.parametrize('object_phrase', [
    'DNS와 방화벽 변경 사항을 점검하라는 절차를',
    'DNS와 방화벽 점검 내용과 지침을',
])
def test_reading_a_runbook_procedure_does_not_claim_performing_it(object_phrase):
    case = inspection_cases()[0]
    draft = ('1) gateway-service와 payments-service를 확인합니다. '
             f'2) 런북에서 {object_phrase} 확인했습니다.')
    result = asyncio.run(AnswerValidator(ReviewStub(review_reply())).validate(
        case['query'], draft, case['collected'], case['targets']))
    assert result.status == 'passed' and result.review_called
