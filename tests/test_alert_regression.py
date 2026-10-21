"""실제 기록의 잘못된 피드백과 알림 삭제를 회귀 평가가 놓치지 않는지 검사한다."""
import copy
import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[1] / 'evaluation/fixtures/alert-feedback.json'


def test_recorded_false_rejection_does_not_satisfy_attribution_policy():
    from evaluation.alert_regression import attribution_feedback_ok
    original = json.loads(FIXTURE.read_text())
    assert not attribution_feedback_ok(original['recorded_review']['checks'])
    corrected = copy.deepcopy(original['recorded_review']['checks'])
    temporal = next(c for c in corrected if c['criterion'] == 'temporal')
    temporal.update(reason='ALERT_ATTRIBUTION: 알림 출처를 명시해야 합니다.',
                    correction='cache-service의 모니터링 알림에 따르면 최근 6시간 동안 메모리가 증가했습니다.')
    assert attribution_feedback_ok(corrected)


@pytest.mark.parametrize('draft,expected', [
    ('cache-service의 모니터링 알림에 따르면 최근 6시간 동안 메모리가 증가했습니다. 힙 한계 근접 알림도 있습니다.', True),
    ('cache-service 메모리 89%입니다. 증가하는지 확인이 필요합니다.', False),
    ('payments-service의 모니터링 알림에 따르면 최근 6시간 동안 메모리가 증가했습니다.', False),
    ('cache-service는 6시간 동안 메모리가 증가했습니다.', False),
])
def test_retention_screen_distinguishes_reported_alert_from_generic_uncertainty(draft, expected):
    from evaluation.alert_regression import retention_signals
    assert retention_signals(draft)['attributed_memory_trend'] is expected


def test_negative_cases_remove_generated_analysis_and_only_relevant_alerts():
    from evaluation.alert_regression import load_fixture, regression_cases
    fixture = load_fixture()
    original = copy.deepcopy(fixture)
    cases = {c['id']: c for c in regression_cases(fixture)}
    absent = cases['no_alert']['collected']
    assert all(c['source'] != 'analysis' for c in absent)
    assert next(c for c in absent if c['source'] == 'monitoring' and c['data']['system_name'] == 'cache-service')['data']['alerts'] == []
    assert fixture == original
    assert set(cases) == {'original', 'attributed', 'no_alert', 'other_metric', 'other_system', 'restart_89'}


def test_preflight_does_not_call_models_or_read_credentials(monkeypatch):
    from evaluation import alert_regression
    import dotenv
    def forbidden(*args, **kwargs):
        pytest.fail('preflight must not load credentials')
    monkeypatch.setattr(dotenv, 'dotenv_values', forbidden)
    assert alert_regression.main([]) == 0


@pytest.mark.parametrize('feedback_kind', ['attribution', 'recorded'])
def test_graph_rewrite_transports_alerts_and_feedback_to_real_adapters(monkeypatch, feedback_kind):
    """HTTP 응답은 대역이다. 모델의 의미 판단이 아니라 실제 노드·전송 경로를 검증한다."""
    import asyncio
    from types import SimpleNamespace
    import httpx
    from agent import graph
    from agent.openai_llm import OpenAILLM, OpenAIQualityLLM
    from agent.validation import AnswerValidator, CRITERIA
    from evaluation.alert_regression import load_fixture, regression_cases, retention_signals

    fixture = load_fixture()
    corrected = regression_cases(fixture)[1]['draft']
    review = copy.deepcopy(fixture['recorded_review'])
    if feedback_kind == 'attribution':
        temporal = next(c for c in review['checks'] if c['criterion'] == 'temporal')
        temporal.update(reason='ALERT_ATTRIBUTION: 알림 근거가 있으므로 출처를 명시하세요.',
                        correction='cache-service의 모니터링 알림에 따르면 최근 6시간 동안 '
                                   '메모리가 증가했고 힙 한계에 근접했습니다.')
    passed = {'checks': [dict(criterion=c, status='pass', quote='', evidence_ids=[],
                             reason='대역 판정', correction='') for c in CRITERIA]}
    responses = iter([json.dumps(review), corrected, json.dumps(passed)])
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            'choices': [{'message': {'content': next(responses)}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}})
    original_client = httpx.AsyncClient
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-key')
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original_client(
        transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(httpx, 'post', lambda url, *, json, **kw: respond(
        httpx.Request('POST', url, json=json)))
    monkeypatch.setattr(graph, 'get_stream_writer', lambda: lambda event: None)
    deps = SimpleNamespace(quality_llm=OpenAIQualityLLM('writer', 0))
    validator = AnswerValidator(OpenAILLM('validator', 0))
    state = dict(query=fixture['query'], draft=fixture['draft'], collected=fixture['collected'],
                 resolved_targets=fixture['targets'], events=[])
    async def run():
        state.update(await graph.validate_node(state, deps, validator=validator))
        assert graph._after_validate(state) == 'write'
        feedback = state['validation_feedback']
        state.update(await graph.write_node(state, deps))
        state.update(await graph.validate_node(state, deps, validator=validator))
        assert graph._after_validate(state) == 'finalize'
        state.update(await graph.finalize_node(state, deps))
        return feedback
    feedback = asyncio.run(run())
    assert [r['model'] for r in requests] == ['validator', 'writer', 'validator']
    assert feedback in requests[1]['messages'][1]['content']
    for request in requests:
        user = request['messages'][1]['content']
        assert 'memory usage climbing steadily over 6h' in user
        assert 'heap near limit' in user
    assert state['collected'] == fixture['collected']
    assert state['validation_attempts'] == 2
    assert state['validation_status'] == 'passed'
    assert state['final'] == corrected
    assert all(retention_signals(state['final']).values())


@pytest.mark.parametrize('status,expected_writes', [('rejected', 2), ('unavailable', 1)])
def test_evaluation_does_not_pass_exhausted_or_unavailable_rewrites(status, expected_writes):
    import asyncio
    from agent.validation import ValidationResult, Issue
    from evaluation.alert_regression import evaluate, load_fixture
    fixture = load_fixture()
    class Validator:
        calls = 0
        async def validate(self, *args):
            self.calls += 1
            current = 'rejected' if self.calls <= 6 else status
            return ValidationResult(current, (Issue('temporal', '', 'test', 'test'),), (), {}, 0, True)
    class Writer:
        async def write(self, *args):
            return fixture['draft'], {}
    records = []
    asyncio.run(evaluate(fixture, Writer(), Validator(), records.append))
    rewrites = [r for r in records if r['kind'] == 'rewrite']
    assert len(records) == 8
    assert len(rewrites) == 2
    assert all(r['writes'] == expected_writes and not r['passed'] for r in rewrites)
