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


@pytest.mark.parametrize('transport', ['http', 'sse'])
@pytest.mark.parametrize('failure,attempts', [('invalid_review', 1), ('exhausted', 3)])
def test_limited_final_response_and_history_preserve_original_alerts(transport, failure, attempts):
    """외부 수집·모델 응답만 고정하고 실제 그래프와 응답·저장 경로를 실행한다."""
    from fastapi.testclient import TestClient
    from test_app_runtime import make_test_app, decode_sse
    from agent.evidence import CollectionResult, Evidence
    from agent.graph import build_graph
    from agent.validation import AnswerValidator
    from evaluation.alert_regression import load_fixture

    fixture = load_fixture()
    original = copy.deepcopy(fixture)
    monitoring = {c['data']['system_name']: c['data'] for c in fixture['collected']
                  if c['source'] == 'monitoring'}
    runbooks = {c['document_id']: c['data'] for c in fixture['collected'] if c['source'] == 'runbook'}

    class Collector:
        async def collect(self, tasks):
            return CollectionResult([
                Evidence(task.index, task.kind,
                         copy.deepcopy(monitoring[task.target]) if task.kind == 'monitoring'
                         else runbooks[task.target], 0)
                for task in tasks
            ], elapsed=0)

    async def write(*args, **kwargs):
        draft = (fixture['draft'] if failure == 'invalid_review' else
                 '1) payments-service CPU 99%입니다. cache-service 메모리 89%입니다. '
                 '2) 원인을 확정했으므로 즉시 프로세스를 재시작하세요.')
        return draft, {'usd': 0}

    async def review(*args):
        return '{', {'usd': 0}

    app, store, _ = make_test_app()
    with TestClient(app) as client:
        deps = app.state.deps
        # 원본 질문의 실모델 라우팅 결과를 고정한다(fake 라우터의 키워드 평가는 범위 밖).
        deps.fast_llm.route = lambda query: ({'route': 'incident', 'needs_analysis': True}, {'usd': 0})
        deps.evidence_collector = Collector()
        deps.quality_llm.write = write
        deps.quality_llm.review = review
        deps.answer_validator = AnswerValidator(deps.quality_llm)
        app.state.graph = build_graph(deps)
        request = {'message': fixture['query'], 'thread_id': 'limited-alerts'}
        if transport == 'sse':
            events = decode_sse(client.get('/chat/stream', params=request))
            final = events[-1]['reply']
            assert events[-1]['validation_status'] == 'limited'
        else:
            response = client.post('/chat', json=request)
            assert response.status_code == 200
            data = response.json()
            events, final = data['events'], data['reply']
        checks = [e for e in events if e['type'] == 'Validation']
        assert len(checks) == attempts
        assert checks[-1]['status'] == 'limited'
        assert checks[-1]['review_status'] == ('unavailable' if failure == 'invalid_review' else 'rejected')
        assert store.load('limited-alerts')[-1]['content'] == final

    payments, cache = final.split('- cache-service:', 1)
    assert 'CPU 91.0%' in payments and '메모리 89.0%' in cache
    for alert in monitoring['payments-service']['alerts']:
        assert alert in payments and alert not in cache
    for alert in monitoring['cache-service']['alerts']:
        assert alert in cache and alert not in payments
    assert '모니터링 알림' in payments and '모니터링 알림' in cache
    assert '원인과 조치를 확정하지 않습니다' in final
    assert '시계열' in final and '추가 확인' in final
    assert '즉시 프로세스를 재시작' not in final
    assert fixture == original


@pytest.mark.parametrize('alerts', [[], None])
def test_limited_answer_without_alerts_does_not_invent_or_leak_other_targets(alerts):
    from agent.validation import limited_answer
    from evaluation.alert_regression import load_fixture
    fixture = load_fixture()
    cache = fixture['collected'][1]['data']
    if alerts is None:
        cache.pop('alerts')
    else:
        cache['alerts'] = alerts
    final = limited_answer(fixture['collected'], ['cache-service', 'unknown-service'])
    assert '메모리 89.0%' in final
    assert 'unknown-service: 관측 근거 없음' in final
    assert 'payments-service' not in final
    assert 'heap near limit' not in final
    assert '모니터링 알림' not in final
