"""관측·런북 조건을 어긴 초안이 최종 답변으로 통과하지 않는지 검증한다."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.fake_llm import FastLLM
from agent.graph import validate_node

ROOT = Path(__file__).resolve().parents[1]


def evidence(memory=89, cpu=91, alerts=None):
    return [
        {"source": "monitoring", "data": {
            "system_name": "payments-service", "status": "degraded", "cpu_percent": cpu,
            "memory_percent": memory, "last_checked": "2026-10-21T09:00:00",
            "alerts": alerts or [],
        }},
        *[{"source": "runbook", "document_id": name,
           "uri": f"runbook://incident-response/{name}",
           "data": (ROOT / 'data/runbooks' / f'{name}.md').read_text()}
          for name in ('cpu-spike', 'memory-leak')],
    ]


def validate(draft, *, attempts=0, collected=None):
    state = {"draft": draft, "query": "payments-service CPU와 메모리 진단",
             "resolved_targets": ["payments-service"], "events": [], "cost_usd": 0,
             "collected": collected if collected is not None else evidence(),
             "validation_attempts": attempts}
    return asyncio.run(validate_node(state, SimpleNamespace(quality_llm=FastLLM('fake', 0))))


@pytest.mark.parametrize('draft', [
    '1) payments-service CPU 91%입니다. 2) CPU가 90%를 넘고 지속되고 있으므로 인스턴스를 추가하세요.',
    '1) payments-service 메모리 89%입니다. 2) OOM 직전이므로 해당 프로세스를 즉시 재시작하세요.',
    '1) payments-service CPU 55%입니다. 2) 현재 수치를 기준으로 상태를 확인하고 원인을 추가 조사하세요.',
])
def test_invalid_content_is_rejected_despite_valid_format(draft):
    result = validate(draft)
    assert result['validation_feedback'] is not None
    assert result['events'][-1]['verdict'] == '불합격'


def test_max_attempt_never_forces_bad_answer_to_pass():
    result = validate('잘못된 짧은 답변', attempts=2)
    assert result['events'][-1]['verdict'] != '합격'
    assert result['draft'] != '잘못된 짧은 답변'
    assert '91' in result['draft']
    assert '확인' in result['draft']
    assert result['validation_status'] == 'limited'


@pytest.mark.parametrize('cpu,memory,draft', [
    (89, 89, '1) payments-service CPU 89%, 메모리 89%입니다. 2) CPU가 90%를 넘고 지속된다면 인스턴스를 추가하세요.'),
    (90, 90, '1) payments-service CPU 90%, 메모리 90%입니다. 2) 메모리 90% 이상이면 프로세스를 재시작하세요.'),
    (91, 91, '1) payments-service CPU 91%, 메모리 91%입니다. 2) 지속 여부는 확인되지 않았으므로 추가 관측이 필요합니다.'),
])
def test_conditional_advice_and_unknown_duration_are_not_rejected(cpu, memory, draft):
    result = validate(draft, collected=evidence(cpu=cpu, memory=memory))
    assert result['validation_status'] == 'passed'


@pytest.mark.parametrize('value,expected', [(89, 'rejected'), (90, 'passed'), (91, 'passed')])
def test_memory_restart_uses_runbook_boundary(value, expected):
    draft = f'1) payments-service 메모리 {value}%입니다. 2) 메모리 부족으로 프로세스를 즉시 재시작하세요.'
    assert validate(draft, collected=evidence(memory=value))['validation_status'] == expected


def test_numeric_rule_does_not_accept_other_system_value():
    collected = evidence()
    collected.append({'source': 'monitoring', 'data': {
        'system_name': 'cache-service', 'cpu_percent': 42, 'memory_percent': 89,
        'status': 'degraded', 'alerts': [], 'last_checked': '2026-10-21T09:00:00'}})
    result = validate('1) payments-service CPU 42%입니다. cache-service CPU 91%입니다. 2) 로그 확인이 필요합니다.', collected=collected)
    assert result['validation_status'] == 'rejected'
    assert len([i for i in result['validation_result']['issues'] if i['code'] == 'numeric']) == 2


def review_reply(status='pass', criterion='grounding'):
    import json
    names = ('numeric', 'conditions', 'temporal', 'grounding', 'coverage', 'discrepancy')
    return json.dumps({'checks': [
        {'criterion': c, 'status': status if c == criterion else 'pass', 'quote': '',
         'evidence_ids': [0], 'reason': '로그 수집 근거 없음' if c == criterion else '관측과 일치',
         'correction': '로그를 확인했다고 단정하지 마세요.' if c == criterion else ''}
        for c in names]}, ensure_ascii=False)


class ReviewStub:
    """외부 의미 검토 응답만 대체하고 검증 정책과 그래프 노드는 실제 구현을 사용한다."""
    def __init__(self, content=None, error=None, delay=0):
        self.content, self.error, self.delay = content, error, delay

    async def review(self, *args):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.content, {'usd': 0.001, 'tokens': 200}


def run_review(model, *, timeout=1):
    from agent.validation import AnswerValidator
    return asyncio.run(AnswerValidator(model, timeout_seconds=timeout).validate(
        'payments-service 진단', '1) payments-service CPU 91%입니다. 2) 로그를 조회해서 원인을 확인했습니다.',
        evidence(), ['payments-service']))


def test_semantic_failure_preserves_actionable_reason_and_usage():
    result = run_review(ReviewStub(review_reply('fail')))
    assert result.status == 'rejected'
    assert '로그 수집 근거 없음' in result.feedback()
    assert '단정하지' in result.feedback()
    assert result.usage['tokens'] == 200


@pytest.mark.parametrize('content', ['{}', '{', '{"checks": []}', '{"checks": "pass"}'])
def test_incomplete_or_invalid_review_never_passes(content):
    assert run_review(ReviewStub(content)).status == 'unavailable'


def test_invented_quote_or_evidence_rejected():
    import json
    data = json.loads(review_reply())
    data['checks'][0]['quote'] = '초안에 존재하지 않는 문장'
    assert run_review(ReviewStub(json.dumps(data))).status == 'unavailable'
    data['checks'][0]['quote'] = ''
    data['checks'][0]['evidence_ids'] = [999]
    assert run_review(ReviewStub(json.dumps(data))).status == 'unavailable'


@pytest.mark.parametrize('model', [ReviewStub(error=RuntimeError('secret')), ReviewStub(delay=1)])
def test_review_error_or_timeout_is_not_success(model):
    result = run_review(model, timeout=0.01)
    assert result.status == 'unavailable'
    assert 'secret' not in result.feedback()


def test_unknown_review_is_not_passed():
    assert run_review(ReviewStub(review_reply('unknown'))).status == 'unavailable'


def test_cancellation_propagates():
    async def run():
        from agent.validation import AnswerValidator
        task = asyncio.create_task(AnswerValidator(ReviewStub(delay=10)).validate(
            '진단', '1) 현재 관측을 확인했습니다. 2) 발생 시각과 추가 근거를 확인해 주세요.', [], []))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())


def test_semantic_error_finishes_with_limited_observations_and_real_cost():
    import json
    from agent.validation import AnswerValidator
    result = asyncio.run(validate_node({
        'draft': '1) payments-service CPU 91%입니다. 2) 로그를 확인해 원인을 확정했습니다.',
        'query': '진단', 'resolved_targets': ['payments-service'], 'collected': evidence(),
        'cost_usd': 0.01, 'events': []}, SimpleNamespace(),
        validator=AnswerValidator(ReviewStub('{'))))
    assert result['validation_status'] == 'limited'
    assert result['validation_feedback'] is None
    assert '로그를 확인해 원인을 확정' not in result['draft']
    assert '91' in result['draft'] and '89' in result['draft']
    assert result['cost_usd'] == 0.011


def test_stream_labels_draft_and_limited_result():
    from test_app_runtime import make_test_app, decode_sse
    from fastapi.testclient import TestClient
    from agent.validation import AnswerValidator
    from agent.graph import build_graph
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        app.state.deps.answer_validator = AnswerValidator(ReviewStub('{'))
        app.state.graph = build_graph(app.state.deps)
        events = decode_sse(client.get('/chat/stream', params={'message': 'payments-service CPU 급증 진단'}))
        tokens = [e for e in events if e['type'] == 'AnswerToken']
        assert tokens and all(e.get('phase') == 'draft' for e in tokens)
        assert events[-1]['validation_status'] == 'limited'
        assert '검증을 완료하지 못해' in events[-1]['reply']
        saved = app.state.deps.session_store.load('default')
        assert saved[-1]['content'] == events[-1]['reply']


@pytest.mark.parametrize('backend', ['openai', 'ollama'])
def test_real_review_adapter_sends_grounding_context_and_preserves_usage(monkeypatch, backend):
    import json
    import httpx
    from agent.validation import AnswerValidator
    from agent.openai_llm import OpenAIFastLLM
    from agent.ollama_llm import OllamaFastLLM
    original = httpx.AsyncClient
    requests = []

    async def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        content = review_reply('fail')
        return httpx.Response(200, json=(
            {'model': 'model', 'choices': [{'message': {'content': content}}],
             'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
            if backend == 'openai' else
            {'model': 'model', 'message': {'content': content}, 'prompt_eval_count': 100, 'eval_count': 20}))

    monkeypatch.setenv('OPENAI_API_KEY', 'offline-key')
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(transport), **kw))
    model = (OpenAIFastLLM('configured-fast', 0.005) if backend == 'openai' else OllamaFastLLM('configured-fast', 0.005))
    result = run_review(model)
    assert result.status == 'rejected'
    assert result.usage['tokens'] == 120
    assert result.usage['usd'] == 0.0006
    body = requests[0]
    assert body['model'] == 'configured-fast'
    context = json.loads(body['messages'][1]['content'])
    assert context['evidence'][0]['data']['last_checked'] == '2026-10-21T09:00:00'
    assert context['targets'] == ['payments-service']
    assert '90% 이상' in context['evidence'][2]['data']
    assert body.get('response_format') == {'type': 'json_object'} if backend == 'openai' else body['format']['required'] == ['checks']


def test_rejected_drafts_are_rewritten_with_specific_feedback_and_only_final_is_saved():
    from test_app_runtime import make_test_app
    from fastapi.testclient import TestClient
    app, store, _ = make_test_app()
    feedbacks = []
    async def write(query, collected, attempt, feedback, on_token=None):
        feedbacks.append(feedback)
        value = 91 if attempt == 1 else 34
        draft = f'1) payments-service CPU {value}%입니다. 2) 추가 로그와 발생 시각을 확인해 주세요.'
        return draft, {'usd': 0}
    with TestClient(app) as client:
        app.state.deps.quality_llm.write = write
        response = client.post('/chat', json={'message': 'payments-service CPU 급증 진단', 'thread_id': 'repair'}).json()
    assert response['metadata']['validation_attempts'] == 2
    assert '[numeric]' in feedbacks[1] and '34' in feedbacks[1]
    assert 'CPU 34%' in response['reply']
    assert 'CPU 91%' not in response['reply']


def test_three_rejections_finish_without_publishing_bad_advice():
    from test_app_runtime import make_test_app
    from fastapi.testclient import TestClient
    app, _, _ = make_test_app()
    async def write(*args, **kwargs):
        return '1) payments-service CPU 99%입니다. 2) 로그 확인 없이 즉시 프로세스를 재시작하세요.', {'usd': 0}
    with TestClient(app) as client:
        app.state.deps.quality_llm.write = write
        response = client.post('/chat', json={'message': 'payments-service CPU 급증 진단'}).json()
        saved = app.state.deps.session_store.load('default')
    checks = [e for e in response['events'] if e['type'] == 'Validation']
    assert len(checks) == 3 and checks[-1]['status'] == 'limited'
    assert not any(c['verdict'] == '합격' for c in checks)
    assert 'CPU 34.0%' in response['reply']
    assert '즉시 프로세스를 재시작' not in response['reply']
    assert saved[-1]['content'] == response['reply']


def test_trend_alert_for_other_metric_does_not_prove_cpu_duration():
    result = validate('1) payments-service CPU 91%입니다. 2) CPU가 지속되고 있으므로 인스턴스를 추가하세요.',
                      collected=evidence(alerts=['memory usage climbing steadily over 6h']))
    assert result['validation_status'] == 'rejected'


def test_alert_attributed_trend_is_allowed():
    result = validate('1) payments-service 메모리 89%입니다. 2) 알림에 따르면 메모리가 6시간 증가했습니다. 추가 확인이 필요합니다.',
                      collected=evidence(alerts=['memory usage climbing steadily over 6h']))
    assert result['validation_status'] == 'passed'


@pytest.mark.parametrize('cpu,memory,cpu_ok,memory_ok', [(89,89,False,False),(90,90,False,True),(91,91,True,True)])
def test_review_receives_computed_conditions_not_model_arithmetic(cpu,memory,cpu_ok,memory_ok):
    import json
    from agent.validation import review_input
    context=json.loads(review_input('진단','초안',evidence(cpu=cpu,memory=memory),['payments-service']))
    conditions=context['computed_numeric_conditions']
    assert [(c['metric'],c['threshold'],c['operator'],c['numeric_satisfied']) for c in conditions] == [
        ('cpu_percent',90.0,'>',cpu_ok),('memory_percent',90.0,'>=',memory_ok)]
    assert conditions[0]['duration_satisfied'] is None
    assert conditions[0]['system'] == 'payments-service'


def test_reported_value_is_not_confused_with_current_observation():
    draft = '1) payments-service 사용자 제보는 CPU 99%입니다. 현재 CPU 34%이며 제보와 다릅니다. 2) 발생 시각과 당시 지표를 확인해 주세요.'
    assert validate(draft, collected=evidence(cpu=34))['validation_status'] == 'passed'


def test_default_graph_uses_quality_model_for_semantic_review():
    from test_app_runtime import make_test_app
    from fastapi.testclient import TestClient
    app, _, _ = make_test_app()
    async def forbidden(*args):
        pytest.fail('기본 의미 검토는 경량 모델을 사용하지 않아야 한다')
    async def unavailable(*args):
        return '{}', {'usd': 0}
    with TestClient(app) as client:
        app.state.deps.fast_llm.review = forbidden
        app.state.deps.quality_llm.review = unavailable
        result = client.post('/chat', json={'message':'payments-service CPU 급증 진단'}).json()
    assert result['events'][-2]['status'] == 'limited'


def test_pass_checks_may_omit_inapplicable_correction_but_failure_may_not():
    import json
    data=json.loads(review_reply())
    for c in data['checks']:
        c.pop('correction')
    assert run_review(ReviewStub(json.dumps(data))).status == 'passed'
    data['checks'][0]['status']='fail'
    assert run_review(ReviewStub(json.dumps(data))).status == 'unavailable'


def test_korean_particles_do_not_mix_multi_system_observations():
    from agent.validation import rule_issues
    data=evidence()
    data.append({'source':'monitoring','data':{'system_name':'cache-service','cpu_percent':42,
        'memory_percent':89,'status':'degraded','alerts':[],'last_checked':'2026-10-21T09:00:00'}})
    draft='1) payments-service는 CPU 사용률 91%, cache-service는 CPU 42%, 메모리 89%입니다. 2) 추가 로그를 확인하세요.'
    assert not rule_issues(draft,data,['payments-service','cache-service'])


def test_simultaneous_onset_is_not_proven_by_current_snapshots():
    draft='1) payments-service CPU 91%입니다. 2) 두 서비스의 성능 저하가 동시에 발생한 것은 공유 자원의 문제일 수 있다는 가설입니다.'
    assert validate(draft)['validation_status']=='rejected'


def test_normal_snapshot_requires_incident_time_followup_for_cpu_spike_report():
    from agent.validation import AnswerValidator
    collected=evidence(cpu=34);collected[0]['data']['status']='healthy'
    draft='1) payments-service는 현재 정상 상태이고 CPU 34%입니다. 2) 서비스 상태를 계속 확인하고 런북을 참고하세요.'
    result=asyncio.run(AnswerValidator(FastLLM('fake',0)).validate('payments-service CPU 급증 진단',draft,collected,['payments-service']))
    assert result.status=='rejected'
    assert any(i.code=='discrepancy' for i in result.issues)


def test_simple_health_question_does_not_require_incident_time():
    from agent.validation import AnswerValidator
    collected=evidence(cpu=34);collected[0]['data']['status']='healthy'
    draft='1) payments-service는 현재 정상 상태이고 CPU 34%입니다. 2) 서비스 상태를 계속 확인하고 런북을 참고하세요.'
    result=asyncio.run(AnswerValidator(FastLLM('fake',0)).validate('payments-service 상태 조회',draft,collected,['payments-service']))
    assert result.status=='passed'
