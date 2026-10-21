"""실행 환경에서 Writer와 Validator 모델 설정이 독립적으로 적용되는지 검사한다."""
from functools import partial

import pytest
from fastapi.testclient import TestClient

from agent import runtime
from agent.session_store import SqliteSessionStore
from test_app_runtime import StubMCP
import server


@pytest.mark.parametrize('backend,default', [('openai', 'gpt-4o'), ('ollama', 'qwen3:8b')])
@pytest.mark.parametrize('writer,validator', [
    (None, None), ('writer-custom', None), (None, 'validator-custom'),
    ('writer-custom', 'validator-custom'), (None, ''),
])
def test_runtime_selects_writer_and_validator_independently(monkeypatch, backend, default, writer, validator):
    monkeypatch.setenv('LLM_BACKEND', backend)
    monkeypatch.setenv('SUPERVISOR_VERSION', 'v2')
    for suffix, value in [('QUALITY', writer), ('VALIDATOR', validator)]:
        key = f'{backend.upper()}_{suffix}_MODEL'
        monkeypatch.delenv(key, raising=False)
        if value is not None:
            monkeypatch.setenv(key, value)
    app = server.create_app(runtime_factory=partial(
        runtime.create_runtime, mcp_factory=StubMCP,
        session_store_factory=lambda: SqliteSessionStore(':memory:'), retriever_factory=None,
    ))
    with TestClient(app) as client:
        deps = app.state.deps
        assert deps.answer_validator is not None
        assert deps.quality_llm.model == (writer or default)
        assert deps.answer_validator.model.model == (validator or default)
        assert deps.answer_validator.model is not deps.quality_llm
        health = client.get('/health').json()
        assert health['quality_llm'] == f'{backend}:{writer or default}'
        assert health['validator_llm'] == f'{backend}:{validator or default}'


def test_legacy_runtime_does_not_construct_validator(monkeypatch):
    monkeypatch.setenv('LLM_BACKEND', 'fake')
    monkeypatch.setenv('SUPERVISOR_VERSION', 'v1')
    app = server.create_app(runtime_factory=partial(
        runtime.create_runtime, mcp_factory=StubMCP,
        session_store_factory=lambda: SqliteSessionStore(':memory:'), retriever_factory=None,
    ))
    with TestClient(app) as client:
        assert app.state.deps.answer_validator is None
        assert client.get('/health').json()['validator_llm'] is None


@pytest.mark.parametrize('backend', ['openai', 'ollama'])
@pytest.mark.parametrize('compact', [False, True])
def test_graph_sends_writing_and_review_to_separate_models(monkeypatch, backend, compact):
    import json
    import httpx
    from agent.fake_llm import FastLLM
    from test_answer_validation import review_reply
    from test_app_runtime import make_test_app

    monkeypatch.setenv('LLM_BACKEND', backend)
    monkeypatch.setenv('SUPERVISOR_VERSION', 'v2')
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-key')
    monkeypatch.setenv(f'{backend.upper()}_QUALITY_MODEL', 'writer-custom')
    monkeypatch.setenv(f'{backend.upper()}_VALIDATOR_MODEL', 'validator-custom')
    requests = []
    draft = '1) web-01 CPU 34%입니다. 2) 추가 로그와 발생 시각을 확인해 주세요.'

    def response(request):
        body = json.loads(request.content)
        requests.append(body)
        review = json.loads(review_reply())
        if compact:
            review['checks'] = [{key: check[key] for key in ('criterion', 'status', 'evidence_ids')}
                                for check in review['checks']]
        content = draft if body['model'] == 'writer-custom' else json.dumps(review)
        payload = ({'model': body['model'], 'choices': [{'message': {'content': content}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
                   if backend == 'openai' else
                   {'model': body['model'], 'message': {'content': content}, 'done': True,
                    'prompt_eval_count': 100, 'eval_count': 20})
        return httpx.Response(200, json=payload, request=request)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original_client(
        transport=httpx.MockTransport(response), **kw))
    monkeypatch.setattr(httpx, 'post', lambda url, *, json, **kw: response(
        httpx.Request('POST', url, json=json)))
    fixture_app, _, _ = make_test_app()
    app = server.create_app(runtime_factory=partial(
        runtime.create_runtime, mcp_factory=StubMCP,
        session_store_factory=lambda: SqliteSessionStore(':memory:'), retriever_factory=None,
    ))
    with TestClient(fixture_app), TestClient(app) as client:
        deps = app.state.deps
        deps.fast_llm = FastLLM('offline-fast', 0)
        deps.evidence_collector = fixture_app.state.deps.evidence_collector
        result = client.post('/chat', json={'message': 'web-01 장애 진단'}).json()
        assert result['reply'] == draft
        checks = [e for e in result['events'] if e['type'] == 'Validation']
        assert checks[-1]['status'] == 'passed'
        assert len(checks[-1]['checks']) == 6
        assert all(set(c) == {'criterion', 'status', 'quote', 'evidence_ids', 'reason', 'correction'}
                   for c in checks[-1]['checks'])
        assert checks[-1]['usage']['model'] == f'{backend}:validator-custom'
        assert [r['model'] for r in requests] == ['writer-custom', 'validator-custom']


def test_health_supports_injected_review_model_without_display_name():
    from agent.validation import AnswerValidator
    from test_answer_validation import ReviewStub
    from test_app_runtime import make_test_app
    app, _, _ = make_test_app()
    with TestClient(app) as client:
        app.state.deps.answer_validator = AnswerValidator(ReviewStub())
        assert client.get('/health').json()['validator_llm'] is None
