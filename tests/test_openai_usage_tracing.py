"""평가용 래퍼 없이 실제 OpenAI 백엔드가 사용량을 기록하는지 검사한다."""
import asyncio
import copy

import httpx
import pytest
from langsmith import Client, traceable, tracing_context

from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM


@pytest.fixture
def invoke(monkeypatch):
    """외부 HTTP만 대체하고 실제 모델 메서드와 LangSmith RunTree를 사용한다."""
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    client = Client(api_url="http://127.0.0.1:1", api_key="offline-test-key",
                    auto_batch_tracing=False, info={"version": "0"})

    def call(raw_usage, method="route", content=None, status=200, broken_choices=False):
        runs = []
        body = {
            "model": "resolved-model",
            "usage": raw_usage,
            "choices": [{"finish_reason": "stop", "message": {
                "content": content or '{"route":"incident","needs_analysis":false,"tasks":[]}'}}],
        }
        if broken_choices:
            body["choices"] = []
        monkeypatch.setattr("agent.openai_llm.httpx.post", lambda url, **kwargs:
                            httpx.Response(status, json=body, request=httpx.Request("POST", url)))
        fast = OpenAIFastLLM("gpt-4o-mini", 0.00015)
        quality = OpenAIQualityLLM("gpt-4o", 0.005)
        extra = {"on_end": runs.append}

        @traceable(name="test_parent")
        async def parent():
            if method == "route":
                return fast.route("CPU 확인", langsmith_extra=extra)
            if method == "plan":
                return fast.plan("CPU 확인", "확인 절차", langsmith_extra=extra)
            if method == "analyze":
                return fast.analyze([], langsmith_extra=extra)
            if method == "chat":
                return await fast.chat("안녕", [], langsmith_extra=extra)
            return await quality.write("CPU 확인", [], 1, None, langsmith_extra=extra)

        with tracing_context(enabled="local", client=client):
            try:
                result = asyncio.run(parent(langsmith_extra={"on_end": runs.append}))
                error = None
            except Exception as exc:
                result, error = None, exc
        return runs, result, error

    yield call
    client.close()


@pytest.mark.parametrize("method", ["route", "plan", "analyze", "chat", "write"])
def test_all_model_calls_record_provider_usage_on_existing_llm_span(invoke, method):
    raw = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
           "prompt_tokens_details": {"cached_tokens": 40},
           "completion_tokens_details": {"reasoning_tokens": 5}}
    original = copy.deepcopy(raw)
    runs, result, error = invoke(raw, method)
    assert error is None
    assert raw == original
    assert len(runs) == 2
    llm, parent = runs
    assert llm.parent_run_id == parent.id
    assert llm.metadata["ls_provider"] == "openai"
    assert llm.metadata["ls_model_name"] == "resolved-model"
    assert llm.metadata["usage_metadata"] == {
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "input_token_details": {"cache_read": 40},
        "output_token_details": {"reasoning": 5},
    }
    assert "usage_metadata" not in parent.metadata
    assert result[1]["tokens"] == 120
    assert result[1]["usd"] == (0.0006 if method == "write" else 0.000018)
    assert "total_cost" not in llm.metadata["usage_metadata"]
    assert "offline-test-key" not in str(llm.metadata)


@pytest.mark.parametrize("raw", [
    None, {}, {"total_tokens": 10},
    {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
    {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
    {"prompt_tokens": 1.5, "completion_tokens": 1, "total_tokens": 2.5},
    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99},
])
def test_unknown_or_invalid_usage_is_marked_without_inventing_standard_zero(invoke, raw):
    runs, result, error = invoke(raw)
    assert error is None
    assert result[0]["route"] == "incident"
    assert runs[0].metadata["provider_usage_status"] == "missing_or_invalid"
    assert "usage_metadata" not in runs[0].metadata


def test_measured_zero_is_recorded(invoke):
    runs, _, error = invoke({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    assert error is None
    assert runs[0].metadata["usage_metadata"]["total_tokens"] == 0


@pytest.mark.parametrize("broken_choices", [False, True])
def test_usage_survives_output_parsing_failure_or_fallback(invoke, broken_choices):
    runs, _, error = invoke(
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        content="not JSON", broken_choices=broken_choices)
    assert runs[0].metadata["usage_metadata"]["total_tokens"] == 120
    assert isinstance(error, IndexError) if broken_choices else error is None


def test_http_failure_does_not_record_fake_usage(invoke):
    runs, _, error = invoke(None, status=429)
    assert isinstance(error, httpx.HTTPStatusError)
    assert "usage_metadata" not in runs[0].metadata
