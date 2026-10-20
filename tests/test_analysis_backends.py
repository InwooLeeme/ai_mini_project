"""분석 Adapter의 실제 비동기 HTTP 실행·취소와 반환 계약을 확인한다."""
import asyncio
import json

import httpx
import pytest

from agent.openai_llm import OpenAIFastLLM
from agent.ollama_llm import OllamaFastLLM
from agent.fake_llm import FastLLM


@pytest.mark.parametrize("backend", ["openai", "ollama"])
@pytest.mark.parametrize("cancel", [False, True])
def test_network_analysis_yields_to_other_work_and_cleans_up(monkeypatch, backend, cancel):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    original = httpx.AsyncClient
    calls = []

    async def exercise():
        entered, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def transport(request):
            calls.append(json.loads(request.content))
            entered.set()
            try:
                await release.wait()
                if backend == "openai":
                    return httpx.Response(200, json={
                        "model": "resolved-model",
                        "choices": [{"message": {"content": "추가 근거가 필요합니다"}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                    })
                return httpx.Response(200, text=json.dumps({
                    "message": {"content": "추가 근거가 필요합니다"}, "done": True,
                    "prompt_eval_count": 100, "eval_count": 20,
                }) + "\n")
            finally:
                cleaned.set()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
                            original(transport=httpx.MockTransport(transport), **kwargs))
        model = (OpenAIFastLLM("test-openai", 0.005) if backend == "openai"
                 else OllamaFastLLM("test-ollama", 0.005))
        task = asyncio.create_task(model.analyze([{
            "source": "monitoring",
            "data": {"system_name": "a", "status": "degraded", "cpu_percent": 91, "alerts": []},
        }]))
        await asyncio.wait_for(entered.wait(), 1)
        assert not task.done()  # HTTP 대기 중에도 이 코루틴이 실행돼야 한다.
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            text, usage = await asyncio.wait_for(task, 1)
            assert text == "추가 근거가 필요합니다"
            assert usage["tokens"] == 120
            assert usage["usd"] == 0.0006
        assert cleaned.is_set()

    asyncio.run(exercise())
    assert len(calls) == 1


def test_fake_analysis_uses_same_async_interface_without_claiming_simultaneous_faults():
    model = FastLLM("test", 0.005)
    text, usage = asyncio.run(model.analyze([
        {"source": "monitoring", "data": {"system_name": name, "status": "degraded"}}
        for name in ("a", "b")
    ]))
    assert "가설" in text
    assert "동시에" not in text
    assert usage["tokens"] > 0
