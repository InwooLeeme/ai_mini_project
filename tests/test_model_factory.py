"""모델 생성 모듈의 환경변수·기본값·fake 대체 동작을 검사한다."""
import pytest


@pytest.mark.parametrize("backend", [None, "fake", "unknown"])
def test_factory_preserves_fake_fallback(monkeypatch, backend):
    from agent.model_factory import build_llms, build_answer_validator
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    if backend is not None:
        monkeypatch.setenv("LLM_BACKEND", backend)
    selected, fast, writer = build_llms()
    assert selected == (backend or "fake")
    assert fast.name == "fast-model-stub"
    assert writer.name == "quality-model-stub"
    assert build_answer_validator(selected, writer).model is writer


@pytest.mark.parametrize("backend,default", [("openai", "gpt-4o-mini"), ("ollama", "qwen3:0.6b")])
@pytest.mark.parametrize("override", [None, "custom-fast", ""])
def test_factory_preserves_fast_model_selection(monkeypatch, backend, default, override):
    from agent.model_factory import build_llms
    monkeypatch.setenv("LLM_BACKEND", backend)
    key = f"{backend.upper()}_FAST_MODEL"
    monkeypatch.delenv(key, raising=False)
    if override is not None:
        monkeypatch.setenv(key, override)
    selected, fast, _ = build_llms()
    assert selected == backend
    assert fast.model == (default if override is None else override)
    assert fast.usd_per_1k == 0.00015
