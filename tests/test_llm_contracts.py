"""세 백엔드 구현체가 fast·quality 역할 계약의 메서드 형태를 그대로 따르는지 검사한다."""
import inspect

import pytest

from agent.fake_llm import FastLLM, QualityLLM
from agent.llm_roles import FastModel, QualityModel
from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM


def protocol_methods(protocol):
    return {name: getattr(protocol, name) for name in dir(protocol)
            if not name.startswith("_") and inspect.isfunction(getattr(protocol, name))}


def positional_params(fn):
    # traceable이 덧붙이는 키워드 전용 config는 계약에 포함하지 않는다.
    return [p.name for p in inspect.signature(fn).parameters.values()
            if p.kind is not inspect.Parameter.KEYWORD_ONLY]


@pytest.mark.parametrize("cls,protocol", [
    (FastLLM, FastModel), (OllamaFastLLM, FastModel), (OpenAIFastLLM, FastModel),
    (QualityLLM, QualityModel), (OllamaQualityLLM, QualityModel),
    (OpenAIQualityLLM, QualityModel),
])
def test_backend_follows_role_protocol(cls, protocol):
    assert "name" in protocol.__annotations__
    assert isinstance(cls("m", 0.001).name, str)
    methods = protocol_methods(protocol)
    assert methods
    for name, expected in methods.items():
        actual = getattr(cls, name)
        assert inspect.iscoroutinefunction(actual) == inspect.iscoroutinefunction(expected), name
        assert positional_params(actual) == positional_params(expected), name


def test_quality_model_includes_review_for_default_validator():
    assert "review" in protocol_methods(QualityModel)
