"""백엔드별 모델 설정과 생성을 한곳에서 관리한다. 생성 시 모델 호출은 하지 않는다."""
import os

from .fake_llm import FastLLM, QualityLLM
from .validation import AnswerValidator


def build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API)."""
    backend = os.environ.get("LLM_BACKEND", "fake")
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
        fast = OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"),
                             usd_per_1k_tokens=0.00015)
        quality = OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
    quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
    return backend, fast, quality


def build_answer_validator(backend: str, fallback_model) -> AnswerValidator:
    """실모델 검증 설정은 Writer와 독립적이며 fake/주입 모델은 기존 동작을 유지한다."""
    if backend == "openai":
        from .openai_llm import OpenAILLM
        model = OpenAILLM(os.environ.get("OPENAI_VALIDATOR_MODEL") or "gpt-4o",
                          usd_per_1k_tokens=0.005)
    elif backend == "ollama":
        from .ollama_llm import OllamaLLM
        model = OllamaLLM(os.environ.get("OLLAMA_VALIDATOR_MODEL") or "qwen3:8b",
                          usd_per_1k_tokens=0.005)
    else:
        model = fallback_model
    return AnswerValidator(model)
