"""평가 계측 전용. 실제 사용량/가격표 추정 비용과 baseline 가상 비용을 분리한다."""
from functools import wraps

from langsmith import get_current_run_tree


SCHEMA_VERSION = "openai-usage-v1"
COST_POLICY = {
    "cost_schema": SCHEMA_VERSION,
    "langsmith_cost_basis": "provider_usage_x_langsmith_model_pricing_estimate",
    "actual_billed_usd": None,
    "actual_billed_status": "not_verified_against_invoice",
    "legacy_cost_usd_semantics": "simulated_cost_usd",
}


def normalize_usage(raw):
    """누락/잘못된 usage를 0으로 만들지 않는다. 하위 토큰은 합계에 중복 가산하지 않는다."""
    if not isinstance(raw, dict):
        return None

    def count(value):
        return type(value) is int and value >= 0

    input_tokens = raw.get("prompt_tokens")
    output_tokens = raw.get("completion_tokens")
    if not count(input_tokens) or not count(output_tokens):
        return None
    total = raw.get("total_tokens", input_tokens + output_tokens)
    if not count(total) or total != input_tokens + output_tokens:
        return None
    result = {"input_tokens": input_tokens, "output_tokens": output_tokens,
              "total_tokens": total}
    for source, destination, limit, mapping in (
        ("prompt_tokens_details", "input_token_details", input_tokens,
         {"cached_tokens": "cache_read", "audio_tokens": "audio"}),
        ("completion_tokens_details", "output_token_details", output_tokens,
         {"reasoning_tokens": "reasoning", "audio_tokens": "audio"}),
    ):
        details = raw.get(source)
        if details is None:
            continue
        if not isinstance(details, dict):
            return None
        mapped = {}
        for provider_key, standard_key in mapping.items():
            if provider_key not in details:
                continue
            value = details[provider_key]
            if not count(value) or value > limit:
                return None
            mapped[standard_key] = value
        if mapped:
            result[destination] = mapped
    return result


def attach_openai_usage(response, requested_model):
    """기존 LLM span에만 기록한다. 새 span/상위 usage를 만들지 않아 중복 집계를 방지한다."""
    run = get_current_run_tree()
    if run is None or run.run_type != "llm":
        return
    normalized = normalize_usage(response.get("usage"))
    metadata = {
        **COST_POLICY,
        "ls_provider": "openai",
        "ls_model_type": "chat",
        "ls_model_name": response.get("model") or requested_model,
        "requested_model": requested_model,
        "provider_usage_status": "recorded" if normalized is not None else "missing_or_invalid",
    }
    run.set(metadata=metadata)
    if normalized is not None:
        # Cost 필드를 직접 넣지 않는다. LangSmith의 모델별 가격표가 추정 비용을 계산한다.
        run.set(usage_metadata=normalized)


def with_virtual_cost_metadata(original_chat):
    """원래 모델 인스턴스의 가상 단가와 반환 비용을 보존한다."""
    @wraps(original_chat)
    def wrapped(self, *args, **kwargs):
        result = original_chat(self, *args, **kwargs)
        run = get_current_run_tree()
        if run is not None and run.run_type == "llm":
            run.set(metadata={
                **COST_POLICY,
                "simulated_cost_usd": result[1]["usd"],
                "simulated_usd_per_1k_tokens": self.usd_per_1k,
            })
        return result
    return wrapped
