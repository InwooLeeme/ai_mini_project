"""LLM_BACKEND=openai — 실제 OpenAI API 백엔드 (fake_llm/ollama_llm과 동일 인터페이스).

Wave 10a 구도를 실제 상용 API로: fast(기본 gpt-4o-mini)는 분류·계획·단순 대화,
quality(기본 gpt-4o)는 최종 답변 작성 전담. ollama_llm.py와 마찬가지로 실모델이라
결정론은 아니다 — gate(W6)·replan(W7)·validator(W8)가 연출이 아니라 실제 방어선으로
동작한다. Ollama와 달리 실제 과금이 발생하는 원격 API이므로, 이 백엔드로 실행하면
OpenAI 계정에 실제 비용이 청구된다(run_openai.sh 실행 전 반드시 인지할 것).

구조화 출력은 OpenAI Chat Completions의 `response_format={"type": "json_object"}`
(JSON 모드)로 강제한다 — Ollama의 `format=<JSON Schema>`처럼 스키마 자체를 강제하는
기능은 아니라서, 시스템 프롬프트에 원하는 필드를 명시하고 ollama_llm.py와 동일하게
코드 레벨 방어(파싱 실패·필드 누락 시 보정)를 그대로 둔다.

비용(cost_usd)은 OpenAI 응답의 `usage.total_tokens`(실측) × 가상 단가다 — 실제 청구
금액과는 별개로, W10a의 모델 분리 효과를 fake/ollama와 같은 잣대로 비교하기 위함이다.
(OpenAI는 입력/출력 토큰 단가가 실제로는 다르지만, 세 백엔드의 인터페이스를
`usd_per_1k_tokens` 하나로 통일해두기 위해 여기서도 단일 블렌디드 단가로 단순화한다.)

사전 조건: 유효한 `OPENAI_API_KEY` 환경변수(.env).
"""
import os

import httpx
from langsmith import get_current_run_tree, traceable
from .analysis_policy import ANALYSIS_INSTRUCTIONS, analysis_facts
from .llm_roles import (PLAN_SYSTEM, ROUTE_SYSTEM, build_chat_prompt, parse_plan, parse_route,
                        plan_user, rendered_prompt, token_usage)
from .writer import build_writer_prompt

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# JSON 모드는 "유효한 JSON"만 보장할 뿐 스키마까지 강제하진 않으므로 공통 지시 뒤에
# 원하는 필드를 텍스트로 덧붙인다(Ollama의 format=<JSON Schema>에 대응).
ROUTE_INSTRUCTIONS = (
    'JSON 객체로만 응답하라. 스키마: '
    '{"route": "chat" 또는 "incident", "needs_analysis": true 또는 false}'
)

PLAN_INSTRUCTIONS = (
    'JSON 객체로만 응답하라. 스키마: {"needs_analysis": boolean, '
    '"tasks": [{"agent": "research"|"analysis", "target": string(선택)}]}'
)


def _record_openai_usage(response: dict, requested_model: str) -> None:
    """기존 LLM 추적에 실측 사용량을 기록한다. 상위 실행에 중복 기록하지 않는다."""
    run = get_current_run_tree()
    if run is None or run.run_type != "llm":
        return
    raw = response.get("usage")
    raw = raw if isinstance(raw, dict) else {}
    counts = [raw.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")]
    valid = (all(type(value) is int and value >= 0 for value in counts)
             and counts[2] == counts[0] + counts[1])
    metadata = {
        "ls_provider": "openai", "ls_model_type": "chat",
        "ls_model_name": response.get("model") or requested_model,
        "requested_model": requested_model,
        "provider_usage_status": "recorded" if valid else "missing_or_invalid",
        # 기존 화면의 usd는 가상 단가 계산값이며, 실제 사용량 기반 비용과 구분한다.
        "legacy_cost_usd_semantics": "simulated_cost_usd",
    }
    if valid:
        usage = dict(zip(("input_tokens", "output_tokens", "total_tokens"), counts))
        for source, destination, limit, mapping in (
            ("prompt_tokens_details", "input_token_details", counts[0],
             {"cached_tokens": "cache_read", "audio_tokens": "audio"}),
            ("completion_tokens_details", "output_token_details", counts[1],
             {"reasoning_tokens": "reasoning", "audio_tokens": "audio"}),
        ):
            details = raw.get(source)
            if isinstance(details, dict):
                mapped = {target: details[key] for key, target in mapping.items()
                          if type(details.get(key)) is int and 0 <= details[key] <= limit}
                if mapped:
                    # 캐시·추론 토큰은 이미 합계에 포함되므로 다시 더하지 않는다.
                    usage[destination] = mapped
        metadata["usage_metadata"] = usage
    run.add_metadata(metadata)

class OpenAILLM:
    def __init__(self, model: str, usd_per_1k_tokens: float):
        self.name = f"openai:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def _request(self, system: str, user: str, json_mode: bool = False) -> tuple[dict, dict]:
        """동기·비동기 호출이 동일한 모델·인증·입력 설정을 사용하도록 구성한다."""
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않았습니다 — .env 또는 환경변수로 지정하세요."
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        return payload, headers

    def _response(self, resp: httpx.Response) -> tuple[str, dict]:
        """호출 방식과 관계없이 실제 사용량과 기존 반환 계약을 보존한다."""
        resp.raise_for_status()
        data = resp.json()
        # 출력 파싱 실패나 상위 JSON 보정 여부와 무관하게 실측 사용량을 보존한다.
        _record_openai_usage(data, self.model)
        content = data["choices"][0]["message"]["content"]
        raw_usage = data.get("usage")
        tokens = raw_usage.get("total_tokens") if isinstance(raw_usage, dict) else None
        # 기존 반환 계약의 기본값만 유지하며, 미측정값을 표준 추적의 0으로 기록하지 않는다.
        if type(tokens) is not int or tokens < 0:
            tokens = 0
        return content, token_usage(self.name, tokens, self.usd_per_1k)

    def _chat(self, system: str, user: str, json_mode: bool = False) -> tuple[str, dict]:
        payload, headers = self._request(system, user, json_mode)
        return self._response(httpx.post(OPENAI_URL, json=payload, headers=headers, timeout=120.0))

    async def _achat(self, system: str, user: str, json_mode: bool = False) -> tuple[str, dict]:
        """분석 요청을 비차단 HTTP로 실행하고 취소 시 연결을 정리한다."""
        payload, headers = self._request(system, user, json_mode)
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENAI_URL, json=payload, headers=headers)
            return self._response(response)

    @traceable(run_type="llm", name="openai.review")
    async def review(self, query, draft, collected, targets):
        from .validation import REVIEW_INSTRUCTIONS, review_input
        return await self._achat(REVIEW_INSTRUCTIONS,
                                 review_input(query, draft, collected, targets), json_mode=True)


class OpenAIFastLLM(OpenAILLM):
    """분류·계획·단순 대화 전담 (저비용 모델, 기본 gpt-4o-mini)."""

    @traceable(run_type="llm", name="openai_fast.route")
    def route(self, query: str) -> tuple[dict, dict]:
        content, usage = self._chat(ROUTE_SYSTEM + " " + ROUTE_INSTRUCTIONS, query, json_mode=True)
        return parse_route(content, query), usage

    @traceable(run_type="llm", name="openai_fast.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        content, usage = self._chat(PLAN_SYSTEM + " " + PLAN_INSTRUCTIONS,
                                    plan_user(query, procedure_prompt), json_mode=True)
        return parse_plan(content, query), usage

    @traceable(run_type="llm", name="openai_fast.analyze")
    async def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        """현재 관측의 비교 가설을 반환하며 RAG 실행을 막지 않는다."""
        return await self._achat(ANALYSIS_INSTRUCTIONS, analysis_facts(collected))

    @traceable(run_type="llm", name="openai_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        """ollama_llm.py와 인터페이스(on_token 콜백)만 맞춘다 — OpenAI Chat Completions를
        아직 stream=True로 호출하지 않아 토큰 단위 중계는 하지 않고 완성된 응답을 한 번에
        전달한다."""
        system, user = build_chat_prompt(query, history)
        content, usage = self._chat(system, user)
        usage["prompt"] = rendered_prompt(system, user)
        if on_token is not None:
            on_token(content)
        return content, usage

class OpenAIQualityLLM(OpenAILLM):
    """최종 답변 작성 전담 (고품질 모델, 기본 gpt-4o)."""

    @traceable(run_type="llm", name="openai_quality.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
              feedback: str | None, on_token=None) -> tuple[str, dict]:
        """ollama_llm.py와 인터페이스(on_token 콜백)만 맞춘다 — OpenAI Chat Completions를
        아직 stream=True로 호출하지 않아 토큰 단위 중계는 하지 않고 완성된 응답을 한 번에
        전달한다."""
        system, user = build_writer_prompt(query, collected, feedback)
        content, usage = self._chat(system, user)
        usage["prompt"] = rendered_prompt(system, user)
        if on_token is not None:
            on_token(content)
        return content, usage
