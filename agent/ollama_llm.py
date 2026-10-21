"""LLM_BACKEND=ollama — 로컬 Ollama 실모델 백엔드 (fake_llm과 동일 인터페이스).

Wave 10a 구도를 실물로: fast(기본 qwen3:0.6b)는 분류·계획·단순 대화, quality(기본
qwen3:8b)는 최종 답변 작성 전담. Ollama의 구조화 출력(format=<JSON Schema>)과
temperature=0으로 변동을 줄이지만 실모델이므로 결정론은 아니다 — 그래서 gate(W6)·
replan(W7)·validator(W8)가 연출이 아니라 실제 방어선으로 동작한다.
fc
비용(cost_usd)은 '가상 단가 × 실측 토큰 수(prompt_eval_count+eval_count)'다 — 로컬
추론이라 실제 과금은 0이지만, W10a의 모델 분리 효과를 같은 잣대로 비교하기 위함이다.

사전 조건: `ollama serve` 실행 중 + `ollama pull qwen3:0.6b qwen3:8b`.
"""
import json

import httpx
from langsmith import traceable
from .analysis_policy import ANALYSIS_INSTRUCTIONS, analysis_facts
from .llm_roles import (PLAN_SCHEMA, PLAN_SYSTEM, ROUTE_SCHEMA, ROUTE_SYSTEM, build_chat_prompt,
                        parse_plan, parse_route, plan_user, rendered_prompt, token_usage)
from .writer import build_writer_prompt

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

class OllamaLLM:
    def __init__(self, model: str, usd_per_1k_tokens: float):
        self.name = f"ollama:{model}"
        self.model = model
        self.usd_per_1k = usd_per_1k_tokens

    def _chat(self, system: str, user: str, schema: dict | None = None) -> tuple[str, dict]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if schema is not None:
            payload["format"] = schema
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        return content, token_usage(self.name, tokens, self.usd_per_1k)

    async def _chat_stream(self, system: str, user: str, on_token=None) -> tuple[str, dict]:
        """사용자 대면 텍스트(최종 답변류) 전용 — Ollama NDJSON 스트림을 토큰 단위로
        받아 on_token(조각)을 즉시 호출해 SSE로 실시간 중계할 수 있게 한다. 분류/계획
        등 JSON 스키마 출력(route/plan)은 _chat을 쓰고, 분석은 비차단 실행을 위해
        콜백 없이 이 메서드에서 완성된 내용을 받는다."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": True,
            "options": {"temperature": 0.0},
        }
        chunks: list[str] = []
        usage = token_usage(self.name, 0, self.usd_per_1k)
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        chunks.append(piece)
                        if on_token is not None:
                            on_token(piece)
                    if data.get("done"):
                        tokens = ((data.get("prompt_eval_count") or 0)
                                  + (data.get("eval_count") or 0))
                        usage = token_usage(self.name, tokens, self.usd_per_1k)
        usage["prompt"] = rendered_prompt(system, user)
        return "".join(chunks), usage

    @traceable(run_type="llm", name="ollama.review")
    async def review(self, query, draft, collected, targets):
        from .validation import REVIEW_INSTRUCTIONS, REVIEW_SCHEMA, review_input
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": REVIEW_INSTRUCTIONS},
            {"role": "user", "content": review_input(query, draft, collected, targets)}],
            "stream": False, "format": REVIEW_SCHEMA, "options": {"temperature": 0.0}}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        tokens = (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
        return data["message"]["content"], token_usage(self.name, tokens, self.usd_per_1k)


class OllamaFastLLM(OllamaLLM):
    """분류·계획·단순 대화 전담 (저사양 소형 모델)."""

    @traceable(run_type="llm", name="ollama_fast.route")
    def route(self, query: str) -> tuple[dict, dict]:
        # Ollama는 format=<JSON Schema>로 스키마 자체를 강제한다.
        content, usage = self._chat(ROUTE_SYSTEM, query, schema=ROUTE_SCHEMA)
        return parse_route(content, query), usage

    @traceable(run_type="llm", name="ollama_fast.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        content, usage = self._chat(PLAN_SYSTEM, plan_user(query, procedure_prompt), schema=PLAN_SCHEMA)
        return parse_plan(content, query), usage

    @traceable(run_type="llm", name="ollama_fast.analyze")
    async def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        """비차단 호출로 현재 관측의 비교 가설을 생성한다."""
        return await self._chat_stream(ANALYSIS_INSTRUCTIONS, analysis_facts(collected))

    @traceable(run_type="llm", name="ollama_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        system, user = build_chat_prompt(query, history)
        return await self._chat_stream(system, user, on_token=on_token)

class OllamaQualityLLM(OllamaLLM):
    """최종 답변 작성 전담 (상대적으로 큰 모델)."""

    @traceable(run_type="llm", name="ollama_quality.write")
    async def write(self, query: str, collected: list[dict], attempt: int,
              feedback: str | None, on_token=None) -> tuple[str, dict]:
        system, user = build_writer_prompt(query, collected, feedback)
        return await self._chat_stream(system, user, on_token=on_token)
