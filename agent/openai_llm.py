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
import json
import os
import re

import httpx
from langsmith import get_current_run_tree, traceable
from .analysis_policy import ANALYSIS_INSTRUCTIONS
from .targets import KNOWN_SYSTEMS, normalize_plan, resolve_targets

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

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
        usage = {"model": self.name, "tokens": tokens,
                 "usd": round(tokens / 1000 * self.usd_per_1k, 6)}
        return content, usage

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
        system = ("당신은 ICT 운영 지원 라우터다. 사용자 요청을 분류하라. "
                  "장애·성능 저하·지연·진단·모니터링·티켓 등 운영 이슈면 route=incident, "
                  "인사·잡담·일반 질문이면 route=chat. "
                  "needs_analysis는 패턴 비교/통계 분석이 꼭 필요할 때만 true. "
                  + ROUTE_INSTRUCTIONS)
        content, usage = self._chat(system, query, json_mode=True)
        try:
            raw = json.loads(content)
            route = raw.get("route") if raw.get("route") in ("chat", "incident") else None
            result = {"route": route or "chat",
                      "needs_analysis": bool(raw.get("needs_analysis", False))}
        except (json.JSONDecodeError, AttributeError):
            # JSON 모드는 "유효한 JSON"만 보장할 뿐 스키마까지 강제하진 않으므로,
            # 필드 누락·오탐 시 코드 레벨 fallback을 둔다(W6 정신, ollama_llm.py와 동일).
            incident = bool(re.search(r"장애|진단|다운|지연|급증|incident|티켓", query))
            result = {"route": "incident" if incident else "chat",
                      "needs_analysis": False}
        return result, usage

    @traceable(run_type="llm", name="openai_fast.plan")
    def plan(self, query: str, procedure_prompt: str) -> tuple[dict, dict]:
        system = ("당신은 ICT 운영 Supervisor의 planning 노드다. 표준 진단 절차에 따라 "
                  "실행 계획(tasks)을 JSON으로 세워라. "
                  "사용자 요청에 등장한 시스템마다 agent=research task를 하나씩 세워라 "
                  f"(알려진 시스템: {', '.join(sorted(KNOWN_SYSTEMS))} — 특정 시스템이 "
                  "언급되지 않았으면 조사 대상을 임의로 만들지 마라). "
                  "패턴 분석이 꼭 필요하다고 판단되면 agent=analysis task를 추가해도 된다. "
                  + PLAN_INSTRUCTIONS)
        user = f"## 표준 절차\n{procedure_prompt}\n\n## 사용자 요청\n{query}"
        content, usage = self._chat(system, user, json_mode=True)

        # 코드 레벨 정규화 — JSON 형식은 지켜져도 내용은 흔들릴 수 있으므로
        # (대상 시스템 오타, task 누락/중복) 실행 가능한 형태로 보정한다.
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            raw = {"needs_analysis": False, "tasks": []}
        result = normalize_plan(raw, resolve_targets(query).targets)
        return result, usage

    @traceable(run_type="llm", name="openai_fast.analyze")
    async def analyze(self, collected: list[dict]) -> tuple[str, dict]:
        """현재 관측의 비교 가설을 반환하며 RAG 실행을 막지 않는다."""
        mon = [c["data"] for c in collected if c["source"] == "monitoring"]
        facts = "\n".join(
            f"- {d['system_name']}: 상태={d['status']}, CPU {d['cpu_percent']}%, "
            f"알림 {d['alerts']}" for d in mon) or "(모니터링 데이터 없음)"
        return await self._achat(ANALYSIS_INSTRUCTIONS, facts)

    @traceable(run_type="llm", name="openai_fast.chat")
    async def chat(self, query: str, history: list[dict], on_token=None) -> tuple[str, dict]:
        """ollama_llm.py와 인터페이스(on_token 콜백)만 맞춘다 — OpenAI Chat Completions를
        아직 stream=True로 호출하지 않아 토큰 단위 중계는 하지 않고 완성된 응답을 한 번에
        전달한다."""
        system = ("당신은 간결한 한국어 비서다. 한두 문장으로 답하라. "
                  "대화 이력에서 사용자가 밝힌 이름이 있으면 정확히 기억해 답하라.")
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
        user = (f"## 지금까지의 대화\n{rendered or '(없음)'}\n\n"
                f"## 사용자의 새 메시지\n{query}")
        content, usage = self._chat(system, user)
        usage["prompt"] = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
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
        system = ("당신은 ICT 운영 보고 작성자다. 수집된 사실만 사용해 한국어로 "
                  "간결한 대응 보고를 작성하라. 반드시 '1) ... 2) ... 3) ...' 형식의 "
                  "번호 절차로 답하고, 수치는 수집된 값을 그대로 인용하라.")
        facts = []
        for c in collected:
            if c["source"] == "monitoring":
                d = c["data"]
                facts.append(f"- 모니터링 {d['system_name']}: 상태={d['status']}, "
                             f"CPU {d['cpu_percent']}%, 메모리 {d['memory_percent']}%, "
                             f"조회 시각={d.get('last_checked', '미확인')}, 알림 {len(d['alerts'])}건 {d['alerts']}")
            elif c["source"] == "runbook":
                facts.append(f"- runbook 출처: {c.get('uri', '(기존 문서)')}\n{c['data'].strip()}")
            elif c["source"] == "analysis":
                facts.append(f"- 패턴 분석: {c['data'].strip()}")
        system += (" 런북을 사용한 조치에는 제공된 출처 URI를 표시하라. "
                   "조치 실행 도구가 없으므로 재시작·증설·설정 변경을 실행했다고 쓰지 마라. "
                   "알림 임계값과 런북 조치 임계값은 다른 기준이므로 구분하라. "
                   "현재 관측·알림·가설·미확인을 구분하고, 단일 수치로 지속을 단정하지 마라. "
                   "지속 조건이 있는 조치에는 현재 지속 여부의 확인/미확인을 먼저 명시하라. "
                   "제보와 현재 관측이 다르면 발생 시각과 해당 구간 확인을 안내하라. "
                   "각 문서의 조건과 임계값을 보존하고, 여러 문서의 서로 다른 절차를 임의로 섞지 마라.")
        user = f"## 사용자 요청\n{query}\n\n## 수집된 사실\n" + "\n".join(facts)
        if feedback:
            user += f"\n\n## 이전 초안에 대한 검증 반려 사유 (반드시 반영)\n{feedback}"
        content, usage = self._chat(system, user)
        usage["prompt"] = f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
        if on_token is not None:
            on_token(content)
        return content, usage
