"""Writer 추출 전후의 프롬프트와 공급자별 전송·콜백 계약을 고정한다."""
import asyncio
import copy
import json

import httpx
import pytest

from agent.openai_llm import OpenAIQualityLLM
from agent.ollama_llm import OllamaQualityLLM


EXPECTED_SYSTEM = (
    "당신은 ICT 운영 보고 작성자다. 수집된 사실만 사용해 한국어로 "
    "간결한 대응 보고를 작성하라. 반드시 '1) ... 2) ... 3) ...' 형식의 "
    "번호 절차로 답하고, 수치는 수집된 값을 그대로 인용하라."
    " 런북을 사용한 조치에는 제공된 출처 URI를 표시하라. "
    "조치 실행 도구가 없으므로 재시작·증설·설정 변경을 실행했다고 쓰지 마라. "
    "알림 임계값과 런북 조치 임계값은 다른 기준이므로 구분하라. "
    "현재 관측·알림·가설·미확인을 구분하고, 단일 수치로 지속을 단정하지 마라. "
    "지속 조건이 있는 조치에는 현재 지속 여부의 확인/미확인을 먼저 명시하라. "
    "제보와 현재 관측이 다르면 발생 시각과 해당 구간 확인을 안내하라. "
    "각 문서의 조건과 임계값을 보존하고, 여러 문서의 서로 다른 절차를 임의로 섞지 마라."
    " 알림이 보고한 추세·기간·위험 신호는 해당 시스템의 '모니터링 알림에 따르면'으로 출처를 명시하라. "
    "재작성 시에도 수정할 주장과 관련된 원문 알림의 시스템·지표·기간·위험 신호를 보존하라. "
    "피드백이 근거가 없다고 하더라도 원문에 해당 알림이 있으면, 삭제하거나 단순히 확인이 필요하다고 바꾸지 말고 알림 출처를 밝혀 서술하라. "
    "알림 보고와 직접 시계열을 검증한 사실은 구분하고, 다른 시스템·지표로 추세를 옮기거나 알림만으로 조치 조건 충족을 단정하지 마라."
)

CASES = [
    pytest.param([], None, "## 사용자 요청\n상태 확인\n\n## 수집된 사실\n", id="empty"),
    pytest.param([
        {"source": "monitoring", "data": {"system_name": "web-01", "status": "degraded",
         "cpu_percent": 91.0, "memory_percent": 89, "alerts": ["CPU high"],
         "last_checked": "2026-10-21T09:00:00"}},
        {"source": "runbook", "uri": "runbook://cpu", "data": " \n# CPU\n1. 확인\n "},
        {"source": "analysis", "data": "  공통 원인은 가설입니다.  "},
    ], "[temporal] 지속 여부를 확인하세요.",
        "## 사용자 요청\n상태 확인\n\n## 수집된 사실\n"
        "- 모니터링 web-01: 상태=degraded, CPU 91.0%, 메모리 89%, "
        "조회 시각=2026-10-21T09:00:00, 알림 1건 ['CPU high']\n"
        "- runbook 출처: runbook://cpu\n# CPU\n1. 확인\n"
        "- 패턴 분석: 공통 원인은 가설입니다.\n\n"
        "## 이전 초안에 대한 검증 반려 사유 (반드시 반영)\n[temporal] 지속 여부를 확인하세요.",
        id="evidence-and-feedback"),
    pytest.param([
        {"source": "runbook", "data": "  # 기존 문서  "},
        {"source": "ignored", "data": "포함하지 않을 근거"},
        {"source": "monitoring", "data": {"system_name": "cache-service", "status": "healthy",
         "cpu_percent": 20, "memory_percent": 30, "alerts": []}},
    ], "", "## 사용자 요청\n상태 확인\n\n## 수집된 사실\n"
        "- runbook 출처: (기존 문서)\n# 기존 문서\n"
        "- 모니터링 cache-service: 상태=healthy, CPU 20%, 메모리 30%, 조회 시각=미확인, 알림 0건 []",
        id="fallbacks-and-order"),
]


@pytest.mark.parametrize("collected,feedback,expected_user", CASES)
@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_writer_preserves_prompt_transport_usage_and_callback(monkeypatch, backend, collected, feedback, expected_user):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    requests, pieces = [], []
    original = copy.deepcopy(collected)

    def respond(request):
        requests.append(json.loads(request.content))
        if backend == "openai":
            return httpx.Response(200, request=request, json={
                "model": "writer-model", "choices": [{"message": {"content": "완성 답변"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            })
        chunks = [{"message": {"content": "완성 "}, "done": False},
                  {"message": {"content": "답변"}, "done": True,
                   "prompt_eval_count": 100, "eval_count": 20}]
        return httpx.Response(200, request=request,
                              content="\n".join(json.dumps(c, ensure_ascii=False) for c in chunks))

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original_client(
        transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(httpx, "post", lambda url, *, json, **kw: respond(
        httpx.Request("POST", url, json=json)))
    model = (OpenAIQualityLLM if backend == "openai" else OllamaQualityLLM)("writer-model", 0.005)
    answer, usage = asyncio.run(model.write("상태 확인", collected, 2, feedback, on_token=pieces.append))
    assert collected == original
    assert answer == "완성 답변"
    assert pieces == (["완성 답변"] if backend == "openai" else ["완성 ", "답변"])
    assert len(requests) == 1
    assert requests[0]["messages"] == [
        {"role": "system", "content": EXPECTED_SYSTEM}, {"role": "user", "content": expected_user}]
    assert requests[0]["model"] == "writer-model"
    assert requests[0].get("stream", False) is (backend == "ollama")
    assert usage == {"model": f"{backend}:writer-model", "tokens": 120, "usd": 0.0006,
                     "prompt": f"[SYSTEM]\n{EXPECTED_SYSTEM}\n\n[USER]\n{expected_user}"}


@pytest.mark.parametrize("collected,feedback,expected_user", CASES)
def test_common_writer_prompt_is_pure_and_preserves_content(collected, feedback, expected_user):
    from agent.writer import build_writer_prompt
    original = copy.deepcopy(collected)
    assert build_writer_prompt("상태 확인", collected, feedback) == (EXPECTED_SYSTEM, expected_user)
    assert collected == original
