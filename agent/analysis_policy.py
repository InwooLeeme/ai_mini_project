"""별도 모델 호출 없이 명시적인 시스템 비교 요청을 판별한다."""
import re


_COMPARISON = re.compile(
    r"비교(?!적)|공통\s*(?:의\s*)?원인|연관성|연관\s*관계|상관\s*(?:관계|성)"
    r"|\bcompar(?:e|ison)\b|\bcorrelat\w*\b|\b(?:common|shared)\s+(?:root\s+)?cause\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"하지\s*(?:마|말)|필요(?:가)?\s*없|생략|제외"
    r"|\b(?:don't|do not|skip|without)\b",
    re.IGNORECASE,
)
_MULTIPLE_SYSTEMS = re.compile(
    r"(?:두|세|여러|복수|모든|각)\s*(?:개\s*)?(?:시스템|서비스|서버)"
    r"|(?:시스템|서비스|서버)\s*(?:들|간|사이)"
    r"|\b(?:both|multiple|two|these)\s+(?:systems|services|servers)\b",
    re.IGNORECASE,
)


def comparison_requested(query: str, system_names: set[str]) -> bool:
    """일반 원인 확인·복수 조회를 제외하고, 부정되지 않은 비교 표현만 허용한다.

    정해진 한국어·영어 표현을 지원하는 보수적인 규칙이다. 계획 모델의 자유로운
    판단이나 시스템 개수만으로 분석 호출을 추가하지 않는다.
    """
    # 한 시스템 내부의 CPU·메모리 비교나 과거 대비 요청은 별도 시스템 비교가 아니다.
    named = {name for name in system_names if name in query}
    if len(named) < 2 and not _MULTIPLE_SYSTEMS.search(query):
        return False
    clauses = re.split(r"[.!?;\n]|하지만|대신|(?<=말고)", query)
    requested = False
    for clause in clauses:
        if _COMPARISON.search(clause):
            # 같은 요청 안에서 뒤에 나온 명시적인 정정·취소를 우선한다.
            requested = not bool(_NEGATION.search(clause))
    return requested


ANALYSIS_INSTRUCTIONS = (
    "당신은 ICT 운영 패턴 분석가다. 제공된 여러 시스템의 현재 모니터링 값을 비교하고 "
    "공통 원인으로 의심할 수 있는 가설과 이를 확인할 추가 근거를 짧게 설명하라. "
    "단일 시점의 값만으로 시간상 동시 발생, 통계적 상관관계나 공통 원인을 확정하지 마라. "
    "로그·시계열·의존성 정보는 제공되지 않았으므로 조회했다고 말하지 마라. "
    "공통 패턴의 근거가 부족하면 부족하다고 명시하라."
)


def analysis_facts(collected: list[dict]) -> str:
    """분석 모델에 넘길 현재 모니터링 관측 요약."""
    mon = [c["data"] for c in collected if c["source"] == "monitoring"]
    return "\n".join(
        f"- {d['system_name']}: 상태={d['status']}, CPU {d['cpu_percent']}%, "
        f"알림 {d['alerts']}" for d in mon) or "(모니터링 데이터 없음)"
