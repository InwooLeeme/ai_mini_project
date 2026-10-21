"""시스템 이름 판별과 조사 대상 제한을 모델 백엔드와 독립적으로 적용한다."""
import json
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Literal


def load_known_systems() -> frozenset[str]:
    """실제 모니터링 목록을 사용한다. 읽기 실패를 임의의 기본 대상으로 숨기지 않는다."""
    path = Path(__file__).resolve().parents[1] / "data/monitoring.json"
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("모니터링 시스템 목록이 비어 있거나 잘못되었습니다")
    return frozenset(data)


KNOWN_SYSTEMS = load_known_systems()
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+")
_ALIAS_HINTS = {
    "결제": "payments-service", "인증": "auth-service", "캐시": "cache-service",
    "게이트웨이": "gateway-service", "로그": "log-service", "웹": "web-01",
}
_ALIAS = re.compile(r"(?:결제|인증|캐시|게이트웨이|로그|웹|그|해당)\s*(?:서비스|서버|시스템)")
_CHOICE_REPLY = re.compile(
    r"\s*(?:(?:네|예|응|맞아요|yes|ok)\s*[,，]?\s*)?"
    r"(?:\s|[,/&+]|와|과|랑|하고|및|그리고|and)*"
    r"(?:(?:으로|로|을|를)?\s*(?:선택|확인|진단)?\s*"
    r"(?:해\s*줘|해\s*주세요|할게요|입니다|이요|요)?)?[.!?]?\s*", re.IGNORECASE)


@dataclass(frozen=True)
class TargetResolution:
    """확정 대상과 확인이 필요한 이유를 구분한다. 추천은 조사 권한이 아니다."""
    targets: tuple[str, ...]
    reason: str | None = None
    suggestions: tuple[str, ...] = ()


def _diagnostic_identifier(token: str) -> bool:
    """명시적인 오류 코드·분위수 지표만 예외로 둔다. 미등록 서비스까지 무시하지 않는다."""
    return bool(
        re.fullmatch(r"(?:ERR|ERROR|HTTP)_[A-Z0-9]+(?:_[A-Z0-9]+)*", token)
        or re.fullmatch(r"p\d{1,2}(?:[._-](?:latency|duration|response|time|ms|seconds))+",
                        token, re.IGNORECASE)
    )


def _ambiguous_scope(query: str, known: tuple[str, ...]) -> bool:
    """시스템에 걸린 배제만 검사한다. 상태 설명의 'not confirmed' 등은 배제가 아니다."""
    for name in known:
        escaped = re.escape(name)
        if re.search(
            escaped + r"(?:은|는|을|를|이|가|의)?\s*(?:"
            r"말고|빼(?:\s*(?:주|줘)|고)|제외|아니라|아니고|아닌|아냐|아니야|대신|"
            r"(?:조사|확인|진단)(?:은|는|을|를)?\s*(?:하)?지\s*(?:말|마|않))", query
        ):
            return True
        if re.search(
            r"\b(?:except|excluding|exclude|instead\s+of|not|do\s+not|don't)"
            r"\s+(?:(?:investigate|diagnose|check|monitor)\s+)?" + escaped
            + r"(?![A-Za-z0-9_.-])", query, re.IGNORECASE
        ):
            return True
    return bool(re.search(
        r"(?<![가-힣A-Za-z0-9])(?:그|이전|다른|해당)\s*(?:서비스|시스템|서버)|나머지", query))


def resolve_targets(query: str) -> TargetResolution:
    """정확한 식별자만 인정하며 오타·배제 표현·불명확한 지칭은 사용자에게 돌린다."""
    identifiers = list(dict.fromkeys(_IDENTIFIER.findall(query)))
    known = tuple(name for name in identifiers if name in KNOWN_SYSTEMS)
    unknown = [name for name in identifiers
               if name not in KNOWN_SYSTEMS and not _diagnostic_identifier(name)]
    suggestions = tuple(dict.fromkeys(
        [name for token in unknown
         for name in get_close_matches(token, sorted(KNOWN_SYSTEMS), n=1, cutoff=0.65)]
        + [name for label, name in _ALIAS_HINTS.items()
           if re.search(label + r"\s*(?:서비스|서버)", query)
           and name in KNOWN_SYSTEMS and name not in known]
    ))
    if unknown:
        return TargetResolution((), "unknown_system", suggestions)
    if _ambiguous_scope(query, known):
        return TargetResolution((), "ambiguous_scope", known)
    if suggestions:
        return TargetResolution((), "alias_requires_confirmation", suggestions)
    return TargetResolution(known, None if known else "missing_system")


@dataclass(frozen=True)
class FollowupDecision:
    """확인 대기의 응답을 한 번만 판별해 선택/재질문/취소/새 질문 중 하나로 보낸다."""
    kind: Literal["select", "clarify", "cancel", "new_request"]
    targets: tuple[str, ...] = ()


def classify_followup(query: str) -> FollowupDecision:
    """공손한 선택 표현은 재개하고, 불확실한 선택으로 원래 질문을 삭제하지 않는다."""
    query = query.strip()
    if re.fullmatch(r"(?:취소(?:해\s*줘|해\s*주세요)?|cancel)[.!?]?", query, re.IGNORECASE):
        return FollowupDecision("cancel")
    resolved = resolve_targets(query)
    rest = _IDENTIFIER.sub("", query)
    if _CHOICE_REPLY.fullmatch(rest):
        return (FollowupDecision("select", resolved.targets) if resolved.targets
                else FollowupDecision("clarify"))
    # 새 증상·명시적인 새 요청은 이전 증상과 섞지 않는다.
    if re.search(r"CPU|메모리|디스크|장애|급증|지연|느려|느리|오류|원인|상태|"
                 r"새\s*(?:질문|요청)|새로|런북|절차|설명|"
                 r"\b(?:memory|disk|latency|error|new\s+question)\b", query, re.IGNORECASE):
        return FollowupDecision("new_request")
    if (_IDENTIFIER.search(query) or _ALIAS.search(query)
            or re.fullmatch(r"(?:네|응|예|맞아|맞아요|확인|yes|ok)[.!?]?", query, re.IGNORECASE)):
        return FollowupDecision("clarify")
    # 일반 대화는 키워드 목록에 가두지 않는다.
    return FollowupDecision("new_request")


def load_runbook_titles(retriever=None) -> tuple[str, ...]:
    """그래프 생성 시 색인의 원문을 재사용한다. 검색기 없는 비교 경로만 파일을 한 번 읽는다."""
    documents = getattr(retriever, "documents", None)
    texts = ([document.text for document in documents.values()] if documents is not None else
             [path.read_text() for path in
              (Path(__file__).resolve().parents[1] / "data/runbooks").glob("*.md")])
    titles = []
    for text in texts:
        lines = text.splitlines()
        if not lines or not lines[0].startswith("# Runbook: "):
            continue
        title = " ".join(lines[0].removeprefix("# Runbook: ").split())
        titles.append(title)
    return tuple(titles)


def is_runbook_request(query: str, titles: tuple[str, ...]) -> bool:
    """시작 때 확보한 제목으로만 검사하므로 요청 경로에서 파일을 읽지 않는다."""
    query = " ".join(query.split())
    for title in titles:
        if re.fullmatch(re.escape(title) + r"(?:를|을)?\s+(?:찾아줘|찾아주세요|보여줘|보여주세요)[.!?]?",
                        query, re.IGNORECASE):
            return True
    return False


def normalize_plan(raw: dict, targets: tuple[str, ...] | list[str]) -> dict:
    """모델이 대상을 추가·누락해도 확정된 대상마다 정확히 하나의 조사 작업을 만든다."""
    raw = raw if isinstance(raw, dict) else {}
    tasks = [
        {"id": f"task-{i}", "agent": "research", "kind": "monitoring",
         "target": target, "status": "pending"}
        for i, target in enumerate(dict.fromkeys(targets))
    ]
    proposed = raw.get("tasks", [])
    if isinstance(proposed, list) and any(
            isinstance(t, dict) and t.get("agent") == "analysis" for t in proposed):
        tasks.append({"id": "task-analysis", "agent": "analysis",
                      "kind": "pattern-analysis", "status": "pending"})
    return {"tasks": tasks, "needs_analysis": bool(raw.get("needs_analysis", False))}
