# 조사 자료 수집 모듈화와 의존성 주입

기존 `research_node()`는 그래프 상태 갱신과 MCP 통신 정책을 함께 처리했다.
현재는 조사 수집 Module과 그래프 노드가 각각의 책임을 맡는다.

| 위치 | 책임 |
| --- | --- |
| `agent/evidence.py` | 수집 Interface, 작업·결과 데이터 구조, 실제 MCP 수집 구현 |
| `agent/deps.py` | 사용할 수집기를 `SupervisorDeps.evidence_collector`로 명시적으로 전달 |
| `agent/graph.py`의 `research_node()` | 대기 작업 선택, 수집 요청, 성공 결과를 상태와 이벤트에 반영 |
| `agent/runtime.py` | 모델·MCP·저장소·수집기·그래프 조립과 자원 정리 |
| `server.py`의 `create_app()` | 실행 환경 생성 함수를 주입받아 독립적인 FastAPI 앱 구성 |

## Interface를 작게 둔 이유

조사 노드는 `await collector.collect(tasks)`만 호출한다.
자료 수집의 MCP 서버 탐색, 런북 URI 구성, 병렬 실행과 재시도는
`MCPEvidenceCollector`의 Implementation 안에 둔다.
통신 정책 변경이 그래프 상태 처리로 번지지 않고, 상태 처리 테스트에
MCP 프로세스가 필요하지 않게 하는 선택이다.

실제 MCP 수집기와 테스트의 가짜 수집기가 같은 Interface를 사용하는 Adapter다.
`Protocol`은 호출 규약을 표현하며 런타임 입력 검증을 대신하지 않는다.
MCP 입력·응답 검증은 기존 `MCPClient`가 계속 담당한다.

`ResearchTask.index`는 원래 전체 작업 목록의 위치를 유지한다.
분석 작업이나 이미 완료한 작업을 제외한 뒤에도 올바른 작업을 완료 처리하기 위해서다.
결과는 입력 순서로 반환하며, 전체 수집에 성공한 뒤에만 복사한 상태를 갱신한다.
`frozen` 데이터 클래스는 필드 재할당을 막지만 내부 자료까지 깊은 불변으로 만들지는 않는다.

## 실행 환경을 별도로 조립하는 이유

`create_runtime()`은 생성 함수를 받아 실제 MCP와 모델, 저장소를 구성한다.
MCP를 만든 뒤 모델·저장소·그래프 조립이 실패하거나 실행 중 예외가 발생해도
종료를 시도하며, MCP 종료가 실패해도 저장소 정리를 시도한다.
AnyIO 취소 범위를 지키기 위해 MCP 시작과 종료는 같은 태스크에서 수행한다.

`create_app(runtime_factory=...)`에 테스트용 실행 환경을 전달하면
전역 앱을 덮어쓰거나 실제 MCP를 실행하지 않고 HTTP 처리를 검사할 수 있다.
각 요청은 전역 `app` 대신 `request.app`의 의존성을 사용하므로 앱별 이력이 분리된다.
생성 횟수는 각 앱의 lifespan 진입 횟수이며, 요청마다 증가하지 않는다.
기본 실행 방식인 `uvicorn server:app`은 유지한다.

## 유지한 동작

- 모니터링만 기존 시간 초과 재시도 적용, 런북 호출은 재시도하지 않음.
- 런북이 필요한 수집 묶음에서만 제공 서버를 한 번 탐색.
- 탐색 후부터 수집 시간을 측정하고 소수 둘째 자리로 반올림.
- 수집 결과·이벤트 순서, 기존 자료·재시도 횟수 누적.
- v1/v2 선택, 모델 선택·단가, 일반 응답과 SSE 이벤트 형식.
- `chat_request` 추적 이름과 입력 필드, 노드의 추적 연결.

RAG 검색, 런북 미조회 처리, 답변 검증 기준, 전문 에이전트와 UI는 이번 변경에
포함하지 않았다. 기존 v1 SSE 경로의 이력 미저장 동작도 별도 기능 수정 없이 유지한다.

## 검증

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  .venv/bin/python -m pytest tests -v -p no:cacheprovider
```

- `tests/test_evidence.py`: 가짜 수집기 주입, 상태 보존, 병렬성·순서,
  탐색 횟수와 시간 범위, 오류 전파·재시도 정책.
- `tests/test_app_runtime.py`: 앱 간 격리, 일반 응답·SSE·이력,
  조립 및 종료 실패 시 정리, 같은 태스크에서 시작·종료,
  로컬 추적 트리, 실제 MCP를 연결한 앱 통합 실행.
- 기존 MCP 안전성 및 baseline 관련 테스트도 유지한다.

LangSmith 추적 테스트는 실제 추적 트리를 만들되 SDK 전송 지점은 대체한다.
클라우드 평가 제출이나 유료 LLM 실행, 성능 재측정은 하지 않는다.
