# 런북 RAG 검색

기존 `replan_check_node()`는 모니터링 알림에서 처음 일치하는 키워드를 찾고
런북 한 개를 읽었다. 일치하는 키워드가 없으면 `db-down`을 선택했다.
현재 기본 실행 환경은 질문과 관측을 함께 검색하고, 적합한 근거가 없으면
특정 런북이나 조치를 임의로 선택하지 않는다.

## 처리 흐름과 선택 이유

1. **원문 확보:** 서버 시작 시 기존 MCP 탐색·리소스 읽기 API로
   `data/runbooks`에 있는 문서 ID의 원문을 읽는다.
   MCP 입력 검증·시간 제한·연결 수명 관리는 기존 구현을 사용한다.
2. **분할·임베딩:** 문서 제목과 번호 절차 두 개씩을 하나의 청크로 만든다.
   조건과 조치 문장을 중간에서 자르지 않는다.
   로컬 `embeddinggemma:300m`으로 청크 벡터를 생성한다.
3. **검색:** 질문·관측의 임베딩과 코사인 유사도를 계산한다.
   같은 문서의 여러 청크는 최고 점수로 합치고 문서 후보 3개를 선택한다.
   현재 6개 문서 규모에서는 별도 벡터 DB보다 메모리 색인이 간단하며,
   앱 수명 동안 색인을 재사용한다.
4. **근거 선택:** 로컬 `qwen3:8b`에 후보 원문을 제공하고
   요청 절차를 실제로 뒷받침하는지 먼저 판단하게 한다.
   관련 문서는 여러 개 선택할 수 있다. 후보 ID를 JSON 스키마로 제한하고
   반환된 ID·중복·판단 형식을 코드에서도 검사한다.
5. **답변 생성:** 런북이 짧으므로 채택한 문서는 원문 전체를 컨텍스트에 넣는다.
   일부 청크만 넣어 조건·복구 절차가 빠지는 문제를 줄인다.
   원문과 URI를 기존 작성 모델에 전달하고 출처를 표시하도록 지시한다.
6. **근거 부족:** 채택 문서가 없으면 원인을 확정하거나 재시작을 권하지 않고
   증상·오류 메시지·발생 시점과 해당 문서 확인을 요청한다.
   통신 오류·형식 오류는 정상적인 미조회와 구분해 예외로 전파한다.

선택 모델은 규칙 검사를 대체하지 않으며 의미 판단이 완벽하지 않다.
문서 순위와 채택 문서, 선택 이유를 별도로 남겨 오판을 확인할 수 있게 했다.
사용자 문장과 문서 내용은 모델이 따를 명령이 아니라 판단 대상 데이터로 전달한다.

## 수정 위치

| 파일 | 역할 |
|---|---|
| `agent/rag.py` | 데이터 구조, 절차 분할, 임베딩, 문서 검색, 근거 선택, MCP 색인 조립 |
| `agent/deps.py` | 주입할 검색기 필드 |
| `agent/runtime.py` | 앱 시작 시 검색기 조립, 실패 시 기존 자원 정리 |
| `agent/graph.py` | 실제 관측을 검색기에 전달, 근거·이벤트 연결, 미조회 응답 |
| `agent/fake_llm.py` | 복수 문서·출처를 표시하는 테스트용 작성 모델 |
| `agent/ollama_llm.py`, `agent/openai_llm.py` | 작성 프롬프트에 원문 출처와 조건 보존 지침 전달 |
| `web_chat.html` | 기존 진행 목록에 검색한 런북 또는 근거 부족 표시 |
| `eval/run_rag_evaluation.py` | 전후 동일 입력의 검색 측정과 원문 포함 결과 저장 |

검색기를 주입하지 않은 `SupervisorDeps`와 `create_runtime(retriever_factory=None)`는
기존 검색 방식의 비교·회귀 테스트 경로다.
일반 `server:app`은 RAG를 기본으로 사용한다. 모델 연결이 실패하면
예전 `db-down` 선택으로 조용히 돌아가지 않고 시작 또는 요청이 실패한다.

## 실행 조건

추가 Python 패키지는 없다. 기존 `httpx`로 로컬 Ollama API를 사용한다.
Ollama를 실행한 뒤 다음 모델이 필요하다.

```bash
ollama pull embeddinggemma:300m
ollama pull qwen3:8b
```

설정:

| 환경변수 | 기본값 | 용도 |
|---|---|---|
| `RAG_OLLAMA_URL` | `http://127.0.0.1:11434` | 임베딩·선택 모델 서버 |
| `RAG_EMBEDDING_MODEL` | `embeddinggemma:300m` | 문서와 질문 임베딩 |
| `RAG_SELECTION_MODEL` | `qwen3:8b` | 후보 문서 근거 판단 |

모델 변경은 검색 결과와 시간이 달라지는 실험 조건이다.
서로 다른 임베딩 모델로 색인·질의를 혼용하지 않는다.
문서나 모델을 바꾸면 서버를 다시 시작해 색인을 재구축한다.

```bash
# 검색은 실제 로컬 모델, 최종 문장 생성은 기존 fake 모델
LLM_BACKEND=fake LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8100

# 작성까지 실제 로컬 모델을 쓸 때는 설치된 모델명을 명시한다.
LLM_BACKEND=ollama OLLAMA_FAST_MODEL=qwen2.5:0.5b OLLAMA_QUALITY_MODEL=qwen3:8b \
  LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8100
```

fake 작성 모델의 기존 첫 초안·재작성 연출은 그대로이므로
그 답변을 실제 모델의 품질 평가로 제시하지 않는다.
로컬 검색에는 외부 API 청구 비용이 없지만 추론 시간과 장비 자원을 사용한다.
OpenAI 작성 백엔드를 별도로 선택하면 최종 답변 생성에는 기존 API 과금이 발생한다.

## 검증과 한계

```bash
PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

단위 테스트는 분할·순서·출처 보존, 잘못된 벡터, 후보 밖 ID,
명시적 미채택, 오류 전파를 검사한다. 앱 테스트는 RAG 결과의 HTTP/SSE 연결,
미조회 시 기존 문서로 대체하지 않는 동작과 시작 실패 시 자원 정리를 검사한다.
실제 로컬 모델 검색 평가는 `eval/RAG_EVALUATION.md`에 따로 설명한다.

이 변경은 런북 RAG에 한정된다. 최종 답변의 모든 주장에 대한 자동 사실 검증,
시스템별 근거 연결 판정, 로그 조회, 전문 에이전트, LangSmith 클라우드 비교 평가는
별도 작업이다. 출처를 포함하도록 프롬프트를 보강한 것과
답변의 모든 주장을 검증하는 것은 다르다.

API 참고: [Ollama 임베딩](https://docs.ollama.com/api/embed),
[Ollama 채팅](https://docs.ollama.com/api/chat),
[EmbeddingGemma 모델](https://ollama.com/library/embeddinggemma).
