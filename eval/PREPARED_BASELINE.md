# 정리된 Before baseline 측정

현재 사용 경로는 `run_prepared_baseline.py`와 `baseline/prepared-manifest.json`이다.
기존 `eval/README.md` 및 `eval/before-source-manifest.json`은 원본 as_is 측정 자료로 보존했다.
원본 예제의 해시 검사를 우회하지 않는다.

## 1. 사전 검사 (모델·네트워크 호출 없음)

프로젝트 루트에서 다음 명령을 실행한다. 서버가 없어도 실행된다.

```bash
.venv/bin/python eval/run_prepared_baseline.py --preflight
```

옵션 없이 실행해도 동일한 오프라인 검사만 한다.
현재 파일·데이터셋·압축본·patch를 prepared manifest의 SHA-256과 대조하고,
강제 실패/인위적 지연 조건과 데이터셋의 요청 수를 검사한다.
불일치 시 exit code 1이며 서버를 호출하거나 측정 폴더를 만들지 않는다.

모의 수집 테스트 (실제 모델·HTTP 없음, 결과는 임시 폴더에서 제거):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_prepared_baseline_runner.py' -v
```

## 2. 이후 실제 측정을 시작할 때

**다음 단계는 실제 모델 호출 및 대화 DB 저장을 수행한다. 사전 검증과 별개다.**
패키지와 Ollama 모델이 이미 준비되어 있고, 8100 포트를 다른 서버가 사용하지 않는지 확인한다.
이 문서는 설치·다운로드나 기존 프로세스 종료를 지시하지 않는다.

터미널 A에서 이 프로젝트의 서버를 직접 실행한다.

```bash
LLM_BACKEND=ollama \
SUPERVISOR_VERSION=v2 \
OLLAMA_FAST_MODEL=qwen3:0.6b \
OLLAMA_QUALITY_MODEL=qwen3:8b \
LANGCHAIN_TRACING_V2=false \
LANGSMITH_TRACING=false \
.venv/bin/python server.py
```

모델을 바꾸면 fast/quality 모델명과 식별자를 기록하고 After에도 같은 모델을 사용한다.
이미 실행 중인 서버가 오래된 코드라면 새 파일 해시 검사만으로 그 서버가 갱신되었다고 판단할 수 없다.
본인이 이 baseline 경로에서 실행한 서버를 사용해야 한다.

터미널 B에서:

```bash
.venv/bin/python eval/run_prepared_baseline.py --run --repetitions 3
```

- 기본 동시성 1, 8개 시나리오 × 3회 = **27개 요청**. C08은 2턴이다.
- `--repetitions` 미지정 시 `EVAL_REPETITIONS` 환경변수 또는 기본 3을 사용한다.
- `--run`을 생략하면 측정하지 않는다.
- `/health`가 Ollama/v2가 아니면 질문을 보내지 않는다. OpenAI·Fake는 이 측정기에서 허용하지 않는다.
- 시나리오·반복마다 새로운 thread_id를 쓰며, C08 두 턴만 같은 세션이다.
- HTTP 오류·timeout·응답 형식 오류에서 원시 결과를 저장하고 중단한다. 클라이언트 재시도 없음.
- 중단한 요청이 서버에서는 계속 처리될 수 있다. 재실행 전 진행 중 요청 상태를 확인한다.
- 미실행 후속 턴은 request-plan과 결과 행 차이로 식별한다. 성공·비용 0으로 처리하지 않는다.

## 3. 실행 조건과 결과

| 항목 | prepared baseline |
|---|---|
| CPU·메모리·상태·알림 | 고정 JSON 데이터, jitter 없음 |
| 연구 태스크의 인위적 지연 | 0초 |
| 최초 MCP 강제 실패 | 기본 비활성화 |
| 실제 timeout 재시도 | 기존 최대 3회 시도·0.3/0.9초 backoff 유지 |
| last_checked | 실제 조회 시각, 고정하지 않음 |
| 모델 출력 | 완전한 결정론을 보장하지 않음 |
| 전체 요청 지연 | POST 직전~응답 전체 수신까지 |
| 비용 | `cost_usd`를 가상 비용으로 기록, 실제 청구액 아님 |

결과는 매번 고유한 `eval/results/before-prepared-<UTC시간>-<ID>/`에 생성한다.

- `manifest.json`: prepared profile/conditions, 원본 prepared manifest 전체와 해시,
  모델 이름, 요청 계획/수신/미실행 수, 완료·중단 상태.
- `request-plan.json`: 예정된 모든 질문·순서·세션.
- `raw.jsonl`: 질문, 응답 원문, 이벤트, metadata, 오류.
- `measurements.csv`: 지연(ms), 가상 비용, 도구 retry, 작성·재작성·재조사 횟수, 완료된 조회 대상.

`response_received`는 정확한 답변이라는 뜻이 아니다. 수동 판정표와 데이터셋의 기준으로 별도 채점한다.
실제 토큰·실제 청구 비용은 현재 API 응답에 없으므로 null/빈 값으로 유지한다.
`parallel_elapsed`는 마지막 조사 라운드의 초 단위 값이며 전체 지연과 별개다.
이름 기억 시나리오의 지연은 턴별로 기록한다. 사람의 입력 대기시간은 측정하지 않는다.

실행 후 결과 manifest의 미기록 환경 항목을 채운다:
하드웨어, 서버 패키지 버전, Ollama 버전·모델 식별자, cold/warm 상태, 추적 설정.
`source_manifest.packages`는 준비 당시의 기록이며 실행 서버 버전을 자동 검증했다는 뜻은 아니다.
원본 `.env`나 키를 저장하지 않는다. 다른 요청·서버 재시작·설정 변경이 끼면 해당 실행에 기록한다.
모델 적재 시간이 포함된 첫 요청과 나머지를 구분하고 After도 같은 워밍업 정책을 적용한다.

## 4. After와 비교할 때

**이 prepared baseline이 기능 개선의 Before다.** 원본 as_is와 혼동하지 않는다.
`eval/dataset.json`의 `baseline_commit`은 원본 계보를 나타내며,
실제로 측정한 수정 전 상태의 식별자는 결과의 `source_manifest_sha256`과 보존본이다.
질문·판정 기준·데이터·모델·동시성·세션·캐시/워밍업 조건을 유지한다.
검색·답변 정확도는 분리하고 오류·관측 불가·보류도 보고한다.
인위적 지연과 강제 실패를 제거한 효과는 기능 개선 점수에 합산하지 않는다.

현재 실행기는 **Before 전용**이다. After 코드로 바뀌면 해시 검사 실패가 정상이다.
After 측정은 별도 manifest/phase/결과 폴더를 사용하는 실행 경로에서 수행해야 하며
Before manifest를 After 해시로 덮어쓰면 안 된다.
보존한 Before를 다시 실행할 때는 압축본을 별도 폴더에 복원하고,
원래 baseline 디렉터리의 prepared manifest·patch·압축본을 함께 보관/배치한다.
압축본은 자기 자신과 최종 manifest/patch를 포함하지 않는다.

측정 결과는 baseline 코드 보존본에 포함하지 않는다. 결과 파일이 생겼다는 이유로 baseline 해시를 갱신하지 않는다.
