# 수정 전후 평가 데이터셋과 Before 측정 안내

## 현재 준비 상태

- `dataset.json`: 사용자가 선정한 **8개 시나리오, 9개 요청**, 기대 행동과 판정 기준.
- `before-source-manifest.json`: 준비 시점의 HEAD, 기존 추적 파일 해시, 데이터·런북 해시. `.env`와 API 키는 포함하지 않는다.
- `manual-review-template.csv`: 1회 실행의 9개 요청에 대한 수동 판정 양식. 판정은 모두 `not_run`.
- 모델 호출, 서버 실행, 실제 성능 측정, LangSmith 전송은 아직 하지 않았다.
- 기존 애플리케이션 코드는 변경하지 않았다. 이미 있던 `docs/` 문서도 수정하지 않았다.
- 이 문서의 명령은 **사용자가 이후 실행할 때** 서버·모델 호출·SQLite 메시지 저장·측정 파일 생성을 수행한다.

## 1. 먼저 Before의 의미를 고정한다

Before는 현재 v2 코드의 커밋 `3fb307358af4d9451c3c3c2040b41f6a042cc1ef`다.
`SUPERVISOR_VERSION=v1`이 아니다. 새 데이터셋 파일을 추가하는 것은 진단 로직 변경이 아니다.

측정은 두 종류를 구분한다.

| 실험 | 실행 조건 | 사용 목적 |
|---|---|---|
| `as_is` | 원본 v2 그대로, 난수·0.35초 sleep·프로세스 최초 강제 timeout 유지 | 지금의 사용자 경험·동작 실패·지연 분포를 보존 |
| `controlled` | 동일 고정 Tool 응답·시각, 양쪽의 동일 지연·실패·캐시 조건 | 개선 효과의 공정한 최종 비교 |

아래 실행 예시는 **as_is**다. 입력 질문과 원본 데이터 파일은 같아도 Tool 응답에 난수가 있으므로, 동일한 관측값을 넣은 통제 실험이라고 부르지 않는다.

최종 발표의 통제 비교에는 별도 평가 하네스에서 다음을 양쪽에 동일하게 적용한다.

1. 사례/시스템/호출 회차별 Tool 응답과 시각을 고정해 재생한다.
2. 정상 실험은 양쪽 모두 강제 timeout과 데모 sleep을 끈다. 복구 실험은 같은 작업·회차에 동일 실패를 주입한다.
3. 원본 Before 코드는 보존하고 평가 환경에서만 대역·계측을 주입한다. 기준 코드 수정이 필요해지면 차이를 따로 기록한다.
4. 합성 지연·실패를 없앤 이득을 RAG·에이전트 개선 효과로 합산하지 않는다.
5. 이 통제 하네스는 이번 데이터셋 준비에서 구현하거나 실행하지 않았다. 현재 코드에는 이 모든 조건을 끄는 환경변수가 없다.

개선 전에 Before 실행을 보존하고, 최종 통제 비교 시에는 보존한 Before와 After를 같은 조건에서 다시 실행한다.

## 2. 정답을 현재 코드의 출력과 분리한다

| 사례 | 목표 행동 | 필수 런북 | 주의점 |
|---|---|---|---|
| C01 | payments 조사 | cpu-spike | 알림의 91%와 난수가 섞인 현재 CPU 수치는 다르다 |
| C02 | gateway와 payments 조사 | network-timeout, cpu-spike | 두 시스템의 근거를 구분한다 |
| C03 | web-01과 auth 조사, 제보/관측 불일치 설명 | auth: high-latency | 현재 web-01은 healthy. cpu-spike는 추가 조사용으로만 허용 |
| C04 | payments와 cache 조사, 공통 원인은 가설로 설명 | cpu-spike, memory-leak | 두 시스템 저하만으로 공통 원인을 확정하지 않는다 |
| C05 | 대상 확인 질문 | 해당 없음 | 로그인이라는 이유만으로 auth를 조회하면 실패 |
| C06 | 미등록 안내와 대상 확인 | 해당 없음 | billing을 다른 시스템으로 바꾸거나 unknown의 0.0을 사실로 쓰면 실패 |
| C07 | 인사·자기소개 | 해당 없음 | 단순 경로 정확도와 답변 내용 충족은 별도 |
| C08 | 같은 세션에서 이름 수용 후 민수 기억 | 해당 없음 | 2턴, 2요청. 재시작 내구성 테스트는 포함하지 않는다 |

- C05/C06은 **첫 요청의 적절한 확인 행동**만 비교한다. 개선 후 확인 질문이 나오는 것은 진단 실패가 아니다.
- 확인 뒤 진단 재개 성능은 후속 사용자 답변까지 고정한 별도 데이터셋으로 평가해야 한다.
- Before와 After의 기준은 같다. 구현 결과를 보고 기준을 바꾸려면 데이터셋 버전을 올리고 양쪽을 다시 평가한다.
- 새 라우터가 `clarification` 같은 경로명을 쓰면 개념상 장애 요청 처리로 매핑한다. 문자열 이름 차이만으로 라우팅 실패를 만들지 않는다.
- 데이터셋은 공개된 소규모 회귀셋이다. 튜닝에 사용한 뒤 일반화 성능을 주장하지 말고, 필요하면 별도 보류 평가셋을 추가한다.

## 3. 측정할 수 있는 항목과 없는 항목

| 항목 | 수집 방법 | 정확한 의미 |
|---|---|---|
| 전체 요청 지연 | 클라이언트 `perf_counter()`로 POST 시작~응답 전체 수신 | 라우팅·MCP·생성·검증·DB 저장·로컬 HTTP 포함 |
| 조사 지연 | `metadata.parallel_elapsed` | 마지막 research 라운드의 시간. 전체 요청 시간이나 모든 조사 시간의 합이 아니다 |
| 가상 비용 | `metadata.cost_usd` | 사용량 × 코드에 지정된 가상 단가. 실제 청구액이 아니다 |
| 도구 retry | `metadata.retry_count` | 성공한 조사에서 집계된 재시도. 실패로 응답이 없으면 null |
| 작성 횟수 | `metadata.validation_attempts` | 답변 작성/검증 시도. 재작성 횟수는 `max(0, attempts - 1)` |
| 재조사 횟수 | `metadata.replan_rounds` | 도구 retry나 답변 재작성과 별개 |
| 조회 시스템·런북 | `TaskCompleted` 이벤트 | 완료된 조회 목록. 실패한 호출 전체 목록은 아니다 |
| LangSmith 연결 | `metadata.langsmith_run_id` | ID 존재만으로 원격 전송 성공을 입증하지 않는다 |
| 모델 실제 토큰 | 현재 응답에는 미노출 | 별도 추적의 LLM usage 또는 향후 계측 필요. 가상 비용을 역산하지 않는다 |
| 수치 정확도 | 동일 실행의 원본 Tool 응답과 답변 대조 | 현재 `/chat`에는 원본 수치 미노출. 없으면 `unobservable` |
| 시스템별 출처 연결 | 시스템+문서 연결 정보 | 현재 런북 이벤트는 incident type만 제공. 문서 집합과 별도로 판정 |
| 첫 토큰 시간·화면 렌더링 | SSE/브라우저 별도 계측 | 아래 POST 측정으로는 알 수 없다 |

Fake의 토큰은 문자열 길이 추정, Ollama/OpenAI 토큰은 모델 응답 계수지만 최종 metadata에는 그 토큰 합계가 없다.
Ollama의 외부 API 청구 없음은 전기·장비 비용까지 0이라는 뜻이 아니다.
실제 청구 비용은 확인하지 못하면 null로 남긴다.

## 4. 실행 환경을 준비한다

아래는 **이미 필요한 가상환경·패키지·로컬 모델이 준비되어 있을 때**의 실행 방법이다.
이 안내를 따라 설치·다운로드가 자동으로 실행되지는 않는다.

1. 다른 터미널의 기존 서버 상태를 확인한다. 기존 프로세스를 임의 종료하거나 포트를 겹쳐 띄우지 않는다.
2. Before용 터미널에서 프로젝트 루트로 이동한다.
3. 로컬 Ollama가 이미 실행 중이며 아래 두 모델이 준비되어 있어야 한다. 실제 설치된 모델을 바꾸면 모델명과 식별자를 기록하고 After도 동일하게 맞춘다.
4. 다음처럼 실행한다. 외부 추적은 우선 끄고 로컬 측정부터 한다. 기존 `.env`를 수정하지 않는다.

```bash
cd /Users/iin-u/Desktop/workspace/AI/mini-project/all_total_vue_student
LLM_BACKEND=ollama \
SUPERVISOR_VERSION=v2 \
OLLAMA_FAST_MODEL=qwen3:0.6b \
OLLAMA_QUALITY_MODEL=qwen3:8b \
LANGCHAIN_TRACING_V2=false \
LANGSMITH_TRACING=false \
.venv/bin/python server.py
```

이 명령은 서버 실행 예시이며 이번 준비 과정에서는 실행하지 않았다.
기존 `sh/run_agent_ollma.sh`에는 모델 준비·워밍업 등 동작이 있으므로 정확한 초기 상태 기록에는 위의 직접 실행을 권한다.
OpenAI로 바꾸어 실행하면 비용이 발생할 수 있다. 아래 측정 예시는 OpenAI 백엔드를 거부하도록 되어 있다.

다른 터미널에서 모델 호출 없는 상태 확인:

```bash
curl --fail --silent http://127.0.0.1:8100/health
```

기록할 항목: OS/하드웨어, Python·설치 패키지 버전, Ollama 버전과 모델 식별자, fast/quality 모델, 온도(현재 0), 데이터 해시, 추적 활성 여부, 동시성 1, 모델 적재 상태.
`requirements.txt`는 범위 지정이므로 파일이 같다는 것만으로 실제 설치 버전까지 같다고 간주하지 않는다.
환경변수 전체나 `.env`를 복사해 기록하지 않는다.

## 5. 8개 시나리오를 순차 실행한다

아래 코드를 **프로젝트 루트의 새 터미널**에 그대로 실행할 수 있다. Python 표준 라이브러리만 사용한다.
새 애플리케이션 기능을 추가하지 않고 기존 `/chat`에 요청한다. 기본 3회 반복 = 27개 요청이다.

- 첫 실행을 포함해 고정 순서로 3회 실행한다. 프로세스 최초 강제 timeout은 보통 첫 장애 요청에만 발생하므로, 첫 요청을 반드시 표시해 별도 해석한다.
- 이후 반복을 새 프로세스에서 한 것처럼 해석하지 않는다. 모델 cold/warm 상태도 실행 기록에 적는다.
- 모든 요청을 한 번씩 시험하려면 실행 전 `export EVAL_REPETITIONS=1`.
- 시나리오/반복마다 새 UUID 세션을 사용한다. C08 두 턴만 같은 세션이다.
- HTTP 오류·timeout이면 결과를 저장하고 전체 실행을 중단한다. 클라이언트 재시도를 하지 않는다. 서버에서 요청이 계속 실행 중일 수 있으므로 상태를 확인하기 전에 재실행하지 않는다.
- `request-plan.json`과 실제 결과를 비교하면 중단 때문에 실행하지 못한 요청을 구분할 수 있다.
- C08 turn 1 실패 시 turn 2도 실행하지 않는다.

```bash
python3 - <<'PY'
import csv
import hashlib
import json
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import request, error

root = Path.cwd()
dataset_path = root / "eval/dataset.json"
dataset = json.loads(dataset_path.read_text())
baseline = json.loads((root / "eval/before-source-manifest.json").read_text())
for name, expected in baseline["files_sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Before 기준 파일이 달라졌습니다: {name}")

base_url = "http://127.0.0.1:8100"
with request.urlopen(base_url + "/health", timeout=10) as response:
    health = json.load(response)
if health.get("llm_backend") != "ollama" or health.get("supervisor_version") != "v2":
    raise SystemExit("이 예시는 ollama/v2 Before 전용입니다. 유료·fake 호출을 시작하지 않습니다.")

repetitions = int(os.environ.get("EVAL_REPETITIONS", "3"))
if not 1 <= repetitions <= 10:
    raise SystemExit("반복 횟수는 1~10으로 설정하세요.")
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
out = root / "eval/results" / ("before-as-is-" + run_id)
out.mkdir(parents=True, exist_ok=False)
plan = []
for repetition in range(1, repetitions + 1):
    for case in dataset["cases"]:
        thread_id = f"eval-{run_id}-{repetition}-{case['id']}"
        for turn in case["turns"]:
            plan.append({
                "sequence": len(plan) + 1, "repetition": repetition,
                "case_id": case["id"], "turn": turn["turn"],
                "thread_id": thread_id, "message": turn["message"],
            })
manifest = {
    "phase": "before", "profile": "as_is", "run_id": run_id,
    "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    "source_manifest": baseline, "health": health,
    "client_python": platform.python_version(), "client_platform": platform.platform(),
    "repetitions": repetitions, "planned_requests": len(plan),
    "concurrency": 1, "client_retry": 0,
    "randomness_controlled": False, "artificial_delay_seconds_per_task": 0.35,
    "forced_timeout_scope": "first monitoring call per MCPClient lifetime",
    "model_load_state": "not_recorded", "hardware": "not_recorded",
    "server_package_versions": "not_recorded", "ollama_version_and_model_ids": "not_recorded",
    "tracing_enabled": "not_verified", "completed_plan": False,
}
(out / "request-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
manifest_path = out / "manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
columns = [
    "sequence", "repetition", "case_id", "turn", "thread_id", "status",
    "http_status", "elapsed_ms", "route", "simulated_cost_usd", "retry_count",
    "validation_attempts", "rewrite_count", "replan_rounds",
    "last_research_elapsed_seconds", "monitoring_targets", "runbook_types",
    "langsmith_run_id", "actual_input_tokens", "actual_output_tokens", "actual_billed_usd",
]
completed = 0
failure = None
try:
    with (out / "raw.jsonl").open("w") as raw_file, (out / "measurements.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for item in plan:
            payload = {"message": item["message"], "thread_id": item["thread_id"]}
            req = request.Request(base_url + "/chat",
                                  data=json.dumps(payload).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            body, http_status, failure, result = None, None, None, {}
            try:
                with request.urlopen(req, timeout=900) as response:
                    http_status = response.status
                    body = response.read().decode("utf-8")
                elapsed = (time.perf_counter() - started) * 1000
                result = json.loads(body)
                if not isinstance(result, dict) or not isinstance(result.get("reply"), str) or not isinstance(result.get("metadata"), dict) or not isinstance(result.get("events"), list):
                    raise ValueError("응답 계약 불일치")
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000
                failure = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, error.HTTPError):
                    http_status = exc.code
                    body = exc.read().decode("utf-8", errors="replace")
                result = {}
            meta = result.get("metadata", {})
            events = result.get("events", [])
            attempts = meta.get("validation_attempts")
            def completed_targets(source):
                return sorted({e["target"] for e in events
                               if e.get("type") == "TaskCompleted" and e.get("source") == source and e.get("target")})
            row = {key: item[key] for key in ("sequence", "repetition", "case_id", "turn", "thread_id")}
            row.update(
                status="error" if failure else "response_received", http_status=http_status,
                elapsed_ms=round(elapsed, 3), route=meta.get("route"),
                simulated_cost_usd=meta.get("cost_usd"), retry_count=meta.get("retry_count"),
                validation_attempts=attempts,
                rewrite_count=max(0, attempts - 1) if isinstance(attempts, int) else None,
                replan_rounds=meta.get("replan_rounds"),
                last_research_elapsed_seconds=meta.get("parallel_elapsed"),
                monitoring_targets=json.dumps(completed_targets("monitoring")),
                runbook_types=json.dumps(completed_targets("runbook")),
                langsmith_run_id=meta.get("langsmith_run_id"),
                actual_input_tokens=None, actual_output_tokens=None, actual_billed_usd=None,
            )
            raw_file.write(json.dumps({**item, "measurement": row, "response": result,
                                       "raw_body": body, "error": failure}, ensure_ascii=False) + "\n")
            raw_file.flush()
            writer.writerow(row)
            csv_file.flush()
            completed += 1
            print(item["case_id"], item["turn"], row["status"], row["elapsed_ms"], "ms")
            if failure:
                raise SystemExit(f"측정 중단: {failure}\n저장 위치: {out}")
finally:
    manifest["recorded_requests"] = completed
    manifest["completed_plan"] = completed == len(plan) and failure is None
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print("저장 위치:", out)
PY
```

`response_received`는 HTTP 응답을 받았다는 뜻이고 정확한 답변이라는 뜻이 아니다.
CSV 빈 값과 JSON null은 미측정/관측 불가이며 0이 아니다.
실행 직후 manifest의 `not_recorded` 항목에 실제 환경을 기록하고, `tracing_enabled`도 확인한 설정으로 채운다.
이 예시는 주어진 서버가 실제로 어느 경로에서 시작됐는지 원격 검증하지 못하므로 반드시 위 프로젝트에서 시작한 서버인지 확인한다.
서버 재시작·다른 요청·동시 작업이 끼면 실험 조건에 기록하고 해당 실행을 무조건 정상 비교군으로 합치지 않는다.

## 6. 정확도를 판정한다

`manual-review-template.csv`를 실행별로 복사한다. 반복 실행은 각 반복의 9행으로 확장하고
`run_id`, `repetition`, `evidence_reference`, `reviewer_notes`를 채운다.

판정값:
- `pass`: 해당 기준을 근거와 함께 충족.
- `fail`: 기준 위반이 관측됨.
- `unobservable`: 기준은 적용되지만 원본 근거가 없어 판단 불가.
- `not_applicable`: 이 사례에 적용되지 않음.
- `not_run`: 실행하지 않음. 선행 턴 오류로 실행하지 않은 경우 사유에 blocked를 기록.

### 라우팅·대상·답변

1. 라우팅: 각 턴의 expected_route와 개념적으로 일치하는지 확인.
2. 대상: `TaskCompleted(source=monitoring)`의 target 집합을 required/allowed와 대조.
   필요한 대상 누락과 허용되지 않은 추가 조회를 각각 기록한다.
   오류 실행에서 완료 이벤트가 없다고 “조회하지 않음”으로 합격시키지 않는다.
3. 답변: criteria 전부 충족하고 forbidden이 없으면 답변 행동 합격.
   확인 질문은 C05/C06의 올바른 행동이다. 키워드 포함만으로 합격시키지 않는다.
4. 수치: 원본 Tool 응답이 없으면 사실 검증은 `unobservable`.
   monitoring.json의 cpu_base는 실제 호출값이 아니며 답변에 적힌 값도 검증의 근거가 될 수 없다.
   정적 alerts 문구는 파일과 비교할 수 있지만 그것만으로 모든 수치 정확도가 검증된 것은 아니다.
5. C08은 두 턴이 모두 적절하고 두 번째 답변이 민수이면 시나리오 합격.
6. HTTP 오류는 요청 성공률에서 실패로 포함한다. 미실행은 planned/requested/received 숫자를 분리해 보고한다.

### 검색: 현재 측정 가능한 문서 집합과 시스템 연결을 구분

Before에서는 `TaskCompleted(source=runbook)` 이벤트의 `target`을 검색한 incident type으로 본다.
이는 규칙 기반 런북 선택의 결과이며 벡터 검색 점수가 아니다.

- 필수 집합 G: `expected_retrieval_document_union`
- 허용 집합 A: G + `allowed_extra_runbooks_by_system`의 문서 합집합
- 실제 집합 R: 성공 완료된 runbook type의 중복 제거 집합
- 필수 근거 회수율: `|R ∩ G| / |G|`
- 반환 문서 관련 비율: `|R ∩ A| / |R|` (R이 비면 정의 불가, 별도로 빈 검색 실패 표시)
- 완전 회수: `G ⊆ R`
- 시스템별 연결: 각 문서가 어느 시스템에 사용되었는지 따로 판정. Before 이벤트로 알 수 없으면 `unobservable`.

예를 들어 C04에서 cpu-spike 하나만 선택했다면 문서 집합 회수율은 1/2지만,
그 문서가 실제로 payments에 연결되었는지는 별도 근거가 필요하다. **이 수치는 설명용 계산 예이며 실행 결과가 아니다.**
C05~C08은 검색 분모에서 제외하고 불필요한 조회 여부를 따로 평가한다.
C01~C04에서 잘못 chat으로 라우팅되어 검색을 하지 않았다면 전체 요청 기준 회수율은 0이다.
서버 오류로 검색 관측이 없으면 `unobservable`로 기록하고 오류율과 관측 가능 비율을 함께 제시한다.

Before에 순위·시스템별 후보가 없으므로 이 결과를 Recall@3로 부르지 않는다.
향후 top-k 검색기에는 같은 원문·정답셋을 사용하되, 검색기 단독의 Recall@k와
전체 요청의 문서 집합 회수율을 별도 보고한다. 검색 정확도와 최종 답변 정확도를 합쳐 하나로 보고하지 않는다.

원본 수치 검증이 필요하다면 허용된 후속 작업에서 원본 수집 계측이나 LangSmith 추적을 추가한다.
LangSmith를 쓰는 경우 요청/근거가 외부로 전송되므로 그 실행 범위를 먼저 확정한다.
현재의 `@traceable`만으로 실제 토큰·Tool 응답·원격 저장이 모두 완전하다고 가정하지 않는다.

## 7. 집계하고 After에 재사용한다

- 사례별 세 번의 원시값·중앙값·최솟값·최댓값을 함께 제시한다. 첫 요청은 cold-start/강제 timeout 영향을 따로 설명한다.
- 전체 지연은 chat, diagnose, clarify 기대 그룹을 나눠 보고한다. 서로 다른 작업을 평균 하나로만 비교하지 않는다.
- 3개 표본의 p95는 거의 최댓값이므로 안정적인 지연 꼬리 추정으로 해석하지 않는다.
- 요청 지연과 C08의 두 요청 처리시간 합은 별도. 사람 입력 대기시간은 포함하지 않는다.
- 가상 비용 합계에는 관측된 값만 더하고 누락 요청 수를 함께 적는다. 실제 청구 비용으로 제목을 붙이지 않는다.
- 정확도는 `pass / (pass + fail)`와 전체 대상 수·unobservable 수를 같이 보고한다.
  더 보수적인 전체 대상 대비 검증된 성공률도 함께 제시하여 미관측 항목을 숨기지 않는다.
- 3회 반복은 독립 질문 24개가 아니다. 독립 시나리오는 여전히 8개다.
- After에서도 dataset.json의 버전·해시·정답을 유지한다. 추가 기능용 데이터셋은 별도 버전/묶음으로 만든다.
- After는 별도 실행기에서 phase·소스 해시를 기록해야 한다. 위 Before 전용 예제의 해시 검사를 단순히 우회해서 같은 폴더에 덮어쓰지 않는다.

발표 표:

| 지표 | Before | After | 분모/조건 |
|---|---|---|---|
| 요청 응답 성공률 | 미측정 | 미측정 | 계획/실행/응답/오류 수 |
| 대상 정확도 | 미측정 | 미측정 | 8시나리오·9요청, 반복 수 |
| 필수 런북 집합 회수율 | 미측정 | 미측정 | C01~C04 |
| 시스템별 근거 연결 | 미확인 | 미확인 | 관측 가능 여부 포함 |
| 최종 답변 행동 정확도 | 미측정 | 미측정 | 기준별 판정 근거 |
| 수치 정확도 | 관측 자료 필요 | 관측 자료 필요 | 원본 Tool 응답 |
| 지연 중앙값·범위 | 미측정 | 미측정 | 같은 모델·실험 프로필 |
| 도구 retry / 재작성 / 재조사 | 미측정 | 미측정 | 각각 별도 |
| 가상 비용 | 미측정 | 미측정 | 실제 비용과 분리 |

현재 8개 사례에는 검색 결과 없음, 최대 검증 횟수 도달, MCP 영구 오류, 로그 조회 실패가 없다.
이 데이터셋만으로 새 기능 전체의 완료를 주장하지 않는다. 해당 회귀 테스트는 개선 단계에서 별도로 추가한다.
