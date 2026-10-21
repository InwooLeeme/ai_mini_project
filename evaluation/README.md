# 자동화 테스트와 LangSmith 비교 평가

기존 8개 시나리오·9개 요청을 변경하지 않고 현재 구현을 Before와 비교한다.
Before는 원본 v2의 prepared 실행이며 v1이 아니다. 평가용으로 실제 제품의
대상·검색·답변을 고치지 않는다. 현재 `current` 프로필은 실행 시 작업 폴더의 코드를 측정한다.
Validator 내용 검증을 포함한 현재 동작은 [내용 검증 안내](../doc/answer-validation.md)를 참조한다.
아래 2026-10-21 최초 검증 기록은 내용 검증 도입 전의 결과이며 새 실행과 구분한다.

현재 평가 실행은 `OPENAI_VALIDATOR_MODEL`(미설정·빈 값이면 `gpt-4o`)을 별도로 적용하고
`manifest.json`의 `requested_validator_model`에 기록한다. `/health`의 실제 설정과
대조하며, Writer·fast 모델은 기존 Before 고정 조건을 유지한다.

## 1. 자동화 테스트

프로젝트 루트에서 실행한다. Python 3.11 이상이 필요하며 현재 검증 환경은 3.14다.

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
sh sh/lint.sh
sh sh/test.sh
# CI나 다른 가상환경
PYTHON_BIN=python sh sh/lint.sh
PYTHON_BIN=python sh sh/test.sh --junitxml=/tmp/incident-tests.xml
```

Ruff는 `ruff.toml`의 `E9`, `F` 규칙으로 Python 코드의 오류·미사용 import 등을 검사한다.
린트와 pytest는 별도 명령이며, CI에서는 린트 통과 후 테스트를 실행한다.
자동 수정·포맷 검사·별도 Ruff 보고서 생성은 수행하지 않는다.

`.env` 로딩·원격 추적을 끄고 테스트용 키를 사용한다. 실제 로컬 stdio MCP와 SQLite
통합 테스트는 포함하지만 OpenAI·Ollama 모델 호출은 없다. 테스트 대역은 HTTP/SDK
외부 경계에 적용한다. `.github/workflows/tests.yml`은 push·PR·수동 실행에서 같은
검사를 실행하며 JUnit 결과를 첨부한다. 저장소에 반영하기 전에는 GitHub CI가
실행된 것으로 간주하지 않는다.

## 2. 파일만 검사

```sh
.venv/bin/python -m evaluation.run_after --preflight
# 다른 보존된 Before를 지정할 때
.venv/bin/python -m evaluation.run_after --preflight --before-dir /absolute/path/to/before
```

기본 Before는 `eval/results/before-prepared-openai-20261020T053746Z-7939ce28`다.
Git에서 제외된 로컬 자료이므로 새 체크아웃에서는 해당 실행 폴더를 따로 준비해야 한다.
자동화 단위 테스트는 이 보존 폴더 없이도 실행된다. 데이터셋 사본·현재 질문·모니터링·
런북의 해시와 기존 원자료의 기록된 해시를 검사한다. 사전 검사는 앱을 import하거나
서버·모델·LangSmith를 호출하지 않는다.

## 3. 실제 After 측정

```sh
.venv/bin/python -m evaluation.run_after --run
```

이 명령은 **유료 OpenAI API와 로컬 Ollama를 호출**한다. 기존 Before가 추적을 사용했다면
같은 LangSmith 프로젝트로 질문·프롬프트·응답·샘플 관측·런북·추적을 전송한다.
키는 기존 환경 또는 `.env`에서만 읽고 결과에 복사하지 않는다.

- 기본 모델: Before와 같은 `gpt-4o-mini`·`gpt-4o`. 다른 명시 모델 설정이면 거부한다.
- RAG: 현재 설정의 `embeddinggemma:300m`·`qwen3:8b`. 모델 다운로드·설정 변경은 하지 않는다.
- 9개 HTTP 요청을 각 1회, 동시성 1, 클라이언트 재시도 없이 실행한다. 내부 LLM 호출은
  라우팅·계획·작성·분석 단계에 따라 여러 번이며, 앱의 기존 타임아웃 재시도는 유지한다.
- 워밍업 없이 새 앱·MCP·SQLite로 시작한다. 색인 준비 시간은 `startup_ms`로 분리한다.
  모델 공급자의 캐시·적재 상태는 초기화하지 않으므로 완전한 cold-start 비교는 아니다.
- 기존 앱 서버를 재시작하지 않는다. 현재 소스를 결과 폴더에 복사하고 새 포트로 실행한다.
  `.env`·기존 DB·Git 메타데이터는 복사하지 않는다.
- 한 요청이 실패하면 다음 요청을 실행하지 않고 모든 남은 행을 `not_run`으로 보존한다.
  기존 결과 폴더를 재사용하거나 Before 안에 결과를 쓰지 않는다.
- `--port`와 `--timeout`으로 별도 포트·요청 제한을 지정할 수 있다. 기본 포트는 자동 선택이다.

새 폴더 `evaluation/results/after-current-<UTC>-<ID>/`에 manifest, 입력 사본, 요청 계획,
HTTP 원자료, 전체 수집 근거, 실제 공급자 사용량, 수동 판정 양식, 소스 스냅샷을 남긴다.
결과 디렉터리는 Git에서 제외한다. `summary.json`은 최초 실행 시 요약이며, 수동 판정·
원격 확인 후 최신 값은 아래 비교 명령으로 다시 계산한다.

## 4. 원격 추적 확인과 수동 판정

```sh
.venv/bin/python -m evaluation.verify_traces --after /absolute/path/to/after
```

LangSmith 원본 trace를 읽어 입력·최종 답변·토큰 합계를 대조하고 원본 링크·trace 지연·
가격표 추정 비용을 로컬 기록에 추가한다. 원본 원격 trace는 수정하지 않는다.
원격 집계가 아직 준비되지 않았거나 값이 다르면 실패하며 성공으로 표시하지 않는다.
이 명령의 재시도는 모델을 다시 호출하지 않는다.

After의 `accuracy-review.json`은 처음에 모두 `not_run`이다. 같은 폴더의
`records.jsonl`/`evidence.jsonl`과 원래 데이터셋의 기준·금지 사항을 대조하여 채운다.

- `reviewer`, `notes`: 판정자와 인용·근거 위치. 자동 LLM judge는 사용하지 않는다.
- `criteria`: 원래 기준 문구를 유지하고 `pass`, `fail`, `not_applicable`, `unobservable`로 판정한다.
- `answer_behavior_grade`: 모든 적용 가능한 기준과 금지 사항을 반영한 전체 행동 판정.
- `claim_grounding_grade`: 관측·알림·런북 조건·가설·실행한 조치의 구분.
- `target_grade`, `numeric_grounding_grade`: 대상과 수치의 근거 일치. 후자는 해당 없으면 `not_applicable`.
- `forbidden_checks`: 금지된 주장 여부의 검토 기록. `pass`는 위반 없음, `fail`은 위반 발견이다.

검토자·근거·기준 판정이 없으면 답변 통과율에 자동 합격으로 채우지 않는다. 기준 누락,
`unobservable`·`not_run`은 미판정이며, 정상적으로 확인된 `not_applicable` 기준은 제외한다.
기존 Before의 수동 판정은 새 자동 점수로 둔갑시키지 않고 같은 출처를 유지한다.

## 5. 로컬 비교표

```sh
.venv/bin/python -m evaluation.compare \
  --before /absolute/path/to/before --after /absolute/path/to/after \
  --output evaluation/results/comparison-new
```

`comparison.json`과 `REPORT.md`를 새 폴더에 만든다. 데이터셋·데이터·모델·반복·동시성·
추적·워밍업 조건이 다르면 비교를 거부한다. 실제 모델 ID는 양쪽에서 관측된 모델끼리
대조한다. 미호출 모델과 사용량 누락은 모델 변경/0토큰으로 취급하지 않는다.

| 지표 | 의미 |
|---|---|
| `request_success` | 성공 HTTP 요청 / 전체 계획 요청. 실패·미실행 포함 |
| `target_exact` | 관측 이벤트의 조회 대상에 필수 대상이 모두 있고 허용 외 대상이 없는지 |
| `clarification_safe` | 대상 확인 이벤트가 있고 모니터링·런북 완료 이벤트가 없는지. 자연어 의미 검사와 별도 |
| `retrieval_micro_recall` | 회수된 필수 문서 수 / 필요한 문서 수. 중복 제거, 실패·미실행도 분모에 포함 |
| `retrieval_macro_recall` | 검색 대상 요청별 recall 평균. 검색 대상이 아니면 해당 없음 |
| `answer_pass` | 수동으로 판정 가능한 요청의 엄격 통과율. 판정 수/전체 수를 함께 확인 |
| `scenario_answer_pass` | 시나리오의 모든 턴이 판정되었고 모두 통과했는지 |
| HTTP·trace 지연 | 서로 다른 측정 경계의 성공 응답 통계, 관측 개수 별도. 작은 표본 p95는 참고치 |
| OpenAI·Ollama 토큰 | 실제 공급자 응답 사용량, 임베딩 제외. 캐시·추론 하위 토큰 중복 합산 없음 |
| `simulated_cost_usd` | 앱의 가상 단가 비용 |
| `langsmith_estimated_cost_usd` | 원본 trace에 대한 LangSmith 가격표 추정 비용 |
| `actual_billed_usd` | 청구서 미확인: 항상 null |

시스템별 문서 연결 정확도는 현재 원자료에서 충분히 관측되지 않아 null이다.
검색 점수가 좋아도 답변이 사실과 맞는다는 뜻은 아니다. 제품 품질이 낮아도 평가
실행 성공과 구분해 결과를 그대로 기록한다.

## 6. LangSmith 실험 비교

먼저 전송 내용을 파일로 만든다. 이 단계에는 네트워크 호출이 없다.

```sh
.venv/bin/python -m evaluation.langsmith_feedback \
  --before /absolute/path/to/before --after /absolute/path/to/after \
  --output evaluation/results/publication-new
```

`publication.json`의 질문·응답·수집 근거·판정·측정값을 확인한 뒤 같은 명령에
`--publish`를 추가하면 기존 LangSmith 계정의 비공개 데이터셋과 실험을 만든다.
외부 전송에 필요한 사용자 승인·실행 환경 권한을 따라야 한다.

- 기존 데이터셋의 9개 요청을 공통 example ID로 연결한다. C08은 같은 세션에서 실행한
  두 턴의 원자료이며 게시 과정에서 대화를 재실행하지 않는다.
- `incident-before-…`·`incident-after-…` 실험은 원자료 **사본**이다. 원본 실행 ID와
  링크를 metadata에 남긴다. 기존 Before trace/feedback을 갱신하지 않는다.
- [LangSmith의 reference dataset 연결](https://reference.langchain.com/python/langsmith/client/Client/create_project)을 사용한다.
  receipt의 데이터셋 화면에서 두 실험을 선택해 Compare로 확인할 수 있다.
- 자동 규칙 점수와 수동 의미 판정의 출처를 feedback에 명시한다. 미판정은 전송하지 않는다.
  토큰·비용은 별도 측정 feedback이며 사본 trace를 새 LLM 호출로 표시하지 않는다.
- 같은 기록·판정으로 재실행하면 같은 ID를 조회해 재사용한다. 중간 실패 후에도 같은
  payload로 재시도한다. 기존 내용이 다르면 덮어쓰지 않고 실패한다.
- 게시한 run·feedback을 다시 읽어 대조한 뒤에만 `receipt.json`에 확인된 개수와 URL을 쓴다.
  불완전한 로컬 실행은 게시하지 않고 비교표에 실패·미실행으로 남긴다.
- 게시 직후 조회의 404는 최대 15.5초 동안 조회만 재시도한다. 계속 보이지 않으면 실패하며
  같은 payload로 재실행할 수 있다. 다른 오류·내용 불일치를 성공으로 처리하지 않는다.
- LangSmith SDK는 feedback 소수를 4자리로 반올림하므로 조회 검증도 같은 규칙을 적용한다.
  원래 정밀도는 `publication.json`·로컬 비교 결과·원본 trace에 남는다. 특히 작은 비용은
  반올림된 feedback 값 대신 원자료를 확인한다.

데이터셋·코드·조건 해시는 재현성 자료이며 평가 품질 보장이 아니다. 같은 공개 회귀셋의
한 번 측정으로 일반화 정확도나 통계적 성능 우위를 주장하지 않는다.

## 실행 검증 기록 — 2026-10-21

- `.env`·보존 평가 결과 없는 별도 소스 복사본: **352 tests + 108 subtests 통과**.
- 실제 OpenAI·로컬 Ollama 평가 1회: 8개 시나리오·9개 요청 성공, LangSmith 원본 추적 9개 입력·답변·토큰 대조 통과.
- 같은 데이터셋에 Before/After 기록 18개와 feedback 202개 게시 후 재조회 검증 완료.
- [로컬 비교 보고서](results/comparison-20261021-verified/REPORT.md),
  [원격 LangSmith 데이터셋](https://smith.langchain.com/o/9f0d284e-01c6-47cf-ae36-a0184ef6efc0/datasets/ab4d892e-f6e7-4bf1-97f9-150750c6212e).
  로컬 결과는 Git 제외 자료이며 실행한 작업공간에 보존되어 있다.
- 대상 선택 7/9→9/9, 필수 문서 회수 4/6→5/6, 엄격 답변 통과 3/9→4/9.
  평균 HTTP 지연은 4.31초→4.81초로 증가했다. 실패 기준도 보고서에 그대로 기록했다.
- GitHub CI는 구성 및 동일 명령의 로컬 검증을 완료했으며 원격 실행은 아직 하지 않았다.
