# 평가 코드와 baseline 보존본 사용

## Git에 포함하는 파일

- 평가 코드: `run_prepared_baseline.py`, `run_openai_baseline.py`,
  `openai_baseline_server.py`, `langsmith_usage.py`.
- 고정 입력과 기준: `dataset.json`, `before-source-manifest.json`,
  `manual-review-template.csv`. 판정 양식은 측정 결과가 아닌 빈 템플릿이다.
- 평가 안내와 `../baseline/README.md`, `../tests/`의 회귀 테스트.

계획 문서, `eval/results/`의 측정값·로그·캡처, baseline 압축본·patch·manifest·
검증 산출물은 Git에 포함하지 않는다. 기존 로컬 파일은 삭제하거나 갱신하지 않는다.
`before-source-manifest.json`은 원본 코드의 기준 해시이며 실제 측정 결과가 아니다.

## 새 체크아웃에서 테스트

프로젝트 의존성을 설치한 환경에서 저장소 루트를 기준으로 실행한다.

```sh
PYTHONDONTWRITEBYTECODE=1 LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  .venv/bin/python -m unittest discover -s tests -v
```

이 테스트는 실제 모델 호출·외부 LangSmith 전송 없이 실행된다. baseline 테스트는
실제 로컬 stdio MCP와 fake LLM을 사용한다. 실행기 테스트는 임시 baseline 파일과
모의 HTTP 응답을 사용하므로 별도 압축본·측정 결과·API 키가 필요하지 않다.

## Before 재측정은 동결된 복원본에서만 실행

평가 실행기의 `preflight`는 현재 파일 전체를 동결된 prepared manifest와 비교한다.
Git 배포용 `.gitignore`·문서 정리도 이 해시를 바꾸므로 현재 체크아웃에서 과거
Before를 다시 측정하려 하면 해시 불일치가 발생할 수 있다. 새 체크아웃에는
보존본 manifest·압축본 자체가 없다. 둘 다 검증을 우회하거나 기존 해시를
새 코드에 맞춰 수정할 이유가 아니다.

재측정이 필요하면 별도 보관한 다음 파일을 준비한다.

- `baseline/prepared-baseline.tar.gz`
- `baseline/prepared-manifest.json`
- `baseline/prepared-baseline.patch`

현재 작업 폴더에 덮어쓰지 말고 압축본을 새로운 빈 디렉터리에 복원한다.
그 디렉터리의 `baseline/`에 동일한 manifest·patch·압축본을 배치하고, 해당
디렉터리를 작업 위치로 하여 [prepared 측정 안내](PREPARED_BASELINE.md)를 따른다.
이 세 파일은 하나의 보존 세트이며 현재 저장소에 맞춰 재생성하지 않는다.
원본 as_is의 압축본과 prepared 압축본을 섞지 않는다.

OpenAI 계측은 복원 디렉터리의 `eval/`에 현재 저장소의
`run_openai_baseline.py`, `openai_baseline_server.py`, `langsmith_usage.py`를 추가해
사용한다. 이 세 파일은 prepared manifest의 동결 대상에 포함되지 않은 계측 코드다.
이미 동결되어 있는 `run_prepared_baseline.py`, 데이터셋, 문서 등은 덮어쓰지 않는다.
호출 조건과 계측 방식은 [LangSmith 사용량 안내](LANGSMITH_USAGE.md)를 따른다.

압축본이 없는 상태에서도 위 단위 테스트와 일반 애플리케이션 실행은 가능하지만
동결된 Before의 재측정은 할 수 없다. 실제 재측정에는 별도 모델/API 설정이 필요하며
`--run`을 명시했을 때만 모델 호출이 시작된다.

## 문서의 시점 구분

- [기존 README](README.md)는 원본 as_is 준비 시점의 기록이다.
- [prepared 안내](PREPARED_BASELINE.md)와 [baseline 안내](../baseline/README.md)는
  변동 요인을 정리하고 보존본을 만든 시점의 설명이다.
- 이들 문서의 `미측정`, `미커밋` 등 표현은 작성 당시 상태다. 실제 실행 상태와
  수치는 Git에서 제외한 개별 결과 폴더의 manifest·원자료를 기준으로 확인한다.
- 기존 사용량 재생 검증 기록은 로컬 결과 폴더에 보존하며 평가 코드 배포에 포함하지 않는다.
