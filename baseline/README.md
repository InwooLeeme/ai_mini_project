# 성능 비교용 baseline 준비

7개 개선 기능을 시작하기 전, 원본 진단 동작을 유지하면서 실습용 변동 요인만 정리한 상태다.
실제 성능·답변 품질 측정은 아직 수행하지 않았다.

## 원본 보존

- 원본 커밋: `3fb307358af4d9451c3c3c2040b41f6a042cc1ef`
- 작업 브랜치: `chore/prepare-baseline` (원본 `main` 커밋 유지)
- 원본 압축본: `original-3fb3073.tar.gz`
- 압축본과 원본 추적 파일의 SHA-256: `original-manifest.json`
- 압축본은 변경 전 Git 추적 파일 47개와 바이트 단위로 일치함을 확인했다.
- 기존 미추적 `docs/`, `eval/`은 작업 폴더에 그대로 두었다. `.env`, `.venv`, `session.db`는 압축본에 포함하지 않는다.

원본을 확인하려면 기존 작업 폴더에 덮어쓰지 말고 새 폴더에 압축을 푼다.
압축본에는 Git 이력이 없으며 이력은 현재 저장소의 원본 커밋에 남아 있다.

## 이번에 바뀐 실행 조건

| 항목 | 원본 | 현재 baseline |
|---|---|---|
| 모니터링 CPU·메모리 | 기본값에 난수 추가 | `data/monitoring.json`의 기본값 사용 |
| 조사 태스크 지연 | 태스크마다 `sleep(0.35)` | 제거 |
| 최초 MCP 모니터링 호출 | 의도적으로 timeout 발생 | 기본 비활성화 |
| 실제 timeout 재시도 | 최대 3회 시도, 0.3초·0.9초 backoff | 변경 없음 |
| 영구 오류 | 재시도 없이 전파 | 변경 없음 |

고정 수치:

| 시스템 | CPU (%) | 메모리 (%) |
|---|---:|---:|
| auth-service | 78.0 | 61.0 |
| web-01 | 34.0 | 48.0 |
| payments-service | 91.0 | 55.0 |
| cache-service | 42.0 | 89.0 |
| gateway-service | 30.0 | 45.0 |
| log-service | 20.0 | 35.0 |

`last_checked`는 실제 조회 시각을 계속 기록한다. 수치·상태·알림이 고정된 것이며
응답 전체의 바이트나 모델 응답까지 결정론적으로 만든 것은 아니다.
미등록 시스템의 `unknown` 응답 등 기존 기능 동작은 유지한다.

장애 복구 평가에서만 `MCPClient(flaky_first_tool_call=True)`로 기존 실패 주입을
명시적으로 켤 수 있다. 정상 성능 평가에는 기본 생성자를 사용한다.
서버 실행 스크립트의 기동 대기와 실제 오류의 backoff는 인위적 조사 지연과 다르므로 유지한다.

## 확인 방법과 범위

저장소 루트에서:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_baseline_preparation.py' -v
```

이 검사는 외부 추적을 비활성화하며 실제 stdio MCP 서버와 fake LLM을 사용한다.
6개 시스템 반복 조회, 기본 강제 실패 비활성화, 명시적 실패 주입 시 복구,
조사 지연 제거, backoff·시도 상한·영구 오류 처리, 기존 v2 그래프 실행을 확인한다.
애플리케이션 `.env`·대화 DB를 읽거나 변경할 필요가 없다.

실모델 응답 품질·지연시간·LangSmith 전송은 이 검사로 검증하지 않는다.
이번 테스트는 baseline 정리의 회귀 검증이며, 7개 개선안의 테스트·평가 기능 구현이 아니다.

## 이후 성능 평가에 사용할 기준

- 원본 코드와 이 baseline을 구분한다. 발표의 기능 개선 Before는 **현재 정리된 baseline**이다.
- `prepared-manifest.json`은 정리된 추적 파일·테스트·평가 데이터의 해시와 환경 버전을 기록한다.
  아직 새 커밋이 없으므로 원본 커밋 SHA만으로 현재 baseline을 식별하지 않는다.
- `prepared-baseline.tar.gz`와 `prepared-baseline.patch`는 개선 시작 후에도 현재 상태를 복원하기 위한 자료다.
- 기존 `eval/before-source-manifest.json`과 `eval/README.md`는 원본 `as_is` 기준 자료다.
  해당 문서의 원본 해시 검사를 우회해 현재 baseline을 원본으로 측정하지 않는다.
  현재 baseline은 [전용 측정 안내](../eval/PREPARED_BASELINE.md)의
  `eval/run_prepared_baseline.py`로 측정하고 `before-prepared-*` 결과 폴더를 사용한다.
- 기존 `eval/dataset.json`의 질문·기대 동작은 바꾸지 않았다.
- 이후 After에서도 같은 고정 데이터·모델·프롬프트 조건·세션 초기화·워밍업 정책을 적용한다.
  프롬프트를 개선하는 실험은 그 변경을 명시한다.
- 강제 실패 없는 정상 평가와 실패를 명시적으로 주입하는 복구 평가는 분리한다.
- 인위적 지연 제거의 효과를 7개 기능 개선의 성과에 포함하지 않는다.

## 전용 측정기와 보존본 갱신

```sh
# 기본 또는 --preflight: HTTP·모델 호출 없는 검사
.venv/bin/python eval/run_prepared_baseline.py --preflight

# 향후 실제 측정 시에만 사용 (이번 준비에서는 실행하지 않음)
.venv/bin/python eval/run_prepared_baseline.py --run --repetitions 3
```

`prepared-manifest.json`은 현재 `.gitignore`, 전용 실행기·안내·테스트까지 포함한다.
원본 `original-*`, 기존 `eval/README.md`·원본 manifest·데이터셋은 유지했다.
patch는 원본 대비 Git 추적 파일 변경만 담고, 신규 실행기·테스트는 tar.gz에 포함한다.
복원에는 manifest·patch·tar.gz를 한 묶음으로 보관한다.
manifest·patch·tar.gz는 자기 참조를 피하기 위해 tar.gz 안에 포함하지 않는다.
측정 결과·환경 비밀값·세션 DB·가상환경·설계 문서는 보존본에서 제외한다.
`verification.json`의 준비 테스트 기록은 변경 없는 진단 코드에 대한 기존 검증 근거이고,
`runner_preflight`는 이번 모의 HTTP 측정기 검사 결과다. 둘 다 실모델 성능 수치가 아니다.
