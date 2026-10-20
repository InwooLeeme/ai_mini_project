# LangSmith 토큰·비용 표시

OpenAI **평가 실행기**의 계측에 적용한다. `server.py`, `agent/openai_llm.py`,
동결된 baseline 57개 파일, 기존 결과 폴더와 기존 원격 trace는 변경하지 않는다.
일반 `server.py` 직접 실행에는 이 평가용 래퍼가 자동 적용되지 않는다.

## 구분하는 값

| 항목 | 출처와 의미 | 표시 위치 |
|---|---|---|
| 실제 사용 토큰 | OpenAI 응답 `usage` | LangSmith Input / Output / Total tokens |
| 공급자 단가 기준 추정 비용 | 실제 사용량 × LangSmith 모델 가격표 | LangSmith 표준 Cost |
| 가상 비용 | 기존 코드의 가상 단가 × 전체 토큰, 호출별 6자리 반올림 | LLM Metadata의 `simulated_cost_usd` |
| 실제 청구액 | 공급자 청구서와 대사해야 확정 가능 | `actual_billed_usd=null`, 미확인 |

LangSmith Cost는 실제 청구서 금액이 아니다. 프로젝트의 가격표, 캐시 가격,
계약 할인 등의 조건에 따라 청구액과 다를 수 있다. 코드에서 현재 공급자 가격을
임의로 고정하거나, 가상 비용을 표준 Cost에 덮어쓰지 않는다.
LangSmith 가격표에 없는 모델은 토큰은 기록되더라도 비용이 미산출될 수 있다.

기존 `/chat`의 `metadata.cost_usd`와 graph state의 `cost_usd`는 호환성을 위해
그대로 **가상 비용**이다. 원격 trace Metadata의 `legacy_cost_usd_semantics`에
이 의미를 표시한다. 실제 청구액을 0으로 채우지 않는다.

## 적용 방법

다음 명령은 사전 검사만 수행한다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python eval/run_openai_baseline.py \
  --langsmith --langsmith-project capstone-all-total-openai
```

다음 명령은 **유료 OpenAI 호출과 LangSmith 전송을 포함**하며 데이터셋을 새로 1회 실행한다.
이번 연결 수정 검증에서는 실행하지 않았다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python eval/run_openai_baseline.py \
  --run --langsmith --langsmith-project capstone-all-total-openai
```

새 결과 폴더에는 `cost_policy`, 계측 모듈의 SHA-256 및 모듈 사본도 보존한다.
`model-usage.jsonl`에 실제 provider usage와 대응하는 `langsmith_llm_run_id`를 기록한다.

## 구현과 확인 방법

- `langsmith_usage.py`가 실제 응답 모델명을 `ls_model_name`, 공급자를
  `ls_provider=openai`로 연결하고 현재 LLM span의 `usage_metadata`를 설정한다.
- 입력·출력 토큰을 분리하고 캐시 읽기는 `input_token_details.cache_read`,
  추론 토큰은 `output_token_details.reasoning`으로 보낸다.
  세부 토큰은 합계에 다시 더하지 않는다.
- 기존 LLM span만 사용하며, 상위 graph/request span에 usage를 중복 주입하지 않는다.
  요청 합계는 LangSmith가 하위 LLM에서 집계한다.
- 가상 단가는 모델 인스턴스에서 가져온다. 요청 모델명이 같더라도 fast/quality
  인스턴스의 가상 단가가 다르면 각 인스턴스의 기존 계산을 유지한다.
- 누락되거나 잘못된 usage를 실제 0토큰으로 변환하지 않고
  `provider_usage_status=missing_or_invalid`로 표시한다.
- 표준 사용량만 추가하므로 기존 답변, 라우팅, 재시도, 반환 비용은 바뀌지 않는다.

원격 검증은 `capstone-all-total-openai-usage-verification` 프로젝트에 기존 usage
두 건을 **재생**하여 수행한다. `validation_only=true`,
`usage_source=historical_provider_usage_replay`인 trace는 새로운 모델 실행이나
성능 측정 결과로 집계하지 않는다. 기존 baseline 프로젝트와 기록은 보존한다.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_langsmith_usage.py' -v
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tests -p 'test_openai_baseline_tracing.py' -v
```

기준: [LangSmith 공식 비용 추적 문서](https://docs.langchain.com/langsmith/cost-tracking).
표준 metadata 사용량을 적용하는 방식은 함수 반환값을 변경하지 않고 비용을
집계할 수 있게 한다.

## 로컬 검증 기록

기존 사용량 재생 검증의 수치·trace 링크는 Git에서 제외한
`eval/results/documentation-archive/LANGSMITH_USAGE-before-git.md`에 보존한다.
새 체크아웃의 테스트와 동결 Before 재측정 방법은
[평가 코드 배포 안내](REPOSITORY_USAGE.md)를 참조한다.
