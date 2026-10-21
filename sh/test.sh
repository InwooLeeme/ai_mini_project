#!/bin/sh
# 외부 모델·추적 전송 없이 기존 회귀와 평가 기능을 함께 검사한다.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
test_python="${PYTHON_BIN:-.venv/bin/python}"
exec env PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 \
  LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false \
  OPENAI_API_KEY=offline-test-key LANGSMITH_API_KEY=offline-test-key LANGCHAIN_API_KEY=offline-test-key \
  "$test_python" -m pytest tests -q -p no:cacheprovider --tb=short "$@"
