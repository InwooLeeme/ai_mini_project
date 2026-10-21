#!/bin/sh
# 프로젝트 가상환경의 Ruff로 Python 코드를 검사한다.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
lint_python="${PYTHON_BIN:-.venv/bin/python}"
exec "$lint_python" -m ruff check . --no-cache "$@"
