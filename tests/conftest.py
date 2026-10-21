"""어떤 방법으로 실행해도 sh/test.sh와 같은 오프라인 환경에서 테스트한다."""
import os

# 셸이나 IDE가 넣은 실제 키·추적 설정보다 우선해, 테스트 기록이 외부로 전송되지 않게 한다.
os.environ.update({
    "PYTHON_DOTENV_DISABLED": "1",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "OPENAI_API_KEY": "offline-test-key",
    "LANGSMITH_API_KEY": "offline-test-key",
    "LANGCHAIN_API_KEY": "offline-test-key",
})
