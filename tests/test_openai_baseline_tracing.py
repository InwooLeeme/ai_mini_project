"""실제 네트워크 없이 평가 실행기의 추적 설정 경계를 검사한다."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_openai_baseline as runner


class TracingTests(unittest.TestCase):
    def test_explicit_enable_resolves_legacy_key_and_project_override(self):
        env, public = runner.tracing_settings(
            {"LANGCHAIN_API_KEY": "test-secret", "LANGCHAIN_PROJECT": "old",
             "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com"},
            True, "selected-project",
        )
        self.assertEqual(env["LANGSMITH_API_KEY"], "test-secret")
        self.assertEqual(env["LANGSMITH_PROJECT"], "selected-project")
        self.assertEqual(env["LANGCHAIN_PROJECT"], "selected-project")
        self.assertEqual(env["LANGSMITH_TRACING"], "true")
        self.assertEqual(env["LANGCHAIN_TRACING_V2"], "true")
        self.assertNotIn("test-secret", str(public))
        self.assertEqual(public["project"], "selected-project")

    def test_missing_credentials_prevents_enabled_execution(self):
        with self.assertRaises(ValueError):
            runner.tracing_settings({}, True, "project")

    def test_default_stays_off_even_when_dotenv_enables_tracing(self):
        env, public = runner.tracing_settings(
            {"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": "test-secret"},
            False, None,
        )
        self.assertEqual(env["LANGSMITH_TRACING"], "false")
        self.assertEqual(env["LANGCHAIN_TRACING_V2"], "false")
        self.assertNotIn("LANGSMITH_API_KEY", env)
        self.assertIsNone(public)


if __name__ == "__main__":
    unittest.main()
