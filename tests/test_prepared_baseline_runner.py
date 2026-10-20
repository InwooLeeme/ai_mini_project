"""측정기는 실제 HTTP 대신 고정 응답으로 검증한다. 모델·DB 접근 없음."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "eval/run_prepared_baseline.py"


class Reply(io.BytesIO):
    status = 200

    def __init__(self, value):
        super().__init__(json.dumps(value).encode())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "prepared baseline 전용 실행기가 필요합니다")
        spec = importlib.util.spec_from_file_location("prepared_runner", SCRIPT)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "eval").mkdir()
        (self.root / "baseline").mkdir()
        (self.root / "source.py").write_text("# frozen source\n")
        (self.root / "eval/dataset.json").write_bytes((ROOT / "eval/dataset.json").read_bytes())
        for name in ("prepared-baseline.tar.gz", "prepared-baseline.patch"):
            (self.root / "baseline" / name).write_bytes(b"test snapshot")
        self.baseline = {
            "profile": "prepared-baseline-fixed-metrics-no-demo-delay-no-forced-failure",
            "files_sha256": {
                name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                for name in ("source.py", "eval/dataset.json")
            },
            "conditions": {
                "monitoring_metrics": "fixed", "last_checked": "actual query time",
                "research_artificial_delay_seconds": 0,
                "forced_first_failure_default": False,
            },
            "archive": "prepared-baseline.tar.gz",
            "archive_sha256": hashlib.sha256(b"test snapshot").hexdigest(),
            "tracked_patch": "prepared-baseline.patch",
            "tracked_patch_sha256": hashlib.sha256(b"test snapshot").hexdigest(),
        }
        (self.root / "baseline/prepared-manifest.json").write_text(json.dumps(self.baseline))
        self.requests = []
        self.backend = "ollama"
        self.failure_at = None
        self.failure_kind = "http"
        self.addCleanup(patch.stopall)
        patch.object(self.runner, "ROOT", self.root).start()
        patch.object(self.runner.request, "urlopen", side_effect=self.urlopen).start()

    def urlopen(self, req, timeout):
        if isinstance(req, str):
            self.assertTrue(req.endswith("/health"))
            return Reply({
                "status": "ok", "llm_backend": self.backend,
                "supervisor_version": "v2", "instance_id": 123,
                "creation_count": 1, "fast_llm": "ollama:fast",
                "quality_llm": "ollama:quality",
            })
        self.assertTrue(req.full_url.endswith("/chat"))
        self.assertEqual(req.get_method(), "POST")
        self.requests.append(json.loads(req.data))
        if len(self.requests) == self.failure_at:
            if self.failure_kind == "timeout":
                raise TimeoutError("offline timeout")
            if self.failure_kind == "malformed":
                return Reply({"reply": "broken", "events": [], "metadata": []})
            raise HTTPError(req.full_url, 500, "offline failure", {}, io.BytesIO(b"failure"))
        return Reply({
            "reply": "offline fixture",
            "events": [{"type": "TaskCompleted", "source": "monitoring", "target": "payments-service"}],
            "metadata": {
                "route": "incident", "task_count": 2, "gate_removed": 0,
                "replan_rounds": 1, "retry_count": 1, "parallel_elapsed": 0.04,
                "validation_attempts": 3, "cost_usd": 0.002,
                "llm_backend": "ollama", "supervisor_version": "v2",
                "instance_id": 123, "langsmith_run_id": None,
            },
        })

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.runner.main(list(args))

    def result(self):
        folders = list((self.root / "eval/results").iterdir())
        self.assertEqual(len(folders), 1)
        folder = folders[0]
        manifest = json.loads((folder / "manifest.json").read_text())
        rows = [json.loads(s) for s in (folder / "raw.jsonl").read_text().splitlines()]
        return folder, manifest, rows

    def test_default_preflight_never_opens_network_or_creates_results(self):
        with patch.object(self.runner.request, "urlopen", side_effect=AssertionError("network forbidden")):
            self.assertEqual(self.run_cli(), 0)
        self.assertFalse((self.root / "eval/results").exists())

    def test_changed_source_stops_before_network(self):
        (self.root / "source.py").write_text("# changed")
        with patch.object(self.runner.request, "urlopen", side_effect=AssertionError("network forbidden")):
            self.assertEqual(self.run_cli("--run"), 1)
        self.assertFalse((self.root / "eval/results").exists())

    def test_changed_archive_is_rejected(self):
        (self.root / "baseline/prepared-baseline.tar.gz").write_bytes(b"changed")
        self.assertEqual(self.run_cli("--preflight"), 1)
        self.assertFalse((self.root / "eval/results").exists())

    def test_changed_dataset_is_rejected(self):
        (self.root / "eval/dataset.json").write_text("{}")
        self.assertEqual(self.run_cli("--run"), 1)
        self.assertEqual(self.requests, [])

    def test_openai_is_rejected_before_chat(self):
        self.backend = "openai"
        self.assertEqual(self.run_cli("--run"), 1)
        self.assertEqual(self.requests, [])
        self.assertFalse((self.root / "eval/results").exists())

    def test_collects_27_requests_preserving_memory_and_separating_sessions(self):
        self.assertEqual(self.run_cli("--run", "--repetitions", "3"), 0)
        folder, manifest, rows = self.result()
        self.assertTrue(folder.name.startswith("before-prepared-"))
        self.assertEqual(len(rows), 27)
        self.assertEqual(len({r["thread_id"] for r in rows}), 24)
        for repetition in (1, 2, 3):
            memory = [r for r in rows if r["repetition"] == repetition and r["case_id"] == "C08"]
            self.assertEqual([r["turn"] for r in memory], [1, 2])
            self.assertEqual(memory[0]["thread_id"], memory[1]["thread_id"])
        self.assertTrue(manifest["completed_plan"])
        self.assertEqual(manifest["conditions"], self.baseline["conditions"])
        self.assertEqual(manifest["successful_responses"], 27)
        self.assertEqual(rows[0]["measurement"]["retry_count"], 1)
        self.assertEqual(rows[0]["measurement"]["rewrite_count"], 2)
        self.assertEqual(rows[0]["measurement"]["simulated_cost_usd"], 0.002)
        self.assertIsNone(rows[0]["measurement"]["actual_billed_usd"])
        self.assertGreaterEqual(rows[0]["measurement"]["elapsed_ms"], 0)

    def test_http_error_on_memory_first_turn_stops_without_retry(self):
        self.failure_at = 8
        self.assertEqual(self.run_cli("--run", "--repetitions", "1"), 1)
        _, manifest, rows = self.result()
        self.assertEqual(len(self.requests), 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[-1]["measurement"]["http_status"], 500)
        self.assertIsNone(rows[-1]["measurement"]["retry_count"])
        self.assertFalse(manifest["completed_plan"])
        self.assertEqual(manifest["unexecuted_requests"], 1)
        self.assertEqual(manifest["successful_responses"], 7)

    def test_timeout_is_recorded_as_unknown_metrics_and_stops(self):
        self.failure_at, self.failure_kind = 1, "timeout"
        self.assertEqual(self.run_cli("--run", "--repetitions", "1"), 1)
        _, manifest, rows = self.result()
        self.assertEqual(len(rows), 1)
        self.assertEqual(manifest["unexecuted_requests"], 8)
        self.assertIsNone(rows[0]["measurement"]["simulated_cost_usd"])
        self.assertIn("TimeoutError", rows[0]["error"])

    def test_malformed_response_is_recorded_and_stops(self):
        self.failure_at, self.failure_kind = 1, "malformed"
        self.assertEqual(self.run_cli("--run"), 1)
        _, manifest, rows = self.result()
        self.assertEqual(rows[0]["measurement"]["status"], "error")
        self.assertFalse(manifest["completed_plan"])


if __name__ == "__main__":
    unittest.main()
