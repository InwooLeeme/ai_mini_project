"""기존 모델 반환값·가상 단가를 유지하면서 실제 사용량만 표준 trace에 기록한다."""
import copy
import asyncio
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from langsmith import Client, traceable, tracing_context, get_current_run_tree
import httpx
import langsmith_usage as usage
from openai_baseline_server import make_recorded_post
from agent.openai_llm import OpenAILLM, OpenAIFastLLM, OpenAIQualityLLM


RESPONSE = {
    "model": "gpt-4o-mini-2024-07-18",
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
              "prompt_tokens_details": {"cached_tokens": 40},
              "completion_tokens_details": {"reasoning_tokens": 5}},
}


class UsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # local tracing도 기본 Client 생성 시 /info를 조회하므로 오프라인 정보를 주입한다.
        cls.client = Client(api_url="http://127.0.0.1:1", api_key="offline-test-key",
                            auto_batch_tracing=False, info={"version": "0"})

    def traced(self, callback, run_type="llm"):
        captured = []

        @traceable(run_type=run_type)
        def call():
            callback()
            captured.append(get_current_run_tree())
            return {"unchanged": True}

        with tracing_context(enabled="local", client=self.client):
            self.assertEqual(call(), {"unchanged": True})
        return captured[0]

    def test_provider_tokens_cache_and_model_survive_trace_finalization(self):
        original = copy.deepcopy(RESPONSE)
        run = self.traced(lambda: usage.attach_openai_usage(RESPONSE, "gpt-4o-mini"))
        self.assertEqual(RESPONSE, original)
        self.assertEqual(run.metadata["ls_model_name"], "gpt-4o-mini-2024-07-18")
        self.assertEqual(run.metadata["ls_provider"], "openai")
        self.assertEqual(run.metadata["usage_metadata"], {
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_token_details": {"cache_read": 40},
            "output_token_details": {"reasoning": 5},
        })
        self.assertIsNone(run.metadata["actual_billed_usd"])
        self.assertNotIn("total_cost", run.metadata["usage_metadata"])
        self.assertEqual(run.outputs, {"unchanged": True})

    def test_virtual_rate_comes_from_instance_without_polluting_real_cost(self):
        result = ("answer", {"tokens": 120, "usd": 0.0006})
        class Model:
            usd_per_1k = 0.005
            def chat(self):
                return result
        Model.chat = usage.with_virtual_cost_metadata(Model.chat)
        run = self.traced(lambda: self.assertIs(Model().chat(), result))
        self.assertEqual(run.metadata["simulated_cost_usd"], 0.0006)
        self.assertEqual(run.metadata["simulated_usd_per_1k_tokens"], 0.005)
        self.assertNotIn("usage_metadata", run.metadata)

    def test_no_parent_usage_to_prevent_double_counting(self):
        run = self.traced(lambda: usage.attach_openai_usage(RESPONSE, "gpt-4o-mini"), "chain")
        self.assertNotIn("usage_metadata", run.metadata)

    def test_unknown_model_preserves_tokens_without_inventing_prices(self):
        data = {**RESPONSE, "model": "custom-model"}
        run = self.traced(lambda: usage.attach_openai_usage(data, "alias"))
        self.assertEqual(run.metadata["ls_model_name"], "custom-model")
        self.assertEqual(run.metadata["usage_metadata"]["total_tokens"], 120)
        self.assertNotIn("total_cost", run.metadata["usage_metadata"])

    def test_missing_or_invalid_usage_is_not_zero_usage(self):
        for bad in (None, {}, {"prompt_tokens": -1, "completion_tokens": 1},
                    {"prompt_tokens": True, "completion_tokens": 1},
                    {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 99},
                    {"prompt_tokens": 10, "completion_tokens": 5,
                     "prompt_tokens_details": {"cached_tokens": 11}}):
            with self.subTest(bad=bad):
                run = self.traced(lambda: usage.attach_openai_usage({"usage": bad}, "alias"))
                self.assertNotIn("usage_metadata", run.metadata)
                self.assertNotEqual(run.metadata["provider_usage_status"], "recorded")

    def test_zero_and_missing_optional_total_are_valid(self):
        run = self.traced(lambda: usage.attach_openai_usage(
            {"usage": {"prompt_tokens": 0, "completion_tokens": 0}}, "alias"))
        self.assertEqual(run.metadata["usage_metadata"]["total_tokens"], 0)

    def test_tracing_off_is_a_noop(self):
        with tracing_context(enabled=False):
            usage.attach_openai_usage(RESPONSE, "alias")

    def test_real_baseline_route_and_async_write_keep_response_and_virtual_cost(self):
        captured, records = [], []
        responses = [
            {**RESPONSE, "choices": [{"message": {"content": '{"route":"incident","needs_analysis":true}'}}]},
            {**RESPONSE, "model": "gpt-4o-2024-08-06",
             "choices": [{"message": {"content": "1) unchanged answer"}}]},
        ]
        def transport(url, **kwargs):
            self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
            self.assertEqual(kwargs["json"]["temperature"], 0)
            captured.append(get_current_run_tree())
            return httpx.Response(200, json=responses.pop(0), request=httpx.Request("POST", url))

        post = make_recorded_post(transport, lambda name, entry: records.append(entry), lambda: "thread")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key"}), \
             patch("httpx.post", post), \
             patch.object(OpenAILLM, "_chat", usage.with_virtual_cost_metadata(OpenAILLM._chat)), \
             tracing_context(enabled="local", client=self.client):
            route, fast_usage = OpenAIFastLLM("gpt-4o-mini", 0.00015).route("CPU 확인")
            answer, quality_usage = asyncio.run(
                OpenAIQualityLLM("gpt-4o", 0.005).write("CPU 확인", [], 1, None))
        self.assertEqual(route, {"route": "incident", "needs_analysis": True})
        self.assertEqual(answer, "1) unchanged answer")
        self.assertEqual(fast_usage["usd"], 0.000018)
        self.assertEqual(quality_usage["usd"], 0.0006)
        self.assertEqual(len(captured), 2)
        self.assertEqual(sum(r.metadata["usage_metadata"]["total_tokens"] for r in captured), 240)
        self.assertEqual(captured[1].metadata["simulated_cost_usd"], 0.0006)
        self.assertEqual(records[0]["langsmith_llm_run_id"], str(captured[0].id))
        self.assertEqual(records[1]["usage"]["total_tokens"], 120)
        self.assertNotIn("offline-test-key", str(records))

    def test_http_error_and_other_endpoints_are_not_recorded_as_usage(self):
        records = []
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
        post = make_recorded_post(lambda *a, **k: response,
                                  lambda name, entry: records.append(entry), lambda: "thread")
        self.assertIs(post("https://example.org"), response)
        self.assertEqual(records, [])
        self.assertIs(post("https://api.openai.com/v1/chat/completions"), response)
        self.assertEqual(records[0]["http_status"], 429)
        self.assertNotIn("usage", records[0])


if __name__ == "__main__":
    unittest.main()
