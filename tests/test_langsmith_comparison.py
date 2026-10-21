import copy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langsmith.utils import LangSmithNotFoundError

from evaluation.langsmith_feedback import build_publication, publish


class MemoryClient:
    """원격 저장소 경계만 대체한다. 중복 ID, 참조 무결성, read-back을 검사한다."""
    def __init__(self):
        self.datasets, self.examples, self.projects, self.runs, self.feedback = {}, {}, {}, {}, {}
        self.writes = 0

    def read_dataset(self, *, dataset_name):
        if dataset_name not in self.datasets:
            raise LangSmithNotFoundError("missing")
        return self.datasets[dataset_name]

    def create_dataset(self, dataset_name, *, description, metadata):
        self.writes += 1
        d = SimpleNamespace(id=uuid4(), name=dataset_name, description=description, metadata=metadata, url="https://smith.test/dataset")
        self.datasets[dataset_name] = d
        return d

    def read_example(self, example_id):
        if str(example_id) not in self.examples:
            raise LangSmithNotFoundError("missing")
        return self.examples[str(example_id)]

    def create_example(self, *, example_id, dataset_id, inputs, outputs, metadata):
        assert str(example_id) not in self.examples
        self.writes += 1
        e = SimpleNamespace(id=example_id, dataset_id=dataset_id, inputs=inputs, outputs=outputs, metadata=metadata)
        self.examples[str(example_id)] = e
        return e

    def read_project(self, *, project_name):
        if project_name not in self.projects:
            raise LangSmithNotFoundError("missing")
        return self.projects[project_name]

    def create_project(self, project_name, *, reference_dataset_id, metadata, description):
        self.writes += 1
        p = SimpleNamespace(id=uuid4(), reference_dataset_id=reference_dataset_id, metadata=metadata,
                            url="https://smith.test/experiment")
        self.projects[project_name] = p
        return p

    def read_run(self, run_id):
        if str(run_id) not in self.runs:
            raise LangSmithNotFoundError("missing")
        return self.runs[str(run_id)]

    def create_run(self, *, id, name, inputs, run_type, project_name, reference_example_id,
                   outputs, start_time, end_time, error, extra):
        assert str(id) not in self.runs
        assert str(reference_example_id) in self.examples
        self.writes += 1
        self.runs[str(id)] = SimpleNamespace(id=id, inputs=inputs, outputs=outputs, error=error,
                                            reference_example_id=reference_example_id, extra=extra)

    def read_feedback(self, feedback_id):
        if str(feedback_id) not in self.feedback:
            raise LangSmithNotFoundError("missing")
        return self.feedback[str(feedback_id)]

    def create_feedback(self, run_id, key, *, feedback_id, score, comment, source_info, session_id=None):
        assert str(feedback_id) not in self.feedback
        assert str(run_id) in self.runs
        self.writes += 1
        assert session_id is not None
        f = SimpleNamespace(id=feedback_id, run_id=run_id, key=key, score=round(score, 4), comment=comment)
        self.feedback[str(feedback_id)] = f
        return f


def bundle(phase):
    from evaluation.metrics import score_record
    expected = {"turn": 1, "message": "어느 시스템?", "expected_route": "incident", "expected_behavior": "clarify_target",
                "required_monitoring_targets": [], "allowed_monitoring_targets": [], "expected_retrieval_document_union": [],
                "criteria": ["대상 확인"], "forbidden": []}
    record = {"case_id": "C05", "turn": 1, "repetition": 1, "message": expected["message"], "status": "ok",
              "response": {"reply": "선택해 주세요", "events": [{"type": "TargetConfirmationRequired"}], "metadata": {"route": "incident"}},
              "trace_elapsed_ms": 50, "http_elapsed_ms": 70, "model_usage": [], "state": None,
              "trace_started_at": "2026-10-20T05:37:00+00:00",
              "source_run_id": str(uuid4()), "review": None}
    record["scores"] = score_record(expected, record)
    return {"manifest": {"run_id": phase, "dataset_sha256": "a" * 64, "repetitions": 1, "data_sha256": {}},
            "dataset": {"dataset_id": "test", "version": "1", "scenario_count": 1, "request_count_per_repetition": 1,
                        "cases": [{"id": "C05", "turns": [expected]}]}, "records": [record]}


def test_same_examples_link_both_experiments_and_never_modify_source_traces():
    before, after = bundle("before"), bundle("after")
    snapshot = copy.deepcopy((before, after))
    spec = build_publication(before, after)
    client = MemoryClient()
    result = publish(spec, client)
    assert len(client.datasets) == 1 and len(client.examples) == 1
    assert len(client.projects) == 2 and len(client.runs) == 2
    assert len({r.reference_example_id for r in client.runs.values()}) == 1
    assert before["records"][0]["source_run_id"] not in client.runs
    assert after["records"][0]["source_run_id"] not in client.runs
    assert result["verified_runs"] == 2
    assert (before, after) == snapshot


def test_rerun_reuses_identical_records_without_duplicate_writes():
    spec, client = build_publication(bundle("before"), bundle("after")), MemoryClient()
    publish(spec, client)
    writes = client.writes
    publish(spec, client)
    assert client.writes == writes


def test_manual_unjudged_score_is_not_published_as_zero():
    spec = build_publication(bundle("before"), bundle("after"))
    client = MemoryClient()
    publish(spec, client)
    assert not any(f.key == "answer_pass" for f in client.feedback.values())
    assert any(f.key == "clarification_safe" and f.score == 1 for f in client.feedback.values())


def test_foreign_existing_dataset_is_not_modified():
    spec, client = build_publication(bundle("before"), bundle("after")), MemoryClient()
    name = spec["dataset_name"]
    client.datasets[name] = SimpleNamespace(id=uuid4(), metadata={"owner": "someone else"})
    with pytest.raises(ValueError, match="dataset"):
        publish(spec, client)
    assert client.writes == 0


def test_readback_mismatch_is_failure_not_success():
    spec, client = build_publication(bundle("before"), bundle("after")), MemoryClient()
    publish(spec, client)
    next(iter(client.runs.values())).outputs = {"reply": "changed"}
    with pytest.raises(ValueError, match="run"):
        publish(spec, client)


class DelayedReadClient(MemoryClient):
    def __init__(self, delay_reads=2):
        super().__init__()
        self.remaining = {}
        self.delay_reads = delay_reads

    def create_run(self, **kwargs):
        super().create_run(**kwargs)
        self.remaining[str(kwargs['id'])] = self.delay_reads

    def create_feedback(self, run_id, key, **kwargs):
        super().create_feedback(run_id, key, **kwargs)
        self.remaining[str(kwargs['feedback_id'])] = self.delay_reads

    def read_run(self, run_id):
        self._wait_for_visibility(run_id)
        return super().read_run(run_id)

    def read_feedback(self, feedback_id):
        self._wait_for_visibility(feedback_id)
        return super().read_feedback(feedback_id)

    def _wait_for_visibility(self, resource_id):
        key = str(resource_id)
        if self.remaining.get(key, 0):
            self.remaining[key] -= 1
            raise LangSmithNotFoundError('not indexed yet')


def test_delayed_run_and_feedback_visibility_retries_reads_without_recreating(monkeypatch):
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    spec = build_publication(bundle('before'), bundle('after'))
    client = DelayedReadClient()
    result = publish(spec, client)
    assert result['verified_runs'] == 2
    assert result['verified_feedback'] == len(client.feedback)
    assert client.writes == 1 + 1 + 2 + 2 + len(client.feedback)


def test_unavailable_readback_stops_after_bounded_reads(monkeypatch):
    sleeps = []
    monkeypatch.setattr('time.sleep', lambda seconds: sleeps.append(seconds))
    client = DelayedReadClient(delay_reads=1000)
    with pytest.raises(LangSmithNotFoundError):
        publish(build_publication(bundle('before'), bundle('after')), client)
    assert 0 < len(sleeps) <= 6
    assert sum(sleeps) <= 30
    assert len(client.runs) == 1 and len(client.feedback) == 0


def test_sdk_feedback_rounding_is_explicit_and_larger_mismatch_is_rejected():
    before, after = bundle('before'), bundle('after')
    after['records'][0]['langsmith_estimated_cost_usd'] = '0.0030831'
    spec = build_publication(before, after)
    client = MemoryClient()
    result = publish(spec, client)
    assert result['feedback_score_decimal_places'] == 4
    recorded = spec['experiments'][1]['rows'][0]['feedback'][-1]
    assert recorded['score'] == 0.0030831
    remote = next(f for f in client.feedback.values() if f.key == 'langsmith_estimated_cost_usd')
    assert remote.score == 0.0031
    remote.score = 0.0032
    with pytest.raises(ValueError, match='feedback'):
        publish(spec, client)
