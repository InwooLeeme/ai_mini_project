import hashlib
import json

import pytest

from evaluation.contracts import load_bundle, preflight, request_plan, write_json
from evaluation.compare import compare_bundles


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "repo"
    before = tmp_path / "before"
    (root / "eval").mkdir(parents=True)
    (root / "data").mkdir()
    before.mkdir()
    dataset = {"dataset_id": "test", "version": "1", "scenario_count": 1,
               "request_count_per_repetition": 2, "cases": [{"id": "C08", "turns": [
                   {"turn": 1, "message": "제 이름은 민수", "expected_route": "chat",
                    "expected_behavior": "acknowledge_name", "required_monitoring_targets": [],
                    "allowed_monitoring_targets": [], "expected_retrieval_document_union": [], "criteria": []},
                   {"turn": 2, "message": "이름은?", "depends_on_turn": 1, "expected_route": "chat",
                    "expected_behavior": "recall_name", "required_monitoring_targets": [],
                    "allowed_monitoring_targets": [], "expected_retrieval_document_union": [], "criteria": []}]}]}
    for p in (root / "eval/dataset.json", before / "dataset.json"):
        write_json(p, dataset)
    (root / "data/monitoring.json").write_text('{}')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"run_id": "before-test", "dataset_sha256": digest(before / "dataset.json"),
                "backend": "openai", "requested_fast_model": "gpt-4o-mini",
                "requested_quality_model": "gpt-4o", "temperature": 0, "repetitions": 1,
                "concurrency": 1, "client_retry": 0, "tracing_enabled": True,
                "warmup": "none; fresh application and MCP processes",
                "conditions": {"monitoring_metrics": "fixed", "research_artificial_delay_seconds": 0,
                               "forced_first_failure_default": False},
                "source_manifest": {"files_sha256": {"data/monitoring.json": digest(root / "data/monitoring.json")}}}
    write_json(before / "manifest.json", manifest)
    rows = [{"case_id": "C08", "turn": 1, "repetition": 1, "thread_id": "s", "message": "제 이름은 민수",
             "response": {"reply": "민수님", "metadata": {"route": "chat"}, "events": []},
             "measurement": {"status": "response_received", "elapsed_ms": 100}}]
    (before / "raw.jsonl").write_text(json.dumps(rows[0], ensure_ascii=False) + '\n')
    return root, before


def test_missing_turn_is_materialized_and_not_reported_as_zero_cost(baseline):
    _, before = baseline
    bundle = load_bundle(before)
    assert len(bundle["records"]) == 2
    assert bundle["records"][1]["status"] == "not_run"
    assert bundle["records"][1]["model_usage"] is None


def test_plan_keeps_two_turn_session_and_separates_repetitions(baseline):
    _, before = baseline
    dataset = json.loads((before / "dataset.json").read_text())
    plan = request_plan(dataset, "run", repetitions=2)
    assert len(plan) == 4
    assert plan[0]["thread_id"] == plan[1]["thread_id"]
    assert plan[1]["thread_id"] != plan[2]["thread_id"]
    assert plan[3]["turn"] == 2


def test_preflight_rejects_changed_dataset_or_monitoring(baseline):
    root, before = baseline
    original = (before / "manifest.json").read_bytes()
    assert preflight(root, before)["manifest"]["run_id"] == "before-test"
    (root / "data/monitoring.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="data/monitoring.json"):
        preflight(root, before)
    assert (before / "manifest.json").read_bytes() == original


def test_preflight_rejects_changed_dataset_before_network_access(baseline):
    root, before = baseline
    (root / "eval/dataset.json").write_text('{}')
    with pytest.raises(ValueError, match="dataset"):
        preflight(root, before)


def test_duplicate_or_foreign_rows_are_rejected(baseline):
    _, before = baseline
    raw = before / "raw.jsonl"
    raw.write_text(raw.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_bundle(before)


def test_compare_rejects_model_mismatch_and_never_overwrites_before(baseline):
    _, before = baseline
    bundle = load_bundle(before)
    import copy
    after = copy.deepcopy(bundle)
    after["manifest"]["requested_quality_model"] = "another-model"
    with pytest.raises(ValueError, match="requested_quality_model"):
        compare_bundles(bundle, after)


def test_compare_pairs_missing_turns_and_keeps_manual_coverage(baseline):
    _, before = baseline
    bundle = load_bundle(before)
    result = compare_bundles(bundle, bundle)
    assert len(result["pairs"]) == 2
    assert result["before"]["request_success"]["eligible"] == 2
    assert result["before"]["answer_pass"]["eligible"] == 1  # unexecuted turn failed; received turn not reviewed
    assert result["before"]["openai_total_tokens"] is None


def test_partial_run_does_not_reject_an_unexecuted_model_as_model_change(baseline):
    import copy
    _, directory = baseline
    before = load_bundle(directory)
    after = copy.deepcopy(before)
    before['records'][0]['model_usage'] = [
        {'provider': 'openai', 'requested_model': 'gpt-4o-mini', 'resolved_model': 'mini-v1', 'usage': None},
        {'provider': 'openai', 'requested_model': 'gpt-4o', 'resolved_model': 'quality-v1', 'usage': None}]
    after['records'][0]['model_usage'] = [
        {'provider': 'openai', 'requested_model': 'gpt-4o-mini', 'resolved_model': 'mini-v1', 'usage': None}]
    assert compare_bundles(before, after)['after']['openai_total_tokens'] is None
    after['records'][0]['model_usage'][0]['resolved_model'] = 'mini-v2'
    with pytest.raises(ValueError, match='resolved models'):
        compare_bundles(before, after)


def test_default_cli_preflight_never_reads_credentials_or_starts_network(baseline, monkeypatch):
    import evaluation.run_after as runner
    import dotenv
    import socket
    root, before = baseline
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight must not touch network or credentials')
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(dotenv, 'dotenv_values', forbidden)
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    assert runner.main(['--before-dir', str(before)]) == 0
    assert not (root / 'evaluation').exists()
