"""입력 최적화 비교에서 실행 순서와 미측정 사용량을 검증한다."""
import pytest
import json
from pathlib import Path

from eval import compare_rag_inputs as evaluation


def test_balanced_order_for_every_case():
    cases = [{"id": "a"}, {"id": "b"}]
    rows = list(evaluation.paired_inputs(cases, 4))
    for case in cases:
        pairs = [rows[i:i+2] for i in range(0, len(rows), 2)
                 if rows[i][1] == case]
        assert [p[0][2] for p in pairs].count("baseline") == 2
        assert [p[0][2] for p in pairs].count("optimized") == 2
        assert all({r[2] for r in p} == {"baseline", "optimized"} for p in pairs)


@pytest.mark.parametrize("repetitions", [0, 1, 3, -2])
def test_uneven_comparison_rejected(repetitions):
    with pytest.raises(ValueError):
        list(evaluation.paired_inputs([{"id": "a"}], repetitions))


def test_missing_tokens_are_not_reported_as_zero():
    summary = evaluation.token_summary([
        {"usage": {"selector_input_tokens": 100, "selector_output_tokens": 20}},
        {"usage": {}},
        {"usage": {"selector_input_tokens": 0, "selector_output_tokens": 0}},
    ])
    assert summary["selector_input_tokens"] == {"sum": 100, "mean": 50, "n": 2}
    assert summary["selector_output_tokens"] == {"sum": 20, "mean": 10, "n": 2}


def test_diagnostic_fixtures_match_app_profiles_and_evidence():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "eval/rag-diagnostic-dataset.json").read_text())
    profiles = json.loads((root / "data/monitoring.json").read_text())
    assert len({c["id"] for c in data["cases"]}) == len(data["cases"])
    for case in data["cases"]:
        for obs in case["input"]["observations"]:
            profile = profiles[obs["system_name"]]
            assert obs["cpu_percent"] == profile["cpu_base"]
            assert obs["memory_percent"] == profile["memory_base"]
            assert obs["status"] == profile["status"]
            assert obs["alerts"] == profile["alerts"]
        assert {e["document_id"] for e in case["evidence"]} == set(case["required_document_ids"])
        for evidence in case["evidence"]:
            text = (root / "data/runbooks" / (evidence["document_id"] + ".md")).read_text()
            assert all(anchor["text"] in text for anchor in evidence["required_evidence"])
