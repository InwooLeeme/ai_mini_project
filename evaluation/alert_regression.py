"""알림 출처 피드백과 재작성 보존 회귀. --run에서만 실제 OpenAI 모델을 호출한다."""
import argparse
import asyncio
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'evaluation/fixtures/alert-feedback.json'


def load_fixture():
    return json.loads(FIXTURE.read_text())


def regression_cases(fixture):
    original = fixture['draft']
    attributed = original.replace(
        '`cache-service`는 메모리 사용량이 89%로 높고 지속적으로 증가하고 있습니다.',
        '`cache-service`는 메모리 사용량이 89%입니다. 모니터링 알림에 따르면 최근 6시간 동안 '
        '메모리 사용량이 증가했고 힙 한계에 근접했습니다.')
    evidence = copy.deepcopy([c for c in fixture['collected'] if c['source'] != 'analysis'])
    no_alert = copy.deepcopy(evidence)
    for c in no_alert:
        if c['source'] == 'monitoring' and c['data']['system_name'] == 'cache-service':
            c['data']['alerts'] = []
    tail = ' 공통 원인은 미확인이며 추가 로그와 시계열 확인이 필요합니다.'
    return [
        {'id': 'original', 'draft': original, 'collected': fixture['collected'], 'expected': 'attribution'},
        {'id': 'attributed', 'draft': attributed, 'collected': fixture['collected'], 'expected': 'passed'},
        {'id': 'no_alert', 'draft': '1) payments-service CPU 91%입니다. '
         'cache-service는 메모리 89%이며 최근 6시간 동안 지속적으로 증가했습니다.' + tail,
         'collected': no_alert, 'expected': 'temporal'},
        {'id': 'other_metric', 'draft': '1) payments-service CPU 91%가 지속되고 있습니다. '
         'cache-service 메모리 89%입니다.' + tail, 'collected': evidence, 'expected': 'temporal'},
        {'id': 'other_system', 'draft': '1) payments-service는 메모리 55%이며 최근 6시간 동안 지속적으로 증가했습니다. '
         'cache-service 메모리 89%입니다.' + tail, 'collected': evidence, 'expected': 'temporal'},
        {'id': 'restart_89', 'draft': '1) payments-service CPU 91%입니다. '
         'cache-service 메모리 89%이므로 프로세스를 즉시 재시작하세요.' + tail,
         'collected': evidence, 'expected': 'conditions'},
    ]


def attribution_feedback_ok(checks):
    return any(c['criterion'] == 'temporal' and c['status'] == 'fail'
               and c['reason'].startswith('ALERT_ATTRIBUTION:')
               and 1 in c['evidence_ids']  # 고정 원본의 cache-service 모니터링 근거
               and retention_signals(c['correction'])['attributed_memory_trend'] for c in checks)


def retention_signals(draft):
    """고정 사례의 어휘 기반 선별 검사다. 의미·부정·인과성의 최종 판정은 수동 검토한다."""
    # 다른 시스템의 알림을 cache-service의 근거로 잘못 집계하지 않는다.
    blocks = re.split(r'(?=`?(?:payments-service|cache-service)`?)', draft)
    cache = ' '.join(b for b in blocks if re.match(r'`?cache-service', b))
    return {
        'attributed_memory_trend': bool(re.search(r'알림|경고|alert', cache, re.I)
            and re.search(r'메모리|memory', cache, re.I)
            and re.search(r'6\s*시간|6\s*h(?:ours?)?', cache, re.I)
            and re.search(r'증가|상승|climbing|increas', cache, re.I)),
        'heap_alert': bool(re.search(r'힙|heap', cache, re.I)
                           and re.search(r'한계|limit', cache, re.I)),
    }


async def evaluate(fixture, writer, validator, emit):
    results = {}
    for case in regression_cases(fixture):
        result = await validator.validate(fixture['query'], case['draft'], case['collected'], fixture['targets'])
        results[case['id']] = result
        expected = case['expected']
        passed = (result.status == 'passed' if expected == 'passed' else
                  result.status == 'rejected' and (attribution_feedback_ok(result.checks)
                  if expected == 'attribution' else any(i.code == expected for i in result.issues)))
        if case['id'] in ('no_alert', 'other_system'):
            passed = passed and any(c['criterion'] == 'temporal' and c['status'] == 'fail'
                                    and c['reason'].startswith('UNSUPPORTED_TEMPORAL:')
                                    for c in result.checks)
        emit({'kind': 'validation', **case, 'result': asdict(result), 'passed': passed})

    from agent.validation import ValidationResult, parse_review
    checks, issues = parse_review(json.dumps(fixture['recorded_review']), fixture['draft'], fixture['collected'])
    recorded = ValidationResult('rejected', tuple(issues), tuple(checks), {}, 0, True)
    for name, initial in [('current_feedback', results['original']), ('recorded_feedback', recorded)]:
        result, draft, writes, history = initial, fixture['draft'], 0, []
        # 원본은 1차 초안이다. 운영 정책과 같이 총 3차 작성까지만 확인한다.
        while result.status == 'rejected' and writes < 2:
            feedback = result.feedback()
            draft, usage = await writer.write(fixture['query'], fixture['collected'], writes + 2, feedback)
            writes += 1
            result = await validator.validate(fixture['query'], draft, fixture['collected'], fixture['targets'])
            history.append({'feedback': feedback, 'draft': draft, 'writer_usage': usage, 'validation': asdict(result)})
        signals = retention_signals(draft)
        emit({'kind': 'rewrite', 'id': name, 'writes': writes, 'history': history,
              'draft': draft, 'result': asdict(result), 'retention_signals': signals,
              'passed': writes > 0 and result.status == 'passed' and all(signals.values()),
              'manual_review': 'pending: confirm alert attribution, no fabricated trends or causal claims'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    fixture = load_fixture()
    if not args.run:
        print(f'사전 검사: {len(regression_cases(fixture))}개 판정 사례 + 2개 재작성 경로. 모델 호출 없음.')
        return 0

    from dotenv import dotenv_values
    config = {**dotenv_values(ROOT / '.env'), **os.environ}
    if not config.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY missing')
    # 이 회귀는 기존 프로젝트 자료만 사용하며, 별도 추적 전송은 하지 않는다.
    os.environ.update(OPENAI_API_KEY=config['OPENAI_API_KEY'],
                      LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false')
    from agent.openai_llm import OpenAILLM, OpenAIQualityLLM
    from agent.validation import AnswerValidator
    writer_name = config.get('OPENAI_QUALITY_MODEL', 'gpt-4o')
    validator_name = config.get('OPENAI_VALIDATOR_MODEL') or 'gpt-4o'
    writer = OpenAIQualityLLM(writer_name, 0.005)
    validator = AnswerValidator(OpenAILLM(validator_name, 0.005))
    output = args.output or ROOT / 'evaluation/results' / (
        'alert-regression-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    sources = ['agent/validation.py', 'agent/writer.py', 'agent/openai_llm.py',
               'evaluation/alert_regression.py', 'evaluation/fixtures/alert-feedback.json']
    hashes = {}
    for name in sources:
        data = (ROOT / name).read_bytes()
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    records = []
    def emit(row):
        records.append(row)
        with (output / 'records.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"{row['id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    error = None
    try:
        asyncio.run(evaluate(fixture, writer, validator, emit))
    except Exception as exc:
        error = type(exc).__name__
    summary = {'writer_model': writer_name, 'validator_model': validator_name,
               'source_sha256': hashes, 'recorded': len(records), 'expected': 8,
               'passed': sum(r['passed'] for r in records), 'error': error,
               'manual_review': 'pending', 'actual_billed_usd': None}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print('결과:', output)
    return 0 if len(records) == 8 and all(r['passed'] for r in records) and not error else 1


if __name__ == '__main__':
    raise SystemExit(main())
