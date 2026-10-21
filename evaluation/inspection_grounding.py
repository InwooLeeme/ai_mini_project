"""실제 C02 초안과 점검 권고의 고정 회귀 사례. 판정 기준은 실행 결과와 독립적이다."""
import json
from pathlib import Path


def inspection_cases():
    fixture = json.loads((Path(__file__).parent / 'fixtures/inspection-grounding.json').read_text())
    cases = [{'id': 'c02_recorded', 'expected': 'grounding', **fixture}]
    examples = [
        ('completed_check', 'DNS와 방화벽 변경 사항을 점검하였습니다.', 'grounding'),
        ('completed_query', '최근 배포 이력을 조회했습니다.', 'grounding'),
        ('completed_logs', '로그를 확인했습니다.', 'grounding'),
        ('recommendation', 'DNS와 방화벽 변경 사항을 점검해 주세요.', 'passed'),
        ('necessity', 'DNS와 방화벽 변경 사항을 점검할 필요가 있습니다.', 'passed'),
        ('whether_checked', 'DNS와 방화벽 변경 사항을 점검했는지 확인해 주세요.', 'passed'),
        ('conditional', 'DNS와 방화벽 변경 사항을 점검했다면 그 결과를 공유해 주세요.', 'passed'),
        ('not_performed', 'DNS와 방화벽 변경 사항은 아직 점검하지 않았습니다.', 'passed'),
        ('unknown', 'DNS와 방화벽 변경 여부는 현재 근거로 확인할 수 없습니다.', 'passed'),
        ('monitoring_read', '모니터링을 조회해 gateway-service의 packet loss 3% 알림을 확인했습니다.', 'passed'),
        ('runbook_read', '런북을 조회해 DNS와 방화벽 변경 사항을 점검하라는 절차를 확인했습니다.', 'passed'),
        ('quoted_bad_claim', 'DNS와 방화벽을 "점검했습니다"라고 단정하지 말고 추가 확인이 필요합니다.', 'passed'),
        ('mixed_claim', '로그 확인이 필요합니다. DNS 변경 사항을 점검했습니다.', 'grounding'),
        ('advice_then_observation', '로그 점검은 필요하며 현재 CPU 91%를 확인했습니다.', 'passed'),
        ('negative_then_observation', 'DNS는 점검하지 않았으며 모니터링 알림을 확인했습니다.', 'passed'),
        ('conditional_then_observation', 'DNS를 점검한다면 결과를 공유해 주시고 현재 CPU 91%를 확인했습니다.', 'passed'),
    ]
    for name, text, expected in examples:
        cases.append({**fixture, 'id': name, 'expected': expected,
            'draft': '1) gateway-service에 timeout 알림, payments-service CPU 91%가 관측되었습니다.\n2) ' + text})
    return cases
