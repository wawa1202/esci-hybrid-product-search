"""保存选定版本的验收凭据；防止在同一发布上反复消费 test。"""
from pathlib import Path
from datetime import datetime, timezone
import re

from product_search.artifacts import read_json, sha256, verify_release, write_json


def _receipt_path(settings, release):
    identity = release['release_id']
    if not isinstance(identity, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', identity):
        raise ValueError('release_id 不能包含路径分隔符')
    return settings.resolve_path(settings.data.reports_dir) / 'frozen' / f'{identity}.json'


def freeze_release(settings, release_dir, validation_report) -> Path:
    release_dir, validation_report = Path(release_dir), Path(validation_report)
    release = verify_release(release_dir, settings)
    report = read_json(validation_report)
    if (report.get('partition') != 'validation' or report.get('queries', 0) < 1
            or report.get('release_id') != release['release_id']
            or report.get('dataset_id') != release['dataset_id']):
        raise ValueError('必须提供同一发布版本、同一数据集的有效 validation 报告')
    evaluation = {'ranking_k': settings.evaluation.ranking_k, 'retrieval_k': settings.evaluation.retrieval_k}
    if any(report.get(key) != value for key, value in evaluation.items()):
        raise ValueError('冻结配置与 validation 的指标截断不同')
    output = _receipt_path(settings, release)
    write_json(output, {'release_id': release['release_id'], 'dataset_id': release['dataset_id'],
                       'release_sha256': sha256(release_dir / 'release.json'),
                       'validation_report_sha256': sha256(validation_report), 'evaluation': evaluation,
                       'frozen_at_utc': datetime.now(timezone.utc).isoformat()})
    return output


def claim_test_run(settings, release_dir, output_dir) -> Path:
    release_dir = Path(release_dir)
    release = verify_release(release_dir, settings)
    frozen = _receipt_path(settings, release)
    receipt = read_json(frozen)
    if (receipt.get('release_sha256') != sha256(release_dir / 'release.json')
            or receipt.get('dataset_id') != release['dataset_id']
            or receipt.get('evaluation') != {'ranking_k': settings.evaluation.ranking_k,
                                            'retrieval_k': settings.evaluation.retrieval_k}):
        raise ValueError('冻结记录与当前模型、数据或评估口径不同')
    used = frozen.with_suffix('.used.json')
    write_json(used, {'output': str(Path(output_dir).resolve()), 'status': 'started',
                     'started_at_utc': datetime.now(timezone.utc).isoformat()})
    return used
