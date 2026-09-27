"""验收命令入口和测试集冻结边界，不访问真实 test 标签。"""
import json
from pathlib import Path
from dataclasses import replace

import pytest

from product_search.artifacts import sha256, write_json
from product_search.settings import load_settings
from product_search.cli import build_parser


def test_freeze_and_benchmark_cli_entries():
    parser = build_parser()
    assert parser.parse_args(['freeze', '--release', 'r', '--validation-report', 'v']).command == 'freeze'
    assert parser.parse_args(['benchmark', '--release', 'r', '--output', 'b.json']).command == 'benchmark'


def test_freeze_requires_matching_validation_and_preserves_receipt(tmp_path):
    from product_search.evaluation.freeze import freeze_release, claim_test_run
    root = Path(__file__).resolve().parents[1]
    settings = replace(load_settings('configs/dev.toml', project_root=root), project_root=tmp_path)
    release = tmp_path / 'release'
    release.mkdir()
    (release / 'data.txt').write_text('original')
    from product_search.artifacts import runtime_signature
    write_json(release / 'release.json', {
        'release_id': 'test-release', 'dataset_id': 'd', 'runtime_signature': runtime_signature(settings),
        'files': {'data.txt': sha256(release / 'data.txt')},
    })
    validation = tmp_path / 'validation.json'
    write_json(validation, {'release_id': 'test-release', 'dataset_id': 'd', 'partition': 'test'})
    with pytest.raises(ValueError):
        freeze_release(settings, release, validation)
    validation.write_text(json.dumps({'release_id': 'test-release', 'dataset_id': 'd', 'partition': 'validation',
                                      'queries': 5, 'ranking_k': 10, 'retrieval_k': 100}))
    frozen = freeze_release(settings, release, validation)
    assert frozen.exists()
    claim_test_run(settings, release, tmp_path / 'out')
    with pytest.raises(FileExistsError):
        claim_test_run(settings, release, tmp_path / 'out2')


def test_evaluation_without_ranker_analyzes_only_available_methods():
    from product_search.evaluation.error_analysis import analyze_errors
    row = {'query_id': 'q', 'text': 'shoe', 'labels': {'a': 'I', 'b': 'E'},
           'metrics': {s: {'ndcg': .63, 'mrr_e': .5} for s in ('bm25', 'dense', 'rrf')},
           'rankings': {s: ['a', 'b'] for s in ('bm25', 'dense', 'rrf')}}
    result = analyze_errors([row])
    assert result and not any('ltr' in c for c in result[0]['categories'])
