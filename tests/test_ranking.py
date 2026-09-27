"""排序模块的特征、数据隔离和发布闭环测试。"""

from dataclasses import replace
from pathlib import Path
import json
import shutil

import numpy as np
import pytest

from product_search.schemas import Product, Query
from product_search.settings import load_settings


@pytest.fixture
def settings():
    root = Path(__file__).resolve().parents[1]
    value = load_settings('configs/dev.toml', project_root=root)
    return replace(value, training=replace(value.training, n_estimators=12,
                                          early_stopping_rounds=3))


def test_features_are_fixed_and_candidate_independent(settings):
    from product_search.ranking.features import FEATURE_NAMES, build_features

    assert isinstance(FEATURE_NAMES, tuple)
    assert FEATURE_NAMES == (
        'bm25_score', 'dense_score', 'query_length', 'title_length',
        'overlap_count', 'overlap_ratio', 'brand_match', 'color_match',
        'bullet_length', 'bullet_overlap_count', 'bullet_overlap_ratio',
    )
    query = Query('q', 'ACME red red shoe', 'us')
    a = Product('a', 'us', 'Red shoe', brand='Acme', color='red',
                bullet_point='red comfortable shoe')
    b = Product('b', 'us', 'Green boot')
    scores = {('us', 'a'): 3.5, ('us', 'b'): 0.0}
    dense = {('us', 'a'): 0.8, ('us', 'b'): -0.2}
    matrix = build_features(query, [a, b], bm25_scores=scores, dense_scores=dense,
                            settings=settings)
    np.testing.assert_allclose(matrix[0], [3.5, .8, 4, 2, 2, 2/3, 1, 1, 3, 2, 2/3])
    alone = build_features(replace(query, query_id='other'), [replace(a, product_id='x')],
                           bm25_scores={('us', 'x'): 3.5},
                           dense_scores={('us', 'x'): .8}, settings=settings)
    np.testing.assert_array_equal(alone[0], matrix[0])
    np.testing.assert_array_equal(
        build_features(query, [b, a], bm25_scores=scores, dense_scores=dense,
                       settings=settings), matrix[::-1])


@pytest.mark.parametrize('failure', ['missing', 'extra', 'duplicate', 'locale', 'nan', 'inf'])
def test_features_reject_bad_keys_or_scores(settings, failure):
    from product_search.ranking.features import build_features

    products = [Product('a', 'us', 'red shoe')]
    bm25, dense = {('us', 'a'): 1.0}, {('us', 'a'): .5}
    if failure == 'missing':
        dense.clear()
    elif failure == 'extra':
        bm25[('us', 'b')] = 0.0
    elif failure == 'duplicate':
        products *= 2
    elif failure == 'locale':
        products[0] = replace(products[0], locale='es')
    else:
        dense[('us', 'a')] = float(failure)
    with pytest.raises(ValueError):
        build_features(Query('q', 'shoe', 'us'), products, bm25_scores=bm25,
                       dense_scores=dense, settings=settings)


def test_features_empty_tokens_and_empty_candidates(settings):
    from product_search.ranking.features import FEATURE_NAMES, build_features

    query = Query('q', '!!!', 'us')
    matrix = build_features(query, [Product('a', 'us', '!!!')],
                            bm25_scores={('us', 'a'): 0},
                            dense_scores={('us', 'a'): 0}, settings=settings)
    assert matrix.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()
    assert not matrix.any()
    assert build_features(query, [], bm25_scores={}, dense_scores={},
                          settings=settings).shape == (0, len(FEATURE_NAMES))


class TinyEmbedder:
    """离线测试编码器，不下载模型，记录实际编码的查询。"""

    def __init__(self, settings):
        self.signature = {
            'model_id': settings.embedding.model_id, 'revision': settings.embedding.revision,
            'max_seq_length': settings.embedding.max_seq_length,
            'normalize_embeddings': True, 'dimension': 2,
            'text_version': settings.text.version, 'max_tokens': settings.text.max_tokens,
            'tokenizer_version': 'casefold-before-filter-v2',
        }
        self.seen = []

    def encode_queries(self, queries):
        self.seen.extend(q.query_id for q in queries)
        assert all(q.query_id != 'test-secret' for q in queries)
        return np.tile(np.array([[1, 0]], dtype=np.float32), (len(queries), 1))


@pytest.fixture
def bundle(tmp_path, settings):
    from product_search.artifacts import runtime_signature, sha256, write_json
    from product_search.data.store import (init_schema, insert_judgment, insert_product,
                                           insert_query, open_database, transaction)
    from product_search.retrieval.bm25 import BM25Index
    from product_search.retrieval.vector_index import VectorIndex
    from product_search.schemas import Judgment

    dataset, release = tmp_path / 'dataset', tmp_path / 'source'
    dataset.mkdir()
    release.mkdir()
    products = [Product(f'p{i}', 'us', f'shoe red size {i}', brand='Acme') for i in range(48)]
    products.sort(key=lambda p: p.product_id)
    with open_database(dataset / 'catalog.sqlite', create=True) as conn:
        init_schema(conn, 'catalog')
        with transaction(conn):
            for p in products:
                insert_product(conn, p)
    with open_database(dataset / 'labels.sqlite', create=True) as conn:
        init_schema(conn, 'labels')
        with transaction(conn):
            for qid, partition in [('a', 'train'), ('b', 'train'), ('v', 'validation'),
                                    ('test-secret', 'test')]:
                insert_query(conn, Query(qid, f'shoe {qid}', 'us'),
                             source_split='test' if partition == 'test' else 'train',
                             partition=partition)
                for i, p in enumerate(products[:44] if qid == 'b' else products):
                    insert_judgment(conn, Judgment(qid, p.product_id, 'us', 'ICSE'[i % 4]))
    files = {p.name: {'sha256': sha256(p), 'size_bytes': p.stat().st_size}
             for p in dataset.iterdir()}
    write_json(dataset / 'manifest.json', {'dataset_id': 'tiny', 'files': files})
    shutil.copy2(dataset / 'catalog.sqlite', release / 'catalog.sqlite')
    shutil.copy2(dataset / 'manifest.json', release / 'dataset_manifest.json')
    with BM25Index.build(products, release / 'bm25.sqlite', text_version=settings.text.version,
                         max_tokens=settings.text.max_tokens, k1=settings.bm25.k1,
                         b=settings.bm25.b):
        pass
    embedder = TinyEmbedder(settings)
    values = np.array([[1, 0] if i % 4 >= 2 else [0, 1]
                       for i in range(len(products))], dtype=np.float32)
    VectorIndex.save(release, values, [('us', p.product_id) for p in products],
                     signature=embedder.signature)
    write_json(release / 'release.json', {
        'release_id': 'source', 'dataset_id': 'tiny',
        'runtime_signature': runtime_signature(settings),
        'dataset_manifest_sha256': sha256(dataset / 'manifest.json'),
        'files': {p.name: sha256(p) for p in release.iterdir()},
    })
    return dataset, release, embedder, products


def test_dataset_uses_all_judged_candidates_and_only_requested_partition(bundle, settings,
                                                                       monkeypatch):
    from product_search.ranking import dataset as module

    original = module.iter_judgments
    seen = []

    def guarded(conn, key):
        assert key[1] != 'test-secret'
        seen.append(key)
        return original(conn, key)

    monkeypatch.setattr(module, 'iter_judgments', guarded)
    dataset, release, embedder, _ = bundle
    train = module.build_training_data(settings, dataset, release, embedder)
    assert list(train.group) == [48, 44]
    assert list(train.query_keys) == [('us', 'a'), ('us', 'b')]
    assert train.X.shape == (92, 11)
    assert train.y.tolist() == [0, 1, 2, 3] * 23
    valid = module.build_training_data(settings, dataset, release, embedder,
                                       partition='validation')
    assert list(valid.group) == [48]
    assert list(valid.query_keys) == [('us', 'v')]
    assert seen == [('us', 'a'), ('us', 'b'), ('us', 'v')]
    assert embedder.seen == ['a', 'b', 'v']


def test_dataset_rejects_test_before_io(settings, tmp_path):
    from product_search.ranking.dataset import build_training_data

    for partition in ['test', 'all', 'valid']:
        with pytest.raises(ValueError, match='partition|train|validation'):
            build_training_data(settings, tmp_path, tmp_path, None, partition=partition)


def test_dataset_detects_embedder_mismatch(bundle, settings):
    from product_search.ranking.dataset import build_training_data

    dataset, release, embedder, _ = bundle
    embedder.signature = {**embedder.signature, 'revision': 'different'}
    with pytest.raises(ValueError, match='signature|编码|签名'):
        build_training_data(settings, dataset, release, embedder)


@pytest.mark.parametrize('tamper', ['dataset_id', 'file', 'manifest'])
def test_train_rejects_dataset_mismatch_before_training(bundle, settings, tmp_path, tamper):
    from product_search.ranking.train import train_ranker

    dataset, release, embedder, _ = bundle
    if tamper == 'file':
        with (dataset / 'labels.sqlite').open('ab') as stream:
            stream.write(b'changed')
    else:
        path = dataset / 'manifest.json'
        value = json.loads(path.read_text())
        value['dataset_id' if tamper == 'dataset_id' else 'extra'] = 'different'
        path.write_text(json.dumps(value))
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError):
        train_ranker(settings, dataset, release, output, embedder=embedder)
    assert not output.exists()


@pytest.mark.parametrize('dropped', [(), ('dense_score', 'brand_match')])
def test_train_predict_release_roundtrip(bundle, settings, tmp_path, dropped):
    from product_search.artifacts import read_json, runtime_signature, sha256, verify_release
    from product_search.ranking.features import FEATURE_NAMES, build_features
    from product_search.ranking.predict import Ranker
    from product_search.ranking.train import train_ranker
    import lightgbm as lgb

    dataset, release, embedder, products = bundle
    before = {p.name: sha256(p) for p in release.iterdir()}
    output = tmp_path / 'trained'
    result = train_ranker(settings, dataset, release, output, embedder=embedder,
                          drop_features=dropped)
    assert result == output
    assert before == {p.name: sha256(p) for p in release.iterdir()}
    manifest = verify_release(output, settings)
    assert manifest['release_id'] != 'source'
    assert manifest['dataset_id'] == 'tiny'
    assert set(manifest['files']) == {p.name for p in output.iterdir()} - {'release.json'}
    assert not (output / 'labels.sqlite').exists()
    model = read_json(output / 'model_manifest.json')
    active = [name for name in FEATURE_NAMES if name not in dropped]
    assert model['feature_names'] == active
    assert model['drop_features'] == list(dropped)
    assert model['runtime_signature'] == runtime_signature(settings)
    assert model['dataset_id'] == 'tiny'
    assert model['train_groups'] == 2 and model['train_rows'] == 92
    assert model['valid_groups'] == 1 and model['valid_rows'] == 48
    assert 1 <= model['best_iteration'] <= settings.training.n_estimators
    assert model['model_sha256'] == sha256(output / 'ranker.txt')
    booster = lgb.Booster(model_file=str(output / 'ranker.txt'))
    assert booster.feature_name() == active
    text = (output / 'ranker.txt').read_text()
    assert '[num_threads: 4]' in text and '[device_type: cpu]' in text
    assert '[label_gain: 0,1,3,7]' in text
    assert f'[seed: {settings.project.seed}]' in text
    ranker = Ranker(output, settings)
    query = Query('online', 'shoe red', 'us')
    selected = products[:3][::-1]
    scores = {('us', p.product_id): 1.0 for p in selected}
    dense = {key: .2 for key in scores}
    X = build_features(query, selected, bm25_scores=scores, dense_scores=dense,
                        settings=settings)
    expected = booster.predict(X[:, [FEATURE_NAMES.index(name) for name in active]])
    results = ranker.predict(query, selected, bm25_scores=scores, dense_scores=dense)
    expected_ids = [selected[i].product_id for i in sorted(
        range(3), key=lambda i: (-expected[i], ('us', selected[i].product_id)))]
    assert [r.product_id for r in results] == expected_ids
    assert [r.rank for r in results] == [1, 2, 3]
    assert all(r.query_id == 'online' and type(r.score) is float for r in results)
    assert ranker.predict(query, selected, bm25_scores=scores, dense_scores=dense, k=1) == results[:1]
    assert ranker.predict(query, [], bm25_scores={}, dense_scores={}) == []
    for bad_k in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            ranker.predict(query, selected, bm25_scores=scores, dense_scores=dense, k=bad_k)
    with pytest.raises(FileExistsError):
        train_ranker(settings, dataset, release, output, embedder=embedder)


@pytest.mark.parametrize('drop', [('unknown',), ('dense_score', 'dense_score'), 'dense_score'])
def test_train_rejects_invalid_ablation_before_io(settings, tmp_path, drop):
    from product_search.ranking.train import train_ranker

    with pytest.raises(ValueError, match='drop_features|特征'):
        train_ranker(settings, tmp_path, tmp_path, tmp_path / 'out', drop_features=drop)


def test_train_rejects_relative_output_before_io(settings, tmp_path):
    from product_search.ranking.train import train_ranker

    with pytest.raises(ValueError, match='绝对|absolute'):
        train_ranker(settings, tmp_path, tmp_path, 'relative-output')
