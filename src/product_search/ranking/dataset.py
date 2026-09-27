"""按查询组织完整人工标注候选，仅开放训练集和验证集。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from product_search.artifacts import read_json, sha256, verify_release
from product_search.contracts import QueryKey, get_query_key
from product_search.data.store import get_product, iter_judgments, iter_queries, open_database
from product_search.ranking.features import FEATURE_NAMES, build_features
from product_search.retrieval.bm25 import BM25Index
from product_search.retrieval.vector_index import VectorIndex
from product_search.settings import Settings


@dataclass(frozen=True)
class TrainingData:
    """连续查询组的矩阵；query_keys 每组一项，group 每项为该组行数。"""

    X: np.ndarray
    y: np.ndarray
    group: list[int]
    query_keys: list[QueryKey]


def _verify_training_inputs(settings: Settings, dataset_dir: Path, release_dir: Path) -> dict:
    """核对数据及索引身份；只哈希标签文件，不解析 test 标签。"""
    release = verify_release(release_dir, settings)
    manifest_path = dataset_dir / 'manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('dataset_id') != release['dataset_id']:
        raise ValueError('dataset_id 与 release 不一致')
    files = manifest.get('files', {})
    if not isinstance(files, dict) or not {'catalog.sqlite', 'labels.sqlite'} <= files.keys():
        raise ValueError('dataset manifest 缺少 catalog/labels 文件哈希')
    for name, metadata in files.items():
        path = (dataset_dir / name).resolve()
        if path.parent != dataset_dir or path.name != name:
            raise ValueError('dataset manifest 只能引用本目录文件')
        expected = metadata.get('sha256') if isinstance(metadata, dict) else metadata
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'dataset 文件 SHA256 校验失败：{name}')
        if isinstance(metadata, dict) and 'size_bytes' in metadata:
            if path.stat().st_size != metadata['size_bytes']:
                raise ValueError(f'dataset 文件大小不一致：{name}')
    required = {'catalog.sqlite', 'bm25.sqlite', 'vector_manifest.json',
                'embeddings.npy', 'embedding_ids.jsonl'}
    if not required <= release['files'].keys():
        raise ValueError('release 缺少数据或索引文件哈希')
    catalog = files['catalog.sqlite']
    catalog_hash = catalog['sha256'] if isinstance(catalog, dict) else catalog
    if release['files']['catalog.sqlite'] != catalog_hash:
        raise ValueError('dataset catalog 与 release 不一致')
    manifest_hash = sha256(manifest_path)
    if ('dataset_manifest_sha256' in release
            and release['dataset_manifest_sha256'] != manifest_hash):
        raise ValueError('dataset manifest SHA256 不一致')
    if 'dataset_manifest.json' in release['files']:
        if release['files']['dataset_manifest.json'] != manifest_hash:
            raise ValueError('dataset manifest 与 release 快照不一致')
    return release


def build_training_data(
    settings: Settings, dataset_dir: str | Path, release_dir: str | Path, embedder,
    *, partition: str = 'train',
) -> TrainingData:
    """为人工 judged 候选全量补算双路分数，不按召回 top-k 截断。"""
    if partition not in ('train', 'validation'):
        raise ValueError('partition 只允许 train 或 validation，禁止读取 test 标签')
    dataset_dir, release_dir = Path(dataset_dir).resolve(), Path(release_dir).resolve()
    _verify_training_inputs(settings, dataset_dir, release_dir)
    vector_manifest = read_json(release_dir / 'vector_manifest.json')
    if embedder.signature != vector_manifest.get('signature'):
        raise ValueError('编码器 signature 与向量索引不一致')
    matrices, labels, groups, query_keys = [], [], [], []
    relevance = {'I': 0, 'C': 1, 'S': 2, 'E': 3}
    with (open_database(dataset_dir / 'labels.sqlite', read_only=True) as label_db,
          open_database(release_dir / 'catalog.sqlite', read_only=True) as catalog,
          BM25Index.load(release_dir / 'bm25.sqlite') as bm25,
          VectorIndex.load(release_dir) as vectors):
        for stored in iter_queries(label_db, partition=partition):
            query = stored.query
            key = get_query_key(query)
            if stored.source_split != 'train':
                raise ValueError('训练与验证查询必须来自官方 train')
            judgments = list(iter_judgments(label_db, key))
            if not judgments:
                raise ValueError(f'查询没有人工标注候选：{key}')
            products, keys, group_labels = [], [], []
            for judgment in judgments:
                if (judgment.locale, judgment.query_id) != key:
                    raise ValueError('人工标注混入了其他查询')
                if judgment.esci_label not in relevance:
                    raise ValueError('无效的 ESCI 标注')
                product_key = (judgment.locale, judgment.product_id)
                product = get_product(catalog, product_key)
                if product is None:
                    raise ValueError(f'catalog 缺少人工标注商品：{product_key}')
                products.append(product)
                keys.append(product_key)
                group_labels.append(relevance[judgment.esci_label])
            encoded = np.asarray(embedder.encode_queries([query]))
            if encoded.ndim != 2 or encoded.shape[0] != 1 or not np.isfinite(encoded).all():
                raise ValueError('查询编码形状或数值不正确')
            matrices.append(build_features(
                query, products, bm25_scores=bm25.score_pairs(query, keys),
                dense_scores=vectors.score_pairs(encoded[0], keys), settings=settings,
            ))
            labels.extend(group_labels)
            groups.append(len(products))
            query_keys.append(key)
    X = np.concatenate(matrices) if matrices else np.empty((0, len(FEATURE_NAMES)))
    return TrainingData(X, np.asarray(labels, dtype=np.int32), groups, query_keys)
