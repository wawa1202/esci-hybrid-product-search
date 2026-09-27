"""内存映射向量与分块精确搜索，作为 ANN 的正确性基准。"""

import heapq
import json
from pathlib import Path

import numpy as np

from product_search.artifacts import read_json, sha256, write_json
from product_search.contracts import Hit, ProductKey, get_query_key
from product_search.schemas import Query
from product_search.validation import validate_query


def _check_keys(keys):
    previous = None
    for key in keys:
        if (type(key) is not tuple or len(key) != 2 or key[0] != 'us'
                or type(key[1]) is not str or not key[1].strip()):
            raise ValueError('商品键必须是 (us, 非空商品ID)')
        if previous is not None and key <= previous:
            raise ValueError('商品键必须严格递增且唯一')
        previous = key


def _check_vectors(vectors):
    if vectors.ndim != 2 or not all(vectors.shape) or vectors.dtype != np.float32:
        raise ValueError('向量必须是非空二维 float32 矩阵')
    for start in range(0, len(vectors), 8192):
        block = vectors[start:start + 8192]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-4):
            raise ValueError('向量必须有限且已做 L2 归一化')


class VectorIndex:
    def __init__(self, vectors, keys, manifest):
        self.vectors = vectors
        self.keys = keys
        self.positions = {key: i for i, key in enumerate(keys)}
        self.manifest = manifest
        self.signature = manifest['signature']

    @staticmethod
    def finalize(directory: Path, *, signature: dict) -> None:
        """流式构建完成后校验，再写标志构建成功的 manifest。"""
        vectors = np.load(directory / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        try:
            _check_vectors(vectors)
            with (directory / 'embedding_ids.jsonl').open(encoding='utf-8') as stream:
                keys = [tuple(json.loads(line)) for line in stream]
            _check_keys(keys)
            if len(keys) != len(vectors):
                raise ValueError('向量数与商品键数不一致')
            write_json(directory / 'vector_manifest.json', {
                'format': 'exact-vectors-v1', 'shape': list(vectors.shape), 'dtype': 'float32',
                'signature': signature,
                'files': {name: sha256(directory / name) for name in ('embeddings.npy', 'embedding_ids.jsonl')},
            })
        finally:
            vectors._mmap.close()

    @classmethod
    def save(cls, directory, vectors, keys, *, signature):
        directory = Path(directory)
        _check_keys(keys)
        _check_vectors(vectors)
        if len(keys) != len(vectors):
            raise ValueError('向量数与商品键数不一致')
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'embeddings.npy').open('xb') as stream:
            np.save(stream, vectors, allow_pickle=False)
        with (directory / 'embedding_ids.jsonl').open('x', encoding='utf-8') as stream:
            for key in keys:
                stream.write(json.dumps(key) + '\n')
        cls.finalize(directory, signature=signature)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = read_json(directory / 'vector_manifest.json')
        if manifest.get('format') != 'exact-vectors-v1' or manifest.get('dtype') != 'float32':
            raise ValueError('不支持的向量格式')
        if set(manifest.get('files', {})) != {'embeddings.npy', 'embedding_ids.jsonl'}:
            raise ValueError('向量 manifest 文件集错误')
        for name, expected in manifest['files'].items():
            if sha256(directory / name) != expected:
                raise ValueError(f'向量 SHA256 hash 校验失败：{name}')
        with (directory / 'embedding_ids.jsonl').open(encoding='utf-8') as stream:
            keys = [tuple(json.loads(line)) for line in stream]
        _check_keys(keys)
        vectors = np.load(directory / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        try:
            _check_vectors(vectors)
            if list(vectors.shape) != manifest['shape'] or len(keys) != len(vectors):
                raise ValueError('向量、键和 manifest 数量不一致')
        except BaseException:
            vectors._mmap.close()
            raise
        return cls(vectors, keys, manifest)

    def _query_vector(self, vector):
        if self.vectors is None:
            raise RuntimeError('向量索引已经关闭')
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all():
            raise ValueError('查询向量维数或数值不正确')
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1) > 1e-4:
            raise ValueError('查询向量必须非零且已归一化')
        return vector

    def search(self, query: Query, vector, *, k=100, block_size=8192) -> list[Hit]:
        validate_query(query)
        if query.locale != 'us':
            raise ValueError('当前向量索引只支持 us')
        if any(type(x) is not int or x < 1 for x in (k, block_size)):
            raise ValueError('k 和 block_size 必须为正整数')
        vector = self._query_vector(vector)
        # heap 最差项在根部；-row 使同分时较早商品键优先。
        heap = []
        for start in range(0, len(self.keys), block_size):
            scores = self.vectors[start:start + block_size] @ vector
            # 每块只挑 top-k 再合并，词典序打破边界同分。
            rows = np.arange(start, start + len(scores))
            local = np.lexsort((rows, -scores))[:k]
            for j in local:
                item = (float(scores[j]), -int(rows[j]))
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        ordered = sorted(heap, key=lambda item: (-item[0], -item[1]))
        return [Hit(get_query_key(query), self.keys[-negative_row], 'dense', rank, score)
                for rank, (score, negative_row) in enumerate(ordered, 1)]

    def score_pairs(self, vector, keys: list[ProductKey]) -> dict[ProductKey, float]:
        vector = self._query_vector(vector)
        keys = list(keys)
        # 调用者的排序可以不同，结构与重复仍须验证。
        for key in keys:
            _check_keys([key])
        if len(set(keys)) != len(keys):
            raise ValueError('候选商品重复')
        result = {}
        for start in range(0, len(keys), 8192):
            block = keys[start:start + 8192]
            rows = [self.positions[key] for key in block]  # 未知键明确抛 KeyError。
            values = self.vectors[rows] @ vector
            result.update((key, float(score)) for key, score in zip(block, values, strict=True))
        return result

    def close(self):
        if self.vectors is not None:
            self.vectors._mmap.close()
            self.vectors = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
