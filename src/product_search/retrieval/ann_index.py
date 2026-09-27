"""可选 FAISS HNSW 近似索引；精确分块搜索仍是默认基准。"""
from pathlib import Path
import os
import uuid
import numpy as np

from product_search.artifacts import fingerprint, read_json, sha256, write_json
from product_search.retrieval.vector_index import _check_keys, _check_vectors


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('需要安装可选依赖 faiss-cpu') from exc
    return faiss


class FaissIndex:
    """近似检索仅作独立实验，不直接替换 SearchEngine 的精确召回。"""
    def __init__(self, index, keys, manifest):
        self.index, self.keys, self.manifest = index, keys, manifest

    @classmethod
    def build(cls, vectors, keys, output, *, ef_search=64, m=32):
        if any(type(x) is not int or x < 1 for x in (ef_search, m)):
            raise ValueError('ef_search/m 必须为正整数')
        output = Path(output)
        sidecar = output.with_suffix(output.suffix + '.json')
        if output.exists() or sidecar.exists():
            raise FileExistsError(output)
        keys = list(keys)
        _check_keys(keys)
        _check_vectors(vectors)
        if len(keys) != len(vectors):
            raise ValueError('向量数与商品键数不同')
        faiss = _faiss()
        faiss.omp_set_num_threads(1)
        index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = max(80, ef_search)
        for start in range(0, len(vectors), 8192):
            index.add(np.ascontiguousarray(vectors[start:start + 8192], dtype=np.float32))
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name('.' + output.name + '.' + uuid.uuid4().hex)
        try:
            faiss.write_index(index, str(temp))
            os.link(temp, output)
        finally:
            if temp.exists():
                temp.unlink()
        manifest = {'type': 'HNSW', 'count': len(keys), 'dimension': index.d,
                    'm': m, 'ef_search': ef_search, 'keys_sha256': fingerprint(keys),
                    'index_sha256': sha256(output)}
        write_json(sidecar, manifest)
        return cls(index, keys, manifest)

    @classmethod
    def load(cls, path, keys):
        path, keys = Path(path), list(keys)
        _check_keys(keys)
        manifest = read_json(path.with_suffix(path.suffix + '.json'))
        if manifest.get('type') != 'HNSW' or manifest.get('keys_sha256') != fingerprint(keys):
            raise ValueError('ANN 商品键或格式不匹配')
        if sha256(path) != manifest.get('index_sha256'):
            raise ValueError('ANN 文件哈希不匹配')
        index = _faiss().read_index(str(path))
        if index.ntotal != len(keys) or index.d != manifest['dimension']:
            raise ValueError('ANN 向量数或维数不一致')
        index.hnsw.efSearch = manifest['ef_search']
        return cls(index, keys, manifest)

    def search(self, vector, k):
        vector = np.asarray(vector, dtype=np.float32)
        if (type(k) is not int or k < 1 or vector.shape != (self.index.d,)
                or not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > 1e-4):
            raise ValueError('k 或查询向量不合法，向量须归一化')
        scores, positions = self.index.search(vector.reshape(1, -1), min(k, len(self.keys)))
        result = [(self.keys[int(pos)], float(score)) for pos, score in zip(positions[0], scores[0]) if pos >= 0]
        return sorted(result, key=lambda item: (-item[1], item[0]))
