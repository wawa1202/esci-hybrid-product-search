import numpy as np
import pytest
pytest.importorskip('faiss')

from product_search.retrieval.ann_index import FaissIndex


def test_hnsw_roundtrip_and_wrong_keys(tmp_path):
    vectors = np.eye(4, dtype=np.float32)
    keys = [('us', str(i)) for i in range(4)]
    path = tmp_path / 'ann.faiss'
    index = FaissIndex.build(vectors, keys, path, ef_search=32)
    assert index.manifest['type'] == 'HNSW'
    assert index.search(vectors[2], 1) == [(keys[2], 1.0)]
    loaded = FaissIndex.load(path, keys)
    assert loaded.search(vectors[1], 1)[0][0] == keys[1]
    with pytest.raises(ValueError):
        FaissIndex.load(path, list(reversed(keys)))
    with pytest.raises(FileExistsError):
        FaissIndex.build(vectors, keys, path)
    with pytest.raises(ValueError):
        loaded.search(np.zeros(4), 2)
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError):
        FaissIndex.load(path, keys)
