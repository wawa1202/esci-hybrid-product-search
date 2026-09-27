import numpy as np
import pytest

from product_search.contracts import Hit
from product_search.retrieval.fusion import fuse_hits
from product_search.retrieval.vector_index import VectorIndex
from product_search.schemas import Query


def test_rrf_uses_ranks_and_deduplicates_across_sources():
    a = Hit(('us', 'q'), ('us', 'a'), 'bm25', 1, 99.0)
    b = Hit(('us', 'q'), ('us', 'b'), 'dense', 1, 0.9)
    c = Hit(('us', 'q'), ('us', 'a'), 'dense', 2, 0.8)
    result = fuse_hits([[a], [b, c]], constant=60)
    assert [x.product_key[1] for x in result] == ['a', 'b']
    assert result[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    with pytest.raises(ValueError):
        fuse_hits([[a, a]])
    with pytest.raises(ValueError):
        fuse_hits([[a], [Hit(('us', 'other'), ('us', 'b'), 'dense', 1, 0.9)]])


def test_vector_block_search_ties_pairs_and_integrity(tmp_path):
    keys = [('us', 'a'), ('us', 'b'), ('us', 'c')]
    vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    VectorIndex.save(tmp_path, vectors, keys, signature={'test': True})
    with VectorIndex.load(tmp_path) as index:
        result = index.search(Query('q', 'hello', 'us'), np.array([1, 0]), k=2, block_size=1)
        assert [x.product_key[1] for x in result] == ['a', 'c']
        assert index.score_pairs(np.array([0, 1]), keys)['us', 'b'] == 1
        with pytest.raises(ValueError):
            index.search(Query('q', 'hello', 'us'), np.array([0, 0]), k=2)
    path = tmp_path / 'embedding_ids.jsonl'
    path.write_text(path.read_text().replace('"a"', '"z"'))
    with pytest.raises(ValueError, match='hash|SHA|哈希'):
        VectorIndex.load(tmp_path)


def test_bad_vectors_rejected(tmp_path):
    with pytest.raises(ValueError):
        VectorIndex.save(tmp_path, np.array([[2, 0]], dtype=np.float32), [('us', 'a')], signature={})

