"""组装检索流水线；数据库连接与调用保持在同一工作线程。"""

import uuid
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from product_search.artifacts import verify_release
from product_search.contracts import get_product_key
from product_search.data.store import get_product, open_database
from product_search.retrieval.bm25 import BM25Index
from product_search.retrieval.embedder import Embedder
from product_search.retrieval.fusion import fuse_hits
from product_search.retrieval.vector_index import VectorIndex
from product_search.schemas import Query, SearchResult
from product_search.service.observability import timed


class SearchEngine:
    def __init__(self, settings, release_dir, *, embedder=None):
        self.settings = settings
        self.directory = Path(release_dir)
        self.manifest = verify_release(self.directory, settings)
        required = {'catalog.sqlite', 'bm25.sqlite', 'embeddings.npy', 'embedding_ids.jsonl', 'vector_manifest.json'}
        if not required.issubset(self.manifest['files']):
            raise ValueError('发布包缺少必要的搜索文件')
        self._stack = ExitStack()
        try:
            self.catalog = self._stack.enter_context(open_database(self.directory / 'catalog.sqlite', read_only=True))
            self.bm25 = self._stack.enter_context(BM25Index.load(self.directory / 'bm25.sqlite'))
            self.vector = self._stack.enter_context(VectorIndex.load(self.directory))
            self.embedder = embedder or Embedder(settings, local_files_only=True)
            if self.vector.signature != self.embedder.signature:
                raise ValueError('查询编码器与商品向量版本不一致')
            self.ranker = None
            if 'ranker.txt' in self.manifest['files']:
                from product_search.ranking.predict import Ranker
                self.ranker = Ranker(self.directory, settings)
        except BaseException:
            self._stack.close()
            raise

    def search(self, text: str, *, top_k: int = 10, locale: str = 'us') -> dict:
        if type(text) is not str or not text.strip() or len(text) > self.settings.service.max_query_chars:
            raise ValueError('查询为空或超过长度上限')
        if type(top_k) is not int or not 1 <= top_k <= self.settings.service.max_top_k:
            raise ValueError('top_k 不合法')
        if locale != self.settings.project.locale:
            raise ValueError('当前只支持 us 市场')
        query = Query(uuid.uuid4().hex, text.strip(), locale)
        timings = {}
        with timed(timings, 'total'):
            with timed(timings, 'embedding'):
                query_vector = self.embedder.encode_queries([query])[0]
            with timed(timings, 'bm25'):
                bm25_hits = self.bm25.search(query, self.settings.bm25.top_k)
            with timed(timings, 'dense'):
                dense_hits = self.vector.search(query, query_vector, k=self.settings.retrieval.dense_top_k,
                                                block_size=self.settings.retrieval.vector_block_size)
            with timed(timings, 'fusion'):
                candidates = fuse_hits([bm25_hits, dense_hits], constant=self.settings.retrieval.rrf_constant,
                                       k=self.settings.retrieval.rerank_top_k)
            with timed(timings, 'ranking'):
                keys = [c.product_key for c in candidates]
                products = [get_product(self.catalog, key) for key in keys]
                if any(p is None for p in products):
                    raise ValueError('索引引用了商品库不存在的商品')
                if self.ranker and products:
                    # 所有候选补齐两路分数，不能把某路未召回当作该路零分。
                    results = self.ranker.predict(query, products,
                        bm25_scores=self.bm25.score_pairs(query, keys),
                        dense_scores=self.vector.score_pairs(query_vector, keys), k=top_k)
                    pipeline = 'hybrid_ltr'
                else:
                    results = [SearchResult(query.query_id, c.product_key[1], locale, i, c.rrf_score)
                               for i, c in enumerate(candidates[:top_k], 1)]
                    pipeline = 'rrf_fallback'
                by_key = {get_product_key(p): p for p in products}
                origins = {c.product_key: c for c in candidates}
                rows = []
                for result in results:
                    key = (result.locale, result.product_id)
                    product, candidate = by_key[key], origins[key]
                    rows.append({**asdict(result), 'title': product.title, 'brand': product.brand,
                                 'sources': [asdict(hit) for hit in candidate.hits],
                                 'rrf_score': candidate.rrf_score})
        return {'query': query.text, 'locale': locale, 'pipeline': pipeline,
                'release_id': self.manifest['release_id'], 'results': rows, 'timings_ms': timings}

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
