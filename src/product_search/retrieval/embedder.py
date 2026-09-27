"""固定版本的本地 MiniLM 编码器；不调用聊天模型，也不读取标签。"""

import numpy as np

from product_search.schemas import Product, Query
from product_search.settings import Settings
from product_search.retrieval.text import product_text, query_tokens
from product_search.validation import validate_query


class Embedder:
    def __init__(self, settings: Settings, *, local_files_only: bool = False):
        settings.require_embedding_ready()
        from sentence_transformers import SentenceTransformer
        import torch
        # 避免小模型在笔记本上启用过多 CPU 线程而反而变慢。
        torch.set_num_threads(settings.training.num_threads)
        self.settings = settings
        self.model = SentenceTransformer(
            settings.embedding.model_id,
            revision=settings.embedding.revision,
            device=settings.embedding.device,
            cache_folder=str(settings.project_root / '.cache' / 'models'),
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.model.max_seq_length = settings.embedding.max_seq_length
        dimension_method = getattr(self.model, 'get_embedding_dimension', None)
        self.dimension = (dimension_method() if dimension_method else self.model.get_sentence_embedding_dimension())
        self.signature = {
            'model_id': settings.embedding.model_id,
            'revision': settings.embedding.revision,
            'max_seq_length': settings.embedding.max_seq_length,
            'normalize_embeddings': True,
            'dimension': self.dimension,
            'text_version': settings.text.version,
            'max_tokens': settings.text.max_tokens,
            'tokenizer_version': 'casefold-before-filter-v2',
        }

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        result = self.model.encode(texts, batch_size=self.settings.embedding.batch_size,
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)
        result = np.asarray(result, dtype=np.float32)
        if result.shape != (len(texts), self.dimension) or not np.isfinite(result).all():
            raise ValueError('编码器输出形状或数值不正确')
        return result

    def encode_queries(self, queries: list[Query]) -> np.ndarray:
        for query in queries:
            validate_query(query)
        return self._encode([' '.join(query_tokens(q.text, version=self.settings.text.version,
                                                  max_tokens=self.settings.text.max_tokens)) for q in queries])

    def encode_products(self, products: list[Product]) -> np.ndarray:
        return self._encode([product_text(p, version=self.settings.text.version,
                                         max_tokens=self.settings.text.max_tokens) for p in products])
