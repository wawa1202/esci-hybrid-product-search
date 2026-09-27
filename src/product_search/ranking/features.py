"""与标签、身份和候选排名无关的逐商品特征。"""

import numpy as np

from product_search.contracts import ProductKey, get_product_key
from product_search.retrieval.text import query_tokens
from product_search.schemas import Product, Query
from product_search.settings import Settings
from product_search.validation import validate_product, validate_query


FEATURE_NAMES = (
    'bm25_score', 'dense_score', 'query_length', 'title_length',
    'overlap_count', 'overlap_ratio', 'brand_match', 'color_match',
    'bullet_length', 'bullet_overlap_count', 'bullet_overlap_ratio',
)


def build_features(
    query: Query,
    products: list[Product],
    *,
    bm25_scores: dict[ProductKey, float],
    dense_scores: dict[ProductKey, float],
    settings: Settings,
) -> np.ndarray:
    """按输入商品顺序返回特征，长度按截断后的 token 数计算。

    overlap 使用去重 token 交集，ratio 的分母为查询去重 token 数；
    brand/color 非空且全部 token 出现在查询中才匹配。商品键仅用于
    严格对齐分数，绝不作为特征，且不做依赖候选集合的归一化。
    """
    validate_query(query)
    keys = [get_product_key(product) for product in products]
    if len(set(keys)) != len(keys):
        raise ValueError('候选商品 keys 不能重复')
    if any(key[0] != query.locale for key in keys):
        raise ValueError('查询与商品必须属于同一市场')
    if set(bm25_scores) != set(keys) or set(dense_scores) != set(keys):
        raise ValueError('BM25、dense 与候选商品 keys 必须完全相同')

    def tokens(text: str) -> tuple[str, ...]:
        return query_tokens(text, version=settings.text.version,
                            max_tokens=settings.text.max_tokens)

    query_words = tokens(query.text)
    query_set = set(query_words)
    denominator = max(1, len(query_set))
    matrix = np.empty((len(products), len(FEATURE_NAMES)), dtype=np.float64)
    for row, (key, product) in enumerate(zip(keys, products, strict=True)):
        validate_product(product)
        title, bullet = tokens(product.title), tokens(product.bullet_point)
        brand, color = set(tokens(product.brand)), set(tokens(product.color))
        title_overlap = len(query_set.intersection(title))
        bullet_overlap = len(query_set.intersection(bullet))
        matrix[row] = (
            bm25_scores[key], dense_scores[key], len(query_words), len(title),
            title_overlap, title_overlap / denominator,
            bool(brand) and brand <= query_set, bool(color) and color <= query_set,
            len(bullet), bullet_overlap, bullet_overlap / denominator,
        )
    if not np.isfinite(matrix).all():
        raise ValueError('特征及检索分数必须为 finite 有限数值')
    return matrix


def _active_features(drop_features: tuple[str, ...]) -> tuple[str, ...]:
    """校验消融列，并保持保留特征的固定顺序。"""
    if isinstance(drop_features, str):
        raise ValueError('drop_features 必须为特征名序列')
    dropped = tuple(drop_features)
    if any(not isinstance(name, str) for name in dropped):
        raise ValueError('drop_features 必须为特征名序列')
    if len(set(dropped)) != len(dropped) or set(dropped) - set(FEATURE_NAMES):
        raise ValueError('drop_features 包含重复或未知特征')
    active = tuple(name for name in FEATURE_NAMES if name not in dropped)
    if not active:
        raise ValueError('不能删除全部特征')
    return active
