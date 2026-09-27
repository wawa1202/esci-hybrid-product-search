"""在给定候选集内做排序，不读取人工相关性标签。"""

import math
from collections.abc import Iterable

from product_search.contracts import (
    Hit,
    ProductKey,
    get_query_key,
)
from product_search.retrieval.bm25 import BM25Index
from product_search.schemas import Query
from product_search.validation import validate_query


def _check_k(k: int | None) -> None:
    """None 返回全部候选；正整数返回前 k 个。"""
    if k is not None and (
        type(k) is not int or k < 1
    ):
        raise ValueError(
            "k 必须是正整数或 None"
        )


def _check_candidates(
    query: Query,
    candidates: Iterable[ProductKey],
) -> list[ProductKey]:
    """先检查键的结构，再检查市场与重复。"""
    validate_query(query)

    values = []
    seen = set()

    for key in candidates:
        if type(key) is not tuple or len(key) != 2:
            raise TypeError(
                "候选商品必须是 "
                "(locale, product_id) 元组"
            )

        locale, product_id = key

        if (
            type(locale) is not str
            or locale != query.locale
        ):
            raise ValueError(
                "查询与候选商品必须属于同一市场"
            )

        if (
            type(product_id) is not str
            or not product_id.strip()
        ):
            raise ValueError(
                "候选 product_id 必须是非空字符串"
            )

        if key in seen:
            raise ValueError(
                f"候选集包含重复商品：{key}"
            )

        seen.add(key)
        values.append(key)

    return values


def rank_bm25_candidates(
    index: BM25Index,
    query: Query,
    candidates: Iterable[ProductKey],
    *,
    k: int | None = None,
) -> list[Hit]:
    """
    在给定候选中做 BM25 排序。

    k=None 返回全部候选。
    零分商品保留，同分按商品键排序。
    """
    _check_k(k)

    keys = _check_candidates(
        query,
        candidates,
    )

    if not keys:
        return []

    scores = index.score_pairs(
        query,
        keys,
    )

    if set(scores) != set(keys):
        raise ValueError(
            "打分结果必须与候选商品集合完全一致"
        )

    for key, score in scores.items():
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
        ):
            raise ValueError(
                f"商品分数必须是有限数值：{key}"
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    if k is not None:
        ranked = ranked[:k]

    query_key = get_query_key(query)

    return [
        Hit(
            query_key=query_key,
            product_key=product_key,
            source="bm25",
            rank=rank,
            score=float(score),
        )
        for rank, (product_key, score)
        in enumerate(ranked, start=1)
    ]