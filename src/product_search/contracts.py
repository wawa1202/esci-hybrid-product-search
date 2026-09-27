"""定义召回、融合和排序模块之间传递的数据，不实现检索算法。"""

import math
from dataclasses import dataclass
from typing import Literal

from product_search.schemas import Product, Query
from product_search.validation import ALLOWED_LOCALES


# 两种键的固定顺序均为：市场、ID。
# type 别名语法要求 Python 3.12 或更新版本。
type QueryKey = tuple[str, str]
type ProductKey = tuple[str, str]

# 当前项目只有两路基础召回。
type RetrievalSource = Literal["bm25", "dense"]


def _check_key(value: object, name: str) -> None:
    """键的固定顺序是 (locale, id)。"""
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(
            f"{name} 必须是包含两个元素的元组"
        )

    locale, identifier = value

    if type(locale) is not str or locale not in ALLOWED_LOCALES:
        raise ValueError(
            f"{name} 的 locale 必须是 us、es 或 jp"
        )

    if type(identifier) is not str or not identifier.strip():
        raise ValueError(
            f"{name} 的 id 必须是非空字符串"
        )


def _check_rank(value: object, name: str) -> None:
    """排名统一从 1 开始。"""
    if type(value) is not int:
        raise TypeError(
            f"{name} 必须是整数"
        )

    if value < 1:
        raise ValueError(
            f"{name} 必须从 1 开始"
        )


def _check_score(value: object, name: str) -> None:
    """分数必须是有限的 Python 浮点数。"""
    if type(value) is not float:
        raise TypeError(
            f"{name} 必须是 Python float，例如 8.0"
        )

    if not math.isfinite(value):
        raise ValueError(
            f"{name} 不能是 NaN 或无穷大"
        )


def get_query_key(query: Query) -> QueryKey:
    """提取查询身份，不复制文本，不重新清洗数据。"""
    key = (query.locale, query.query_id)

    _check_key(key, "query_key")

    return key


def get_product_key(product: Product) -> ProductKey:
    """提取商品身份，不复制标题、描述等资料。"""
    key = (product.locale, product.product_id)

    _check_key(key, "product_key")

    return key


@dataclass(frozen=True)
class Hit:
    """一个商品在某个查询的一路召回结果中的记录。"""

    query_key: QueryKey
    product_key: ProductKey
    source: RetrievalSource
    rank: int
    score: float

    def __post_init__(self) -> None:
        _check_key(self.query_key, "query_key")
        _check_key(self.product_key, "product_key")

        if self.query_key[0] != self.product_key[0]:
            raise ValueError(
                "查询与商品必须属于同一市场"
            )

        if (
            type(self.source) is not str
            or self.source not in {"bm25", "dense"}
        ):
            raise ValueError(
                "source 只能是 bm25 或 dense"
            )

        _check_rank(self.rank, "rank")
        _check_score(self.score, "score")


@dataclass(frozen=True)
class Candidate:
    """同一查询下，一个商品在融合后的记录及其各路召回依据。"""

    query_key: QueryKey
    product_key: ProductKey
    hits: tuple[Hit, ...]
    rrf_rank: int
    rrf_score: float

    def __post_init__(self) -> None:
        _check_key(self.query_key, "query_key")
        _check_key(self.product_key, "product_key")

        if self.query_key[0] != self.product_key[0]:
            raise ValueError(
                "查询与商品必须属于同一市场"
            )

        if type(self.hits) is not tuple or not self.hits:
            raise ValueError(
                "hits 必须是非空元组"
            )

        seen_sources: set[str] = set()

        for hit in self.hits:
            if not isinstance(hit, Hit):
                raise TypeError(
                    "hits 中的每个元素必须是 Hit"
                )

            if hit.query_key != self.query_key:
                raise ValueError(
                    "Candidate 不能混入其他查询的召回结果"
                )

            if hit.product_key != self.product_key:
                raise ValueError(
                    "Candidate 不能混入其他商品的召回结果"
                )

            if hit.source in seen_sources:
                raise ValueError(
                    "同一个商品在同一路召回中只能出现一次"
                )

            seen_sources.add(hit.source)

        _check_rank(self.rrf_rank, "rrf_rank")
        _check_score(self.rrf_score, "rrf_score")

        if self.rrf_score <= 0:
            raise ValueError(
                "至少命中一路的候选，其 RRF 分数必须大于 0"
            )

    def get_hit(self, source: RetrievalSource) -> Hit | None:
        """获取指定召回路的记录；该路未召回此商品时返回 None。"""
        if (
            type(source) is not str
            or source not in {"bm25", "dense"}
        ):
            raise ValueError(
                "source 只能是 bm25 或 dense"
            )

        for hit in self.hits:
            if hit.source == source:
                return hit

        return None