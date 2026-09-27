"""组装一个官方数据划分，检查重复冲突和商品关联。"""

from dataclasses import dataclass
from typing import Iterable, Mapping

from product_search.schemas import Judgment, Product, Query
from product_search.data.normalize import (
    judgment_from_row,
    product_from_row,
    query_from_row,
)


@dataclass
class Dataset:
    """组装后的数据，以及它所属的官方划分。"""

    source_split: str
    queries: list[Query]
    products: list[Product]
    judgments: list[Judgment]


def build_dataset(
    product_rows: Iterable[Mapping[str, object]],
    example_rows: Iterable[Mapping[str, object]],
    *,
    source_split: str,
) -> Dataset:
    """转换原始行，检查关联，返回去重后的数据集。"""

    if source_split not in {"train", "test"}:
        raise ValueError("source_split 必须是 train 或 test")

    products_by_key: dict[tuple[str, str], Product] = {}
    queries_by_key: dict[tuple[str, str], Query] = {}
    judgments_by_key: dict[tuple[str, str, str], Judgment] = {}

    # 先建立商品索引，后面才能检查标注引用的商品是否存在。
    for row in product_rows:
        product = product_from_row(row)
        key = (product.locale, product.product_id)

        if key in products_by_key:
            if products_by_key[key] != product:
                raise ValueError(f"同一商品存在不同资料：{key}")
        else:
            products_by_key[key] = product

    # 再处理查询和相关性标注。
    for row in example_rows:
        if row["split"] != source_split:
            raise ValueError(
                f"数据划分不一致：预期 {source_split}，"
                f"实际 {row['split']}"
            )

        query = query_from_row(row)
        judgment = judgment_from_row(row)

        product_key = (
            judgment.locale,
            judgment.product_id,
        )

        if product_key not in products_by_key:
            raise ValueError(
                f"标注引用了不存在的商品：{product_key}"
            )

        query_key = (query.locale, query.query_id)

        if query_key in queries_by_key:
            if queries_by_key[query_key] != query:
                raise ValueError(
                    f"同一查询 ID 对应不同文本：{query_key}"
                )
        else:
            queries_by_key[query_key] = query

        judgment_key = (
            judgment.locale,
            judgment.query_id,
            judgment.product_id,
        )

        if judgment_key in judgments_by_key:
            if judgments_by_key[judgment_key] != judgment:
                raise ValueError(
                    f"同一查询与商品存在冲突标签：{judgment_key}"
                )
        else:
            judgments_by_key[judgment_key] = judgment

    if not judgments_by_key:
        raise ValueError("数据集没有有效标注")

    return Dataset(
        source_split=source_split,
        queries=list(queries_by_key.values()),
        products=list(products_by_key.values()),
        judgments=list(judgments_by_key.values()),
    )