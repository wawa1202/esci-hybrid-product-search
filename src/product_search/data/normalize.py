"""把原始数据行转换为项目对象，并执行字段校验。"""

import math
from numbers import Integral
from typing import Mapping

from product_search.schemas import Judgment, Product, Query
from product_search.validation import (
    validate_judgment,
    validate_product,
    validate_query,
)


def normalize_text(value: object) -> str:
    """统一常见缺失文本和空白，不修改大小写或标点。"""

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    if not isinstance(value, str):
        raise TypeError("文本必须是字符串、None 或浮点 NaN")

    return " ".join(value.split())


def normalize_id(value: object) -> str:
    """将字符串或整数 ID 统一为非空字符串。"""

    if isinstance(value, bool):
        raise TypeError("ID 不能是布尔值")

    if isinstance(value, Integral):
        return str(value)

    if not isinstance(value, str):
        raise TypeError("ID 必须是字符串或整数")

    normalized = value.strip()

    if not normalized:
        raise ValueError("ID 不能为空")

    return normalized


def query_from_row(row: Mapping[str, object]) -> Query:
    """从原始样本行提取查询，不读取商品标签。"""

    query = Query(
        query_id=normalize_id(row["query_id"]),
        text=normalize_text(row["query"]),
        locale=normalize_text(row["product_locale"]),
    )

    validate_query(query)
    return query


def product_from_row(row: Mapping[str, object]) -> Product:
    """从原始商品行提取商品资料。"""

    product = Product(
        product_id=normalize_id(row["product_id"]),
        locale=normalize_text(row["product_locale"]),
        title=normalize_text(row["product_title"]),
        description=normalize_text(
            row.get("product_description")
        ),
        bullet_point=normalize_text(
            row.get("product_bullet_point")
        ),
        brand=normalize_text(row.get("product_brand")),
        color=normalize_text(row.get("product_color")),
    )

    validate_product(product)
    return product


def judgment_from_row(
    row: Mapping[str, object],
) -> Judgment:
    """从原始样本行提取人工相关性标注。"""

    judgment = Judgment(
        query_id=normalize_id(row["query_id"]),
        product_id=normalize_id(row["product_id"]),
        locale=normalize_text(row["product_locale"]),
        esci_label=normalize_text(row["esci_label"]),
    )

    validate_judgment(judgment)
    return judgment