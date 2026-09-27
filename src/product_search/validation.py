"""检查项目内部数据是否合法，不修改输入对象。"""

import math

from product_search.schemas import (
    Judgment,
    Product,
    Query,
    SearchResult,
)


ALLOWED_LOCALES = {"us", "es", "jp"}
ALLOWED_LABELS = {"E", "S", "C", "I"}


def _check_text(
    value: str,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    """检查字符串类型，以及必填字符串是否有实际内容。"""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} 必须是字符串")

    if not allow_empty and not value.strip():
        raise ValueError(f"{field_name} 不能为空或只有空白字符")


def _check_locale(locale: str) -> None:
    """检查所属市场。"""

    _check_text(locale, "locale")

    if locale not in ALLOWED_LOCALES:
        raise ValueError("locale 必须是 us、es 或 jp")


def validate_query(query: Query) -> None:
    """检查查询。"""

    _check_text(query.query_id, "query_id")
    _check_text(query.text, "text")
    _check_locale(query.locale)


def validate_product(product: Product) -> None:
    """检查商品资料。"""

    _check_text(product.product_id, "product_id")
    _check_locale(product.locale)
    _check_text(product.title, "title")

    _check_text(
        product.description,
        "description",
        allow_empty=True,
    )
    _check_text(
        product.bullet_point,
        "bullet_point",
        allow_empty=True,
    )
    _check_text(product.brand, "brand", allow_empty=True)
    _check_text(product.color, "color", allow_empty=True)


def validate_judgment(judgment: Judgment) -> None:
    """检查人工相关性标注。"""

    _check_text(judgment.query_id, "query_id")
    _check_text(judgment.product_id, "product_id")
    _check_locale(judgment.locale)
    _check_text(judgment.esci_label, "esci_label")

    if judgment.esci_label not in ALLOWED_LABELS:
        raise ValueError("esci_label 必须是 E、S、C 或 I")


def validate_search_result(result: SearchResult) -> None:
    """检查单条搜索结果。"""

    _check_text(result.query_id, "query_id")
    _check_text(result.product_id, "product_id")
    _check_locale(result.locale)

    if type(result.rank) is not int:
        raise TypeError("rank 必须是整数")

    if result.rank < 1:
        raise ValueError("rank 必须从 1 开始")

    if not isinstance(result.score, float):
        raise TypeError("score 必须是浮点数，例如 8.0")

    if not math.isfinite(result.score):
        raise ValueError("score 不能是 NaN 或无穷大")