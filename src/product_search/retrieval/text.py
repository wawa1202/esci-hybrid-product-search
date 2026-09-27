"""统一商品搜索的查询分词和商品文本策略。"""

from collections.abc import Iterable

from product_search.schemas import Product


SUPPORTED_VERSIONS = {
    "title-brand-bullet-v1",
}


def _check_version(version: str) -> None:
    """检查文本策略版本。"""
    if (
        type(version) is not str
        or version not in SUPPORTED_VERSIONS
    ):
        raise ValueError(
            "当前仅支持 text version="
            "title-brand-bullet-v1"
        )


def _clean_text(value: str) -> str:
    """大小写归一化、标点转空格、合并连续空格。"""
    if type(value) is not str:
        raise TypeError("文本必须是字符串")

    characters = []

    # casefold 可能把一个字符展开为字母和组合标记（例如 İ）。
    # 先展开再过滤，才能保证再次清洗得到相同的 token。
    for char in value.casefold():
        if char.isalnum():
            characters.append(char)
        else:
            characters.append(" ")

    return " ".join(
        "".join(characters).split()
    )


def tokenize(
    text: str,
    *,
    version: str,
) -> tuple[str, ...]:
    """
    按固定版本清洗并分词。

    当前策略：
    1. 使用 Unicode casefold；
    2. 字母和数字保留；
    3. 其他字符转为空格；
    4. 合并连续空格；
    5. 不删除停用词；
    6. 不做词干化。
    """
    _check_version(version)

    cleaned = _clean_text(text)

    if not cleaned:
        return ()

    return tuple(cleaned.split())


def _limit_tokens(
    tokens: Iterable[str],
    max_tokens: int | None,
) -> tuple[str, ...]:
    """对 token 序列做可选的前缀截断。"""
    if max_tokens is None:
        return tuple(tokens)

    if (
        type(max_tokens) is not int
        or max_tokens < 1
    ):
        raise ValueError(
            "max_tokens 必须是正整数或 None"
        )

    return tuple(tokens)[:max_tokens]


def query_tokens(
    text: str,
    *,
    version: str,
    max_tokens: int | None = None,
) -> tuple[str, ...]:
    """返回查询 token。"""
    tokens = tokenize(
        text,
        version=version,
    )

    return _limit_tokens(
        tokens,
        max_tokens,
    )


def product_text(
    product: Product,
    *,
    version: str,
    max_tokens: int | None = None,
) -> str:
    """
    按 title、brand、bullet_point 顺序
    构造商品检索文本。
    """
    if not isinstance(product, Product):
        raise TypeError(
            "product 必须是 Product"
        )

    tokens: list[str] = []

    for field in (
        product.title,
        product.brand,
        product.bullet_point,
    ):
        tokens.extend(
            tokenize(
                field,
                version=version,
            )
        )

    limited_tokens = _limit_tokens(
        tokens,
        max_tokens,
    )

    return " ".join(limited_tokens)
