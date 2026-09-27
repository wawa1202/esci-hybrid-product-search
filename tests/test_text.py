"""文本清洗与商品/查询分词的一致性回归。"""

import pytest

from product_search.retrieval.text import _clean_text, product_text, query_tokens, tokenize
from product_search.schemas import Product


VERSION = "title-brand-bullet-v1"


@pytest.mark.parametrize("text", ["İ", "İSTANBUL", "Straße İ CAFÉ", "ΐ ᾷ ﬃ 中文１２"])
def test_clean_text_is_idempotent_after_unicode_casefold(text):
    cleaned = _clean_text(text)
    assert _clean_text(cleaned) == cleaned
    assert all(char.isalnum() or char == " " for char in cleaned)


def test_product_and_query_tokens_agree_after_truncation():
    product = Product("a", "us", "İSTANBUL", brand="STRASSE", bullet_point="中文")
    text = product_text(product, version=VERSION, max_tokens=2)
    assert query_tokens(text, version=VERSION) == query_tokens(
        product.title, version=VERSION, max_tokens=2
    ) == ("i", "stanbul")


def test_text_policy_keeps_field_order_and_unicode():
    product = Product("a", "us", "Straße, RED!", brand="ＡＢ", bullet_point="中文 42")
    assert product_text(product, version=VERSION) == "strasse red ａｂ 中文 42"
    assert tokenize(" \t!?", version=VERSION) == ()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_bad_token_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="max_tokens"):
        query_tokens("hello", version=VERSION, max_tokens=limit)


def test_bad_text_and_version_are_rejected():
    with pytest.raises(TypeError):
        tokenize(None, version=VERSION)
    with pytest.raises(ValueError):
        tokenize("hello", version="unknown")
