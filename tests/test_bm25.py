"""BM25 数值、SQL 边界、持久化及生命周期回归。"""

import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from product_search.retrieval.bm25 import BM25Index
from product_search.schemas import Product, Query


VERSION = "title-brand-bullet-v1"


def build_index(path, products=None, **kwargs):
    if products is None:
        products = [Product("a", "us", "red red shoe"), Product("b", "us", "blue")]
    return BM25Index.build(products, path, text_version=VERSION, max_tokens=2000, **kwargs)


@pytest.fixture
def index_path(tmp_path):
    path = tmp_path / "bm25.sqlite"
    build_index(path).close()
    return path


def change_metadata(path, name, value):
    # sqlite Connection 的 with 只提交/回滚，不关闭连接；Windows 上需显式关闭文件句柄。
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute("UPDATE metadata SET value = ? WHERE name = ?", (json.dumps(value), name))


def test_scores_match_formula_and_round_trip(index_path):
    with BM25Index.load(index_path) as index:
        query = Query("q", "RED red", "us")
        hits = index.search(query, 10)
        expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
        assert len(hits) == 1
        assert hits[0].score == pytest.approx(expected)
        assert hits[0].product_key == ("us", "a")
        scores = index.score_pairs(query, [("us", "b"), ("us", "a")])
        assert list(scores) == [("us", "b"), ("us", "a")]
        assert scores == {("us", "b"): 0.0, ("us", "a"): pytest.approx(expected)}


@pytest.mark.parametrize("lower_limits", [False, True])
def test_large_candidates_and_search_respect_sql_limits(tmp_path, lower_limits):
    products = [Product(f"{i:04}", "us", "red") for i in range(1205)]
    keys = [(p.locale, p.product_id) for p in products]
    with build_index(tmp_path / "large.sqlite", products) as index:
        if lower_limits:
            index._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 32)
            index._conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 32)
        query = Query("q", "red", "us")
        scores = index.score_pairs(query, reversed(keys))
        assert list(scores) == list(reversed(keys))
        assert all(value > 0 for value in scores.values())
        hits = index.search(query, 1300)
        assert [hit.product_key for hit in hits] == keys
        assert [hit.rank for hit in hits] == list(range(1, 1206))
        assert hits[0].score == pytest.approx(scores[keys[0]])
        with pytest.raises(KeyError):
            index.score_pairs(query, keys + [("us", "missing")])


def test_many_query_terms_respect_sql_limits(tmp_path):
    title = " ".join(f"word{i}" for i in range(90))
    path = tmp_path / "terms.sqlite"
    with build_index(path, [Product("a", "us", title)]) as index:
        query = Query("q", title, "us")
        expected = index.search(query, 1)[0].score
    # 使用新连接，避免已缓存的 SQL 绕过调低后的 prepare 限额。
    with BM25Index.load(path) as index:
        index._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 16)
        assert index.search(query, 1)[0].score == pytest.approx(expected)
        assert index.score_pairs(query, [("us", "a")])["us", "a"] == pytest.approx(expected)


@pytest.mark.parametrize("name,value", [
    ("format_version", 1), ("text_version", []), ("text_version", "unknown"),
    ("max_tokens", True), ("max_tokens", 1.5), ("max_tokens", "2"), ("max_tokens", 0),
    ("k1", True), ("k1", "1.2"), ("k1", None), ("k1", float("nan")),
    ("b", False), ("b", "0.75"), ("b", float("inf")),
    ("document_count", True), ("document_count", "2"), ("document_count", 2.5),
    ("document_count", 3),
    ("average_document_length", True), ("average_document_length", "2"),
    ("average_document_length", float("nan")), ("average_document_length", float("inf")),
    ("average_document_length", 3),
    ("keys_sha256", None), ("keys_sha256", "g" * 64), ("keys_sha256", "0" * 64),
])
def test_load_rejects_invalid_metadata(index_path, name, value):
    change_metadata(index_path, name, value)
    with pytest.raises(ValueError, match="BM25"):
        BM25Index.load(index_path).close()


@pytest.mark.parametrize("sql", [
    "UPDATE documents SET product_id = 'changed' WHERE doc_id = 0",
    "UPDATE documents SET doc_length = 9 WHERE doc_id = 0",
    "UPDATE postings SET term_frequency = 2.5 WHERE term = 'red'",
    "DELETE FROM term_stats WHERE term = 'red'",
    "UPDATE term_stats SET idf = -1 WHERE term = 'red'",
    "UPDATE term_stats SET document_frequency = 2 WHERE term = 'red'",
    "INSERT INTO postings VALUES ('orphan', 999, 1)",
    "DELETE FROM metadata WHERE name = 'max_tokens'",
    "UPDATE metadata SET value = '{' WHERE name = 'b'",
    "DROP TABLE postings",
])
def test_load_rejects_damaged_index(index_path, sql):
    with sqlite3.connect(index_path) as conn:
        conn.execute(sql)
    with pytest.raises(ValueError, match="BM25"):
        BM25Index.load(index_path).close()


def test_load_rejects_non_sqlite_file(tmp_path):
    path = tmp_path / "broken.sqlite"
    path.write_bytes(b"broken database")
    with pytest.raises(ValueError, match="BM25"):
        BM25Index.load(path)


def test_build_does_not_overwrite_a_racing_publisher(tmp_path):
    path = tmp_path / "race.sqlite"
    other_contents = b"another publisher owns this path"

    def catalog():
        path.write_bytes(other_contents)
        yield Product("a", "us", "red")

    with pytest.raises(FileExistsError):
        build_index(path, catalog()).close()
    assert path.read_bytes() == other_contents
    assert list(tmp_path.iterdir()) == [path]


def test_build_validates_stage_before_publication(tmp_path, monkeypatch):
    path = tmp_path / "publish.sqlite"
    original_load = BM25Index.load
    checked_stages = []

    def reject_bad_stage(cls, source):
        source = Path(source)
        if source != path:
            checked_stages.append(source)
            assert not path.exists()
            change_metadata(source, "keys_sha256", "0" * 64)
        return original_load(source)

    monkeypatch.setattr(BM25Index, "load", classmethod(reject_bad_stage))
    with pytest.raises(ValueError, match="BM25"):
        build_index(path).close()
    assert checked_stages
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("key,error,match", [
    (["us", "a"], TypeError, "product_key"),
    (("us", []), ValueError, "product_id"),
    (([], "a"), ValueError, "locale"),
])
def test_pair_keys_are_validated_before_hashing(index_path, key, error, match):
    with BM25Index.load(index_path) as index:
        with pytest.raises(error, match=match):
            index.score_pairs(Query("q", "red", "us"), [key])


@pytest.mark.parametrize("operation", ["search", "empty_search", "pairs", "empty_pairs", "enter"])
def test_closed_index_raises_clear_error(index_path, operation):
    index = BM25Index.load(index_path)
    index.close()
    index.close()
    with pytest.raises(RuntimeError, match="关闭|closed"):
        if operation == "enter":
            index.__enter__()
        elif operation in ("search", "empty_search"):
            index.search(Query("q", "!!!" if operation == "empty_search" else "red", "us"), 1)
        else:
            index.score_pairs(Query("q", "red", "us"), [] if operation == "empty_pairs" else [("us", "a")])


def test_empty_tokens_locales_and_key_validation(tmp_path):
    products = [Product("a", "es", "red"), Product("a", "us", "!!!")]
    with build_index(tmp_path / "empty.sqlite", products) as index:
        query = Query("q", "red", "us")
        assert index.search(query, 2) == []
        assert index.score_pairs(query, [("us", "a")]) == {("us", "a"): 0.0}
        assert index.score_pairs(query, []) == {}
        with pytest.raises(ValueError, match="重复"):
            index.score_pairs(query, [("us", "a"), ("us", "a")])
        with pytest.raises(ValueError, match="市场"):
            index.score_pairs(query, [("es", "a")])
        with pytest.raises(KeyError):
            index.score_pairs(Query("q", "!!!", "us"), [("us", "missing")])


@pytest.mark.parametrize("products", [[], [Product("a", "us", "x")] * 2,
    [Product("b", "us", "x"), Product("a", "us", "x")]])
def test_failed_build_removes_staging_files(tmp_path, products):
    with pytest.raises(ValueError):
        build_index(tmp_path / "bad.sqlite", products)
    assert list(tmp_path.iterdir()) == []


def test_successful_build_cleans_stage_and_load_is_read_only(tmp_path):
    path = tmp_path / "index.sqlite"
    with build_index(path) as index:
        assert list(tmp_path.iterdir()) == [path]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            index._conn.execute("DELETE FROM documents")


def test_unsupported_atomic_publication_leaves_no_partial_index(tmp_path, monkeypatch):
    def unavailable_link(source, destination):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(os, "link", unavailable_link)
    with pytest.raises(OSError, match="hard links"):
        build_index(tmp_path / "index.sqlite")
    assert list(tmp_path.iterdir()) == []


def test_all_documents_can_have_zero_tokens(tmp_path):
    with build_index(tmp_path / "empty.sqlite", [Product("a", "us", "!!!")]) as index:
        assert index.average_document_length == 1.0
        assert index.search(Query("q", "anything", "us"), 1) == []


@pytest.mark.parametrize("k1,multiplier", [(1e308, 2 / 1.45), (1e-308, 1.0)])
def test_finite_extreme_k1_produces_finite_scores(tmp_path, k1, multiplier):
    products = [Product("a", "us", "red red"), Product("b", "us", "blue"),
                Product("c", "us", "green"), Product("d", "us", "white")]
    with build_index(tmp_path / "extreme.sqlite", products, k1=k1) as index:
        score = index.search(Query("q", "red", "us"), 1)[0].score
        assert math.isfinite(score)
        assert score == pytest.approx(math.log1p(3.5 / 1.5) * multiplier)
