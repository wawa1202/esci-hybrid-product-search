"""SQLite 存储：显式事务、重复检查和逐条读取。"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from product_search.contracts import ProductKey, QueryKey
from product_search.schemas import Judgment, Product, Query
from product_search.validation import (
    validate_judgment,
    validate_product,
    validate_query,
)


class DataConflictError(ValueError):
    """同一身份对应不同内容，或查询划分存在冲突。"""


@dataclass(frozen=True)
class StoredQuery:
    """查询本身，以及用于训练和评估的数据划分信息。"""

    query: Query
    normalized_group: str
    source_split: str
    partition: str


SCHEMAS = {
    "catalog": (
        """
        CREATE TABLE products (
            locale TEXT NOT NULL
                CHECK(locale IN ('us', 'es', 'jp')),
            product_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            bullet_point TEXT NOT NULL,
            brand TEXT NOT NULL,
            color TEXT NOT NULL,
            PRIMARY KEY (locale, product_id)
        )
        """,
    ),
    "labels": (
        """
        CREATE TABLE queries (
            locale TEXT NOT NULL
                CHECK(locale IN ('us', 'es', 'jp')),
            query_id TEXT NOT NULL,
            text TEXT NOT NULL,
            normalized_group TEXT NOT NULL,
            source_split TEXT NOT NULL,
            partition TEXT NOT NULL,
            PRIMARY KEY (locale, query_id),
            CHECK (
                (
                    source_split = 'train'
                    AND partition IN ('train', 'validation')
                )
                OR (
                    source_split = 'test'
                    AND partition = 'test'
                )
            )
        )
        """,
        """
        CREATE TABLE judgments (
            locale TEXT NOT NULL,
            query_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            esci_label TEXT NOT NULL
                CHECK(esci_label IN ('E', 'S', 'C', 'I')),
            PRIMARY KEY (locale, query_id, product_id),
            FOREIGN KEY (locale, query_id)
                REFERENCES queries(locale, query_id)
        )
        """,
        """
        CREATE INDEX queries_group_idx
        ON queries(locale, normalized_group, partition)
        """,
        """
        CREATE INDEX queries_partition_idx
        ON queries(partition, locale, query_id)
        """,
    ),
}


# 表名和列名只允许来自代码内的固定定义。
# 每个定义中的数字表示：前几列共同构成主键。
TABLES = {
    "products": (
        (
            "locale",
            "product_id",
            "title",
            "description",
            "bullet_point",
            "brand",
            "color",
        ),
        2,
    ),
    "queries": (
        (
            "locale",
            "query_id",
            "text",
            "normalized_group",
            "source_split",
            "partition",
        ),
        2,
    ),
    "judgments": (
        (
            "locale",
            "query_id",
            "product_id",
            "esci_label",
        ),
        3,
    ),
}


@contextmanager
def open_database(
    path: str | Path,
    *,
    read_only: bool = False,
    create: bool = False,
) -> Iterator[sqlite3.Connection]:
    """默认只打开现有文件；create=True 才允许创建数据库。"""
    if read_only and create:
        raise ValueError(
            "read_only 和 create 不能同时为 True"
        )

    mode = "ro" if read_only else (
        "rwc" if create else "rw"
    )

    uri = Path(path).resolve().as_uri() + f"?mode={mode}"

    conn = sqlite3.connect(
        uri,
        uri=True,
        timeout=30.0,
        isolation_level=None,
        autocommit=sqlite3.LEGACY_TRANSACTION_CONTROL,
    )

    try:
        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA foreign_keys = ON")

        enabled = conn.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]

        if enabled != 1:
            raise RuntimeError(
                "SQLite 外键约束未启用"
            )

        yield conn

    finally:
        try:
            if conn.in_transaction:
                conn.rollback()
        finally:
            conn.close()


@contextmanager
def transaction(
    conn: sqlite3.Connection,
) -> Iterator[None]:
    """正常退出提交；发生异常回滚；不支持嵌套事务。"""
    if conn.in_transaction:
        raise RuntimeError(
            "不支持嵌套事务"
        )

    conn.execute("BEGIN IMMEDIATE")

    try:
        yield
        conn.commit()

    except BaseException:
        conn.rollback()
        raise


def init_schema(
    conn: sqlite3.Connection,
    kind: str,
) -> None:
    """仅初始化空数据库，不迁移或覆盖已有表。"""
    if kind not in SCHEMAS:
        raise ValueError(
            "kind 必须是 catalog 或 labels"
        )

    with transaction(conn):
        existing = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchone()

        if existing is not None:
            raise ValueError(
                "数据库已有表；请直接使用，不要重复初始化"
            )

        for statement in SCHEMAS[kind]:
            conn.execute(statement)


def _require_transaction(
    conn: sqlite3.Connection,
) -> None:
    if not conn.in_transaction:
        raise RuntimeError(
            "写入必须放在 with transaction(conn): 内"
        )


def _insert_record(
    conn: sqlite3.Connection,
    table: str,
    values: tuple,
) -> bool:
    """新记录返回 True；完全相同返回 False；主键冲突则报错。"""
    _require_transaction(conn)

    columns, key_size = TABLES[table]

    names = ", ".join(columns)

    where = " AND ".join(
        f"{name} = ?"
        for name in columns[:key_size]
    )

    key = values[:key_size]

    old = conn.execute(
        f"SELECT {names} FROM {table} WHERE {where}",
        key,
    ).fetchone()

    if old is not None:
        if tuple(old) != values:
            raise DataConflictError(
                f"{table} 主键 {key} 对应不同内容"
            )

        return False

    placeholders = ", ".join(
        "?" for _ in columns
    )

    conn.execute(
        f"""
        INSERT INTO {table} ({names})
        VALUES ({placeholders})
        """,
        values,
    )

    return True


def insert_product(
    conn: sqlite3.Connection,
    product: Product,
) -> bool:
    validate_product(product)

    return _insert_record(
        conn,
        "products",
        (
            product.locale,
            product.product_id,
            product.title,
            product.description,
            product.bullet_point,
            product.brand,
            product.color,
        ),
    )


def insert_query(
    conn: sqlite3.Connection,
    query: Query,
    *,
    source_split: str,
    partition: str,
) -> bool:
    _require_transaction(conn)
    validate_query(query)

    allowed = {
        ("train", "train"),
        ("train", "validation"),
        ("test", "test"),
    }

    if (source_split, partition) not in allowed:
        raise ValueError(
            "官方 train 只能分到 train/validation；"
            "官方 test 只能分到 test"
        )

    group = " ".join(
        query.text.casefold().split()
    )

    other = conn.execute(
        """
        SELECT query_id
        FROM queries
        WHERE locale = ?
          AND normalized_group = ?
          AND partition <> ?
        LIMIT 1
        """,
        (
            query.locale,
            group,
            partition,
        ),
    ).fetchone()

    if other is not None:
        raise DataConflictError(
            "同一市场的查询文本组跨分区："
            f"{query.locale}, {group!r}"
        )

    return _insert_record(
        conn,
        "queries",
        (
            query.locale,
            query.query_id,
            query.text,
            group,
            source_split,
            partition,
        ),
    )


def insert_judgment(
    conn: sqlite3.Connection,
    judgment: Judgment,
) -> bool:
    """查询关联由外键检查；跨数据库的商品关联留给数据审计。"""
    validate_judgment(judgment)

    return _insert_record(
        conn,
        "judgments",
        (
            judgment.locale,
            judgment.query_id,
            judgment.product_id,
            judgment.esci_label,
        ),
    )


def get_product(
    conn: sqlite3.Connection,
    key: ProductKey,
) -> Product | None:
    names = ", ".join(
        TABLES["products"][0]
    )

    row = conn.execute(
        f"""
        SELECT {names}
        FROM products
        WHERE locale = ? AND product_id = ?
        """,
        key,
    ).fetchone()

    if row is None:
        return None

    return Product(**dict(row))


def iter_products(
    conn: sqlite3.Connection,
) -> Iterator[Product]:
    """按商品键稳定排序，逐条读取，避免一次装入全部商品。"""
    names = ", ".join(
        TABLES["products"][0]
    )

    cursor = conn.execute(
        f"""
        SELECT {names}
        FROM products
        ORDER BY locale, product_id
        """
    )

    try:
        for row in cursor:
            yield Product(**dict(row))

    finally:
        cursor.close()


def iter_queries(
    conn: sqlite3.Connection,
    *,
    partition: str,
) -> Iterator[StoredQuery]:
    if partition not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "partition 必须是 train、validation 或 test"
        )

    cursor = conn.execute(
        """
        SELECT *
        FROM queries
        WHERE partition = ?
        ORDER BY locale, query_id
        """,
        (partition,),
    )

    try:
        for row in cursor:
            yield StoredQuery(
                query=Query(
                    query_id=row["query_id"],
                    text=row["text"],
                    locale=row["locale"],
                ),
                normalized_group=row["normalized_group"],
                source_split=row["source_split"],
                partition=row["partition"],
            )

    finally:
        cursor.close()


def iter_judgments(
    conn: sqlite3.Connection,
    key: QueryKey,
) -> Iterator[Judgment]:
    cursor = conn.execute(
        """
        SELECT locale, query_id, product_id, esci_label
        FROM judgments
        WHERE locale = ? AND query_id = ?
        ORDER BY product_id
        """,
        key,
    )

    try:
        for row in cursor:
            yield Judgment(**dict(row))

    finally:
        cursor.close()