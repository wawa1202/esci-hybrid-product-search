"""可持久化的 BM25 倒排索引；不读取人工标签。"""

import hashlib
import json
import math
import os
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from tempfile import mkdtemp

from product_search.contracts import (
    Hit,
    ProductKey,
    get_product_key,
    get_query_key,
)
from product_search.schemas import Product, Query
from product_search.retrieval.text import (
    product_text,
    query_tokens,
)
from product_search.validation import (
    ALLOWED_LOCALES,
    validate_product,
    validate_query,
)


FORMAT_VERSION = "bm25-v1"
BUILD_BATCH_SIZE = 4096
QUERY_BATCH_SIZE = 400


def _check_positive_int(
    value: object,
    name: str,
) -> None:
    """检查正整数参数。"""
    if type(value) is not int or value < 1:
        raise ValueError(
            f"{name} 必须是正整数"
        )


def _check_parameters(
    k1: float,
    b: float,
) -> None:
    """检查 BM25 参数。"""
    if (
        type(k1) not in (int, float)
        or not math.isfinite(float(k1))
        or k1 <= 0
    ):
        raise ValueError(
            "k1 必须是有限正数"
        )

    if (
        type(b) not in (int, float)
        or not math.isfinite(float(b))
        or not 0 <= b <= 1
    ):
        raise ValueError(
            "b 必须是 0 到 1 之间的有限数"
        )


def _check_product_key(
    key: ProductKey,
) -> None:
    """检查商品键。"""
    if type(key) is not tuple or len(key) != 2:
        raise TypeError(
            "product_key 必须是 "
            "(locale, product_id) 元组"
        )

    locale, product_id = key

    if (
        type(locale) is not str
        or locale not in ALLOWED_LOCALES
    ):
        raise ValueError(
            "product_key 的 locale 不合法"
        )

    if (
        type(product_id) is not str
        or not product_id.strip()
    ):
        raise ValueError(
            "product_key 的 product_id 不能为空"
        )


def _create_schema(
    conn: sqlite3.Connection,
) -> None:
    """创建 BM25 索引数据库结构。"""
    conn.executescript(
        """
        CREATE TABLE metadata (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            doc_id INTEGER PRIMARY KEY,
            locale TEXT NOT NULL,
            product_id TEXT NOT NULL,
            doc_length INTEGER NOT NULL
                CHECK(doc_length >= 0),
            UNIQUE(locale, product_id)
        );

        CREATE TABLE postings (
            term TEXT NOT NULL,
            doc_id INTEGER NOT NULL,
            term_frequency INTEGER NOT NULL
                CHECK(term_frequency > 0),
            PRIMARY KEY(term, doc_id),
            FOREIGN KEY(doc_id)
                REFERENCES documents(doc_id)
        );

        CREATE TABLE term_stats (
            term TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL
                CHECK(document_frequency > 0),
            idf REAL NOT NULL
        );

        CREATE INDEX postings_doc_idx
        ON postings(doc_id);
        """
    )


class BM25Index:
    """
    使用 SQLite 保存倒排表。

    查询时只读取命中的 posting，
    不把整个倒排表加载到 Python 内存。
    """

    def __init__(
        self,
        path: Path,
        conn: sqlite3.Connection,
        metadata: dict[str, object],
    ):
        self.path = path
        self._conn = conn
        self._metadata = metadata

        self.text_version = str(
            metadata["text_version"]
        )
        self.max_tokens = int(
            metadata["max_tokens"]
        )
        self.k1 = float(metadata["k1"])
        self.b = float(metadata["b"])
        self.document_count = int(
            metadata["document_count"]
        )
        self.average_document_length = float(
            metadata["average_document_length"]
        )

    @classmethod
    def build(
        cls,
        catalog: Iterable[Product],
        path: str | Path,
        *,
        text_version: str,
        max_tokens: int,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "BM25Index":
        """
        从商品目录构建 BM25 索引。

        catalog 必须按：
            (locale, product_id)
        严格递增。
        """
        _check_positive_int(
            max_tokens,
            "max_tokens",
        )

        _check_parameters(k1, b)

        destination = Path(path).resolve()

        if destination.exists():
            raise FileExistsError(
                f"BM25 索引已存在，不覆盖：{destination}"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        stage = Path(
            mkdtemp(
                prefix=f".{destination.name}-pending-",
                dir=destination.parent,
            )
        )

        temporary = stage / destination.name
        conn = None

        def clean_stage():
            resolved = stage.resolve()
            if resolved.parent != destination.parent or not resolved.name.startswith(f".{destination.name}-pending-"):
                raise ValueError("BM25 暂存清理路径超出预期范围")
            if resolved.exists():
                shutil.rmtree(resolved)

        try:
            conn = sqlite3.connect(temporary)
            conn.execute(
                "PRAGMA foreign_keys = ON"
            )

            _create_schema(conn)

            conn.execute("BEGIN")

            previous_key = None
            document_count = 0
            total_length = 0
            keys_digest = hashlib.sha256()

            for product in catalog:
                validate_product(product)

                key = get_product_key(product)

                if (
                    previous_key is not None
                    and key <= previous_key
                ):
                    raise ValueError(
                        "catalog 必须按 "
                        "(locale, product_id) "
                        "严格递增且不能重复"
                    )

                previous_key = key

                text = product_text(
                    product,
                    version=text_version,
                    max_tokens=max_tokens,
                )

                tokens = query_tokens(
                    text,
                    version=text_version,
                )

                frequencies = Counter(tokens)
                doc_id = document_count

                conn.execute(
                    """
                    INSERT INTO documents(
                        doc_id,
                        locale,
                        product_id,
                        doc_length
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        key[0],
                        key[1],
                        len(tokens),
                    ),
                )

                conn.executemany(
                    """
                    INSERT INTO postings(
                        term,
                        doc_id,
                        term_frequency
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        (
                            term,
                            doc_id,
                            frequency,
                        )
                        for term, frequency
                        in frequencies.items()
                    ),
                )

                keys_digest.update(
                    (
                        f"{key[0]}\0{key[1]}\n"
                    ).encode("utf-8")
                )

                document_count += 1
                total_length += len(tokens)

                if (
                    document_count
                    % BUILD_BATCH_SIZE
                    == 0
                ):
                    conn.commit()
                    conn.execute("BEGIN")

            if document_count == 0:
                raise ValueError(
                    "catalog 不能为空"
                )

            conn.commit()

            average_length = (
                total_length / document_count
            )

            if average_length <= 0:
                average_length = 1.0

            conn.execute("BEGIN")

            term_rows = conn.execute(
                """
                SELECT term, COUNT(*)
                FROM postings
                GROUP BY term
                """
            )

            for term, document_frequency in term_rows:
                idf = math.log1p(
                    (
                        document_count
                        - document_frequency
                        + 0.5
                    )
                    / (
                        document_frequency
                        + 0.5
                    )
                )

                conn.execute(
                    """
                    INSERT INTO term_stats(
                        term,
                        document_frequency,
                        idf
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        term,
                        document_frequency,
                        idf,
                    ),
                )

            metadata = {
                "format_version": FORMAT_VERSION,
                "text_version": text_version,
                "max_tokens": max_tokens,
                "k1": float(k1),
                "b": float(b),
                "document_count": document_count,
                "average_document_length": (
                    average_length
                ),
                "keys_sha256": (
                    keys_digest.hexdigest()
                ),
            }

            conn.executemany(
                """
                INSERT INTO metadata(name, value)
                VALUES (?, ?)
                """,
                (
                    (
                        name,
                        json.dumps(
                            value,
                            ensure_ascii=False,
                        ),
                    )
                    for name, value
                    in metadata.items()
                ),
            )

            conn.commit()
            conn.close()
            conn = None

            # 完整验证通过后才发布。硬链接在同一文件系统内原子创建，
            # 目标已存在时会失败，避免 exists + replace 的覆盖竞态。
            with cls.load(temporary):
                pass
            os.link(temporary, destination)

            return cls.load(destination)

        except BaseException:
            if conn is not None:
                conn.rollback()
                conn.close()

            clean_stage()

            raise

        finally:
            if conn is None and stage.exists():
                clean_stage()

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "BM25Index":
        """以只读方式加载已经生成的索引。"""
        source = Path(path).resolve()

        if not source.is_file():
            raise FileNotFoundError(
                f"找不到 BM25 索引：{source}"
            )

        uri = source.as_uri() + "?mode=ro"

        conn = sqlite3.connect(
            uri,
            uri=True,
        )

        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            result = conn.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]

            if result != "ok":
                raise ValueError(
                    "BM25 SQLite 索引完整性检查失败"
                )

            rows = conn.execute(
                """
                SELECT name, value
                FROM metadata
                """
            ).fetchall()

            metadata = {
                row["name"]: json.loads(
                    row["value"]
                )
                for row in rows
            }

            required = {
                "format_version",
                "text_version",
                "max_tokens",
                "k1",
                "b",
                "document_count",
                "average_document_length",
                "keys_sha256",
            }

            if (
                set(metadata) != required
                or metadata["format_version"]
                != FORMAT_VERSION
            ):
                raise ValueError(
                    "BM25 索引格式或元数据不匹配"
                )

            # 不做强制类型转换：bool、数字字符串和小数不能冒充整数。
            _check_parameters(metadata["k1"], metadata["b"])
            _check_positive_int(metadata["max_tokens"], "max_tokens")
            _check_positive_int(metadata["document_count"], "document_count")
            query_tokens("", version=metadata["text_version"])

            average_length = metadata["average_document_length"]
            if (
                type(average_length) not in (int, float)
                or not math.isfinite(average_length)
                or average_length <= 0
            ):
                raise ValueError(
                    "BM25 索引文档统计无效"
                )

            digest = metadata["keys_sha256"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("BM25 keys_sha256 无效")

            cls._validate_contents(conn, metadata)

            return cls(
                source,
                conn,
                metadata,
            )

        except (sqlite3.DatabaseError, ValueError, TypeError, OverflowError) as exc:
            conn.close()
            raise ValueError(f"BM25 索引损坏或元数据无效：{exc}") from exc
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def _validate_contents(
        conn: sqlite3.Connection,
        metadata: dict[str, object],
    ) -> None:
        """逐行核对商品键与统计，避免把整个倒排表载入内存。"""
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("BM25 倒排表引用了不存在的文档")

        keys_digest = hashlib.sha256()
        document_count = 0
        total_length = 0
        previous_key = None
        rows = conn.execute(
            """
            SELECT doc_id, locale, product_id, doc_length
            FROM documents ORDER BY doc_id
            """
        )
        for row in rows:
            key = (row["locale"], row["product_id"])
            _check_product_key(key)
            length = row["doc_length"]
            if (
                row["doc_id"] != document_count
                or (previous_key is not None and key <= previous_key)
                or type(length) is not int
                or not 0 <= length <= metadata["max_tokens"]
            ):
                raise ValueError("BM25 文档顺序或长度无效")
            keys_digest.update(f"{key[0]}\0{key[1]}\n".encode("utf-8"))
            previous_key = key
            document_count += 1
            total_length += length

        if document_count != metadata["document_count"]:
            raise ValueError("BM25 文档数量不匹配")
        if keys_digest.hexdigest() != metadata["keys_sha256"]:
            raise ValueError("BM25 keys_sha256 校验失败")
        average_length = total_length / document_count if total_length else 1.0
        if average_length != metadata["average_document_length"]:
            raise ValueError("BM25 平均文档长度不匹配")

        invalid_posting = conn.execute(
            """
            SELECT 1 FROM postings
            WHERE typeof(term) != 'text' OR term = ''
               OR typeof(term_frequency) != 'integer' OR term_frequency <= 0
            LIMIT 1
            """
        ).fetchone()
        invalid_length = conn.execute(
            """
            SELECT 1 FROM documents AS d
            LEFT JOIN (
                SELECT doc_id, SUM(term_frequency) AS token_count
                FROM postings GROUP BY doc_id
            ) AS p ON p.doc_id = d.doc_id
            WHERE d.doc_length != COALESCE(p.token_count, 0)
            LIMIT 1
            """
        ).fetchone()
        if invalid_posting is not None or invalid_length is not None:
            raise ValueError("BM25 词频与文档长度不匹配")

        # LEFT JOIN 的两个方向分别检查多余和缺失的 term_stats。
        missing_stats = conn.execute(
            """
            SELECT 1 FROM postings AS p
            LEFT JOIN term_stats AS t ON t.term = p.term
            WHERE t.term IS NULL LIMIT 1
            """
        ).fetchone()
        if missing_stats is not None:
            raise ValueError("BM25 缺少词项统计")
        rows = conn.execute(
            """
            SELECT t.term, t.document_frequency, t.idf, p.actual_frequency
            FROM term_stats AS t
            LEFT JOIN (
                SELECT term, COUNT(*) AS actual_frequency
                FROM postings GROUP BY term
            ) AS p ON p.term = t.term
            """
        )
        for row in rows:
            frequency = row["document_frequency"]
            idf = row["idf"]
            if (
                type(frequency) is not int
                or not 1 <= frequency <= document_count
                or frequency != row["actual_frequency"]
                or type(idf) not in (int, float)
                or not math.isfinite(idf)
            ):
                raise ValueError("BM25 词项统计无效")
            expected_idf = math.log1p(
                (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            if not math.isclose(idf, expected_idf, rel_tol=1e-12):
                raise ValueError("BM25 IDF 与文档频率不匹配")

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "BM25Index":
        self._check_open()
        return self

    def _check_open(self) -> None:
        """关闭后统一报错，包括空查询和空候选集。"""
        if self._conn is None:
            raise RuntimeError("BM25 索引已关闭")

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()

    def _score_terms(
        self,
        terms: tuple[str, ...],
        locale: str,
        doc_ids: set[int] | None = None,
    ) -> dict[int, float]:
        """计算一组 token 对候选商品的 BM25 分数。"""
        self._check_open()
        if not terms:
            return {}

        if doc_ids is not None and not doc_ids:
            return {}

        # 同时为 locale、查询词和候选 ID 预留参数位置。
        # 分词分批时累加贡献，文档分批时合并互不重叠的结果。
        limit = self._conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        term_size = min(
            QUERY_BATCH_SIZE,
            (limit - 1) // 2 if doc_ids is not None else limit - 1,
        )
        if term_size < 1:
            raise ValueError("BM25 SQLite 参数限额过低")
        if len(terms) > term_size:
            scores: dict[int, float] = {}
            for start in range(0, len(terms), term_size):
                partial = self._score_terms(
                    terms[start:start + term_size], locale, doc_ids
                )
                for doc_id, score in partial.items():
                    scores[doc_id] = scores.get(doc_id, 0.0) + score
            return scores

        if doc_ids is not None:
            doc_size = min(QUERY_BATCH_SIZE, limit - len(terms) - 1)
            if doc_size < 1:
                raise ValueError("BM25 SQLite 参数限额过低")
            if len(doc_ids) > doc_size:
                scores = {}
                ids = sorted(doc_ids)
                for start in range(0, len(ids), doc_size):
                    scores.update(self._score_terms(
                        terms, locale, set(ids[start:start + doc_size])
                    ))
                return scores

        placeholders = ", ".join(
            "?" for _ in terms
        )

        params: list[object] = list(terms)
        params.append(locale)

        restriction = ""

        if doc_ids is not None:
            if not doc_ids:
                return {}

            doc_placeholders = ", ".join(
                "?" for _ in doc_ids
            )

            restriction = (
                f" AND p.doc_id IN "
                f"({doc_placeholders})"
            )

            params.extend(
                sorted(doc_ids)
            )

        rows = self._conn.execute(
            f"""
            SELECT
                p.doc_id,
                p.term_frequency,
                d.doc_length,
                t.idf
            FROM postings AS p
            JOIN documents AS d
                ON d.doc_id = p.doc_id
            JOIN term_stats AS t
                ON t.term = p.term
            WHERE p.term IN ({placeholders})
              AND d.locale = ?
              {restriction}
            """,
            params,
        )

        scores: dict[int, float] = {}

        for row in rows:
            term_frequency = row[
                "term_frequency"
            ]

            length_factor = (
                1
                - self.b
                + self.b
                * row["doc_length"]
                / self.average_document_length
            )

            # BM25 的词频部分：tf * (k1 + 1) / (tf + k1 * L)。
            # k1 较大时分子分母同除以 k1，避免有限参数的中间溢出；
            # k1 很小时保留原式，避免计算 1 / k1 溢出。
            if self.k1 > 1:
                numerator = term_frequency * (1 + 1 / self.k1)
                denominator = term_frequency / self.k1 + length_factor
            else:
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * length_factor
            contribution = row["idf"] * (numerator / denominator)

            scores[row["doc_id"]] = (
                scores.get(row["doc_id"], 0.0)
                + contribution
            )

        return scores

    def _doc_ids_for_keys(
        self,
        keys: list[ProductKey],
        locale: str,
    ) -> dict[ProductKey, int]:
        """把商品键转换成 SQLite 内部 doc_id。"""
        self._check_open()
        limit = self._conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        batch_size = min(QUERY_BATCH_SIZE, limit - 1)
        if batch_size < 1:
            raise ValueError("BM25 SQLite 参数限额过低")
        found = {}
        for start in range(0, len(keys), batch_size):
            batch = keys[start:start + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            # keys 已验证属于同一 locale；IN 避免大型 OR 表达式树。
            rows = self._conn.execute(
                f"""
                SELECT doc_id, locale, product_id
                FROM documents
                WHERE locale = ? AND product_id IN ({placeholders})
                """,
                [locale, *(key[1] for key in batch)],
            )
            found.update({
                (row["locale"], row["product_id"]): row["doc_id"]
                for row in rows
            })

        missing = [
            key
            for key in keys
            if key not in found
        ]

        if missing:
            raise KeyError(
                f"BM25 索引中不存在商品：{missing[0]}"
            )

        return {
            key: found[key]
            for key in keys
        }

    def _keys_for_doc_ids(
        self,
        doc_ids: Iterable[int],
    ) -> dict[int, ProductKey]:
        """把 doc_id 转回业务商品键。"""
        self._check_open()
        ids = sorted(set(doc_ids))

        if not ids:
            return {}

        limit = self._conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
        batch_size = min(QUERY_BATCH_SIZE, limit)
        if batch_size < 1:
            raise ValueError("BM25 SQLite 参数限额过低")
        found = {}
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            placeholders = ", ".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT doc_id, locale, product_id
                FROM documents
                WHERE doc_id IN ({placeholders})
                """,
                batch,
            )
            found.update({
                row["doc_id"]: (row["locale"], row["product_id"])
                for row in rows
            })
        return found

    def search(
        self,
        query: Query,
        k: int,
    ) -> list[Hit]:
        """执行 BM25 搜索，返回稳定排序的 Hit。"""
        self._check_open()
        _check_positive_int(k, "k")
        validate_query(query)

        query_key = get_query_key(query)

        terms = tuple(
            sorted(
                set(
                    query_tokens(
                        query.text,
                        version=self.text_version,
                        max_tokens=self.max_tokens,
                    )
                )
            )
        )

        scores = self._score_terms(
            terms,
            query.locale,
        )

        keys = self._keys_for_doc_ids(
            scores
        )

        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                keys[item[0]],
            ),
        )[:k]

        return [
            Hit(
                query_key=query_key,
                product_key=keys[doc_id],
                source="bm25",
                rank=rank,
                score=float(score),
            )
            for rank, (doc_id, score)
            in enumerate(ranked, start=1)
        ]

    def score_pairs(
        self,
        query: Query,
        keys: Iterable[ProductKey],
    ) -> dict[ProductKey, float]:
        """
        为指定商品补算 BM25 分数。

        未命中关键词的商品返回 0.0，
        不会被误认为商品不存在。
        """
        self._check_open()
        validate_query(query)

        requested = list(keys)

        for key in requested:
            _check_product_key(key)

            if key[0] != query.locale:
                raise ValueError(
                    "查询与商品必须属于同一市场"
                )

        if len(set(requested)) != len(requested):
            raise ValueError(
                "score_pairs 不接受重复商品键"
            )

        if not requested:
            return {}

        key_to_doc = self._doc_ids_for_keys(
            requested,
            query.locale,
        )

        terms = tuple(
            sorted(
                set(
                    query_tokens(
                        query.text,
                        version=self.text_version,
                        max_tokens=self.max_tokens,
                    )
                )
            )
        )

        scores = self._score_terms(
            terms,
            query.locale,
            set(key_to_doc.values()),
        )

        return {
            key: float(
                scores.get(doc_id, 0.0)
            )
            for key, doc_id
            in key_to_doc.items()
        }
    
