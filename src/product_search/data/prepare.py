"""从固定版本原始文件构建可追溯的 SQLite 数据集。"""

import hashlib
import json
import random
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import batched
from pathlib import Path
from tempfile import mkdtemp

from product_search.contracts import get_query_key
from product_search.data.build_dataset import Dataset
from product_search.data.download import (
    ASSETS,
    REPOSITORY,
    REVISION,
)
from product_search.data.normalize import (
    judgment_from_row,
    normalize_id,
    product_from_row,
    query_from_row,
)
from product_search.data.read_parquet import (
    iter_example_rows,
    iter_product_rows,
)
from product_search.data.split import split_train_validation
from product_search.data.store import (
    DataConflictError,
    init_schema,
    insert_judgment,
    insert_product,
    insert_query,
    open_database,
    transaction,
)
from product_search.settings import Settings


PREPARE_VERSION = "prepare-v1"
BATCH_SIZE = 4096


def _write_json(path: Path, value: object) -> None:
    """把记录写成便于检查的 JSON 文件。"""
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """分块计算文件摘要，不把整个文件读入内存。"""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_inputs(raw_dir: Path) -> dict:
    """同时核对来源记录和文件内容；不下载、不跳过校验。"""
    verified = {}

    for name in ("examples", "products"):
        asset = ASSETS[name]
        path = raw_dir / asset["filename"]

        manifest_path = (
            raw_dir
            / f"{asset['filename']}.manifest.json"
        )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8-sig")
        )

        expected = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "filename": asset["filename"],
            "size_bytes": asset["size"],
            "sha256": asset["sha256"],
        }

        if not isinstance(manifest, dict):
            raise ValueError(
                f"来源记录必须是 JSON 对象：{manifest_path}"
            )

        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(
                    f"{name} 来源记录不匹配：{field}"
                )

        if path.stat().st_size != asset["size"]:
            raise ValueError(
                f"{name} 文件大小不匹配"
            )

        if _sha256(path) != asset["sha256"]:
            raise ValueError(
                f"{name} 文件 SHA256 不匹配"
            )

        verified[name] = expected

    return verified


@contextmanager
def _row_context(
    rejects: Path,
    phase: str,
    row: dict,
):
    """首个行级错误记录下来，并继续抛出，禁止静默丢弃。"""
    try:
        yield

    except Exception as exc:
        record = {
            "phase": phase,
            "error": str(exc),
            "row": row,
        }

        with rejects.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                ) + "\n"
            )

        raise


def _collect_queries(
    examples: Path,
    locale: str,
    rejects: Path,
) -> dict:
    """只保存去重后的查询元数据，不收集商品和全部标注。"""
    collected = {
        "train": {},
        "test": {},
    }

    id_owners = {}
    group_owners = {}

    for source in ("train", "test"):
        with closing(
            iter_example_rows(
                examples,
                locale=locale,
                source_split=source,
            )
        ) as rows:
            for row in rows:
                with _row_context(
                    rejects,
                    "collect_queries",
                    row,
                ):
                    query = query_from_row(row)
                    key = get_query_key(query)

                    group = (
                        query.locale,
                        " ".join(query.text.casefold().split()),
                    )

                    if (
                        key in id_owners
                        and id_owners[key] != source
                    ):
                        raise DataConflictError(
                            f"官方 train/test 查询身份重叠：{key}"
                        )

                    if (
                        group in group_owners
                        and group_owners[group] != source
                    ):
                        raise DataConflictError(
                            f"官方 train/test 查询文本组重叠：{group}"
                        )

                    previous = collected[source].get(key)

                    if previous is not None and previous != query:
                        raise DataConflictError(
                            f"同一查询身份对应不同文本：{key}"
                        )

                    collected[source][key] = query
                    id_owners[key] = source
                    group_owners[group] = source

    if not collected["train"] or not collected["test"]:
        raise ValueError(
            "当前市场的官方 train 和 test 都必须有查询"
        )

    return collected


def _plan_queries(
    collected: dict,
    settings: Settings,
) -> dict:
    """先划分完整训练查询组，再对各分区抽样。"""
    training = Dataset(
        source_split="train",
        queries=list(collected["train"].values()),
        products=[],
        judgments=[],
    )

    split = split_train_validation(
        training,
        validation_fraction=settings.data.validation_fraction,
        seed=settings.project.seed,
    )

    partitions = {
        "train": split.train.queries,
        "validation": split.validation.queries,
        "test": list(collected["test"].values()),
    }

    selected = {}
    limit = settings.data.sample_groups_per_partition

    for partition, queries in partitions.items():
        groups = {}

        for query in queries:
            group = (
                query.locale,
                " ".join(query.text.casefold().split()),
            )
            groups.setdefault(group, []).append(query)

        chosen = sorted(groups)

        if limit > 0:
            chosen = random.Random(
                settings.project.seed
            ).sample(
                chosen,
                min(limit, len(chosen)),
            )

        source = (
            "test"
            if partition == "test"
            else "train"
        )

        for group in chosen:
            for query in groups[group]:
                selected[get_query_key(query)] = (
                    query,
                    source,
                    partition,
                )

    return selected


def _write_labels(
    conn,
    examples: Path,
    selected: dict,
    locale: str,
    rejects: Path,
) -> None:
    """先写查询，再分批写入这些查询对应的全部标注。"""
    with transaction(conn):
        for key in sorted(selected):
            query, source, partition = selected[key]

            insert_query(
                conn,
                query,
                source_split=source,
                partition=partition,
            )

    for source in ("train", "test"):
        with closing(
            iter_example_rows(
                examples,
                locale=locale,
                source_split=source,
            )
        ) as rows:
            for batch in batched(rows, BATCH_SIZE):
                with transaction(conn):
                    for row in batch:
                        with _row_context(
                            rejects,
                            "write_labels",
                            row,
                        ):
                            key = (
                                locale,
                                normalize_id(row["query_id"]),
                            )

                            if key not in selected:
                                continue

                            if selected[key][1] != source:
                                raise DataConflictError(
                                    "查询来源与预定划分不一致"
                                )

                            insert_judgment(
                                conn,
                                judgment_from_row(row),
                            )


def _write_products(
    conn,
    products: Path,
    locale: str,
    required,
    rejects: Path,
) -> None:
    """dev 保存需要的商品，full 保存当前市场全部有效商品。"""
    with closing(
        iter_product_rows(
            products,
            locale=locale,
        )
    ) as rows:
        for batch in batched(rows, BATCH_SIZE):
            with transaction(conn):
                for row in batch:
                    with _row_context(
                        rejects,
                        "write_products",
                        row,
                    ):
                        key = (
                            locale,
                            normalize_id(row["product_id"]),
                        )

                        if (
                            required is not None
                            and key not in required
                        ):
                            continue

                        insert_product(
                            conn,
                            product_from_row(row),
                        )


def _check_databases(stage: Path) -> dict:
    """发布前的基础完整性检查；详细报告由 audit_store.py 扩展。"""
    with open_database(
        stage / "labels.sqlite",
        read_only=True,
    ) as conn:
        catalog_uri = (
            (stage / "catalog.sqlite").resolve().as_uri()
            + "?mode=ro"
        )

        conn.execute(
            "ATTACH DATABASE ? AS catalog",
            (catalog_uri,),
        )

        for schema in ("main", "catalog"):
            result = conn.execute(
                f"PRAGMA {schema}.quick_check"
            ).fetchone()[0]

            if result != "ok":
                raise ValueError(
                    f"{schema} SQLite 文件完整性检查失败"
                )

        if conn.execute(
            "PRAGMA main.foreign_key_check"
        ).fetchone() is not None:
            raise ValueError(
                "标注引用了不存在的查询"
            )

        missing = conn.execute(
            """
            SELECT j.locale, j.query_id, j.product_id
            FROM judgments AS j
            LEFT JOIN catalog.products AS p
                ON j.locale = p.locale
                AND j.product_id = p.product_id
            WHERE p.product_id IS NULL
            LIMIT 1
            """
        ).fetchone()

        if missing is not None:
            raise ValueError(
                f"标注引用了缺失商品：{tuple(missing)}"
            )

        empty = conn.execute(
            """
            SELECT q.locale, q.query_id
            FROM queries AS q
            WHERE NOT EXISTS (
                SELECT 1
                FROM judgments AS j
                WHERE j.locale = q.locale
                  AND j.query_id = q.query_id
            )
            LIMIT 1
            """
        ).fetchone()

        if empty is not None:
            raise ValueError(
                f"查询没有标注：{tuple(empty)}"
            )

        overlap = conn.execute(
            """
            SELECT locale, normalized_group
            FROM queries
            GROUP BY locale, normalized_group
            HAVING COUNT(DISTINCT partition) > 1
            LIMIT 1
            """
        ).fetchone()

        if overlap is not None:
            raise ValueError(
                f"查询文本组跨分区：{tuple(overlap)}"
            )

        counts = dict(
            conn.execute(
                """
                SELECT partition, COUNT(*)
                FROM queries
                GROUP BY partition
                """
            ).fetchall()
        )

        if set(counts) != {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                "train、validation、test 必须各有查询"
            )

        return {
            "status": "passed",
            "queries_by_partition": counts,
            "products": conn.execute(
                "SELECT COUNT(*) FROM catalog.products"
            ).fetchone()[0],
            "judgments": conn.execute(
                "SELECT COUNT(*) FROM judgments"
            ).fetchone()[0],
        }


def prepare_dataset(settings: Settings) -> Path:
    """成功返回正式目录；失败保留暂存目录，不覆盖旧结果。"""
    raw_dir = settings.resolve_path(
        settings.data.raw_dir
    )
    output_root = settings.resolve_path(
        settings.data.processed_dir
    )

    print(
        "1/5 校验原始文件来源、大小和 SHA256",
        flush=True,
    )
    sources = _verify_inputs(raw_dir)

    identity = {
        "prepare_version": PREPARE_VERSION,
        "sources": sources,
        "project": asdict(settings.project),
        "selection": {
            "validation_fraction": (
                settings.data.validation_fraction
            ),
            "sample_groups_per_partition": (
                settings.data.sample_groups_per_partition
            ),
            "catalog_scope": settings.data.catalog_scope,
        },
    }

    signature = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    dataset_id = (
        f"{settings.project.profile}-{signature[:16]}"
    )
    destination = output_root / dataset_id

    if destination.exists():
        raise FileExistsError(
            f"同版本数据集已存在，不重复写入：{destination}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage = Path(
        mkdtemp(
            prefix=f".{dataset_id}-pending-",
            dir=output_root,
        )
    )

    rejects = stage / "rejects.jsonl"
    rejects.touch()

    examples = raw_dir / ASSETS["examples"]["filename"]
    products = raw_dir / ASSETS["products"]["filename"]

    try:
        print(
            "2/5 收集查询、检查隔离、划分并抽样",
            flush=True,
        )

        collected = _collect_queries(
            examples,
            settings.project.locale,
            rejects,
        )
        selected = _plan_queries(
            collected,
            settings,
        )

        del collected

        with open_database(
            stage / "labels.sqlite",
            create=True,
        ) as labels:
            init_schema(labels, "labels")

            print(
                "3/5 写入查询和标注",
                flush=True,
            )

            _write_labels(
                labels,
                examples,
                selected,
                settings.project.locale,
                rejects,
            )

            required = None

            if (
                settings.data.catalog_scope
                == "sampled_judged_union"
            ):
                required = {
                    tuple(row)
                    for row in labels.execute(
                        """
                        SELECT DISTINCT locale, product_id
                        FROM judgments
                        """
                    )
                }

        del selected

        with open_database(
            stage / "catalog.sqlite",
            create=True,
        ) as catalog:
            init_schema(catalog, "catalog")

            print(
                "4/5 写入商品目录",
                flush=True,
            )

            _write_products(
                catalog,
                products,
                settings.project.locale,
                required,
                rejects,
            )

        print(
            "5/5 检查关联、记录版本、发布数据集",
            flush=True,
        )

        checks = _check_databases(stage)

        _write_json(
            stage / "manifest.json",
            {
                "dataset_id": dataset_id,
                "status": "prepared",
                "created_at_utc": (
                    datetime.now(timezone.utc).isoformat()
                ),
                "identity": identity,
                "settings_snapshot": asdict(settings),
                "basic_integrity": checks,
                "files": {
                    name: {
                        "size_bytes": (
                            stage / name
                        ).stat().st_size,
                        "sha256": _sha256(stage / name),
                    }
                    for name in (
                        "catalog.sqlite",
                        "labels.sqlite",
                        "rejects.jsonl",
                    )
                },
            },
        )

        if destination.exists():
            raise FileExistsError(
                f"目标已存在，拒绝覆盖：{destination}"
            )

        stage.rename(destination)

    except Exception as exc:
        _write_json(
            stage / "failure.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "identity": identity,
            },
        )

        raise RuntimeError(
            f"数据准备失败，现场保留在：{stage}"
        ) from exc

    return destination