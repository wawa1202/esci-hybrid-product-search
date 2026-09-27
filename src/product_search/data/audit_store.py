"""只读审计已发布的数据集，生成分区与标签统计报告。"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from product_search.data.store import open_database


PARTITIONS = ("train", "validation", "test")
LABELS = ("E", "S", "C", "I")


def _load_verified_manifest(directory: Path) -> dict:
    """功能一：只接受正式数据集，核对文件与清单。"""
    if (
        directory.name.startswith(".")
        or (directory / "failure.json").exists()
    ):
        raise ValueError(
            "不能审计为正式结果：该目录是暂存目录或失败现场"
        )

    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest["status"] != "prepared":
        raise ValueError("数据集尚未准备完成")

    identity = manifest["identity"]
    signature = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    expected_id = (
        f"{identity['project']['profile']}-{signature[:16]}"
    )

    if (
        manifest["dataset_id"] != expected_id
        or directory.name != expected_id
    ):
        raise ValueError("目录名、数据集 ID 与生成条件不一致")

    for name in ("catalog.sqlite", "labels.sqlite", "rejects.jsonl"):
        path = directory / name
        expected = manifest["files"][name]

        if path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"文件大小与清单不一致：{name}")

        with path.open("rb") as stream:
            digest = hashlib.file_digest(
                stream, "sha256",
            ).hexdigest()

        if digest != expected["sha256"]:
            raise ValueError(f"文件摘要与清单不一致：{name}")

    if (directory / "rejects.jsonl").stat().st_size != 0:
        raise ValueError(
            "当前流程不允许带有拒绝记录的数据集通过审计"
        )

    return manifest


def _check_relations(conn, locale: str) -> None:
    """功能二：检查数据库完整性、关联、标签与市场。"""
    for schema in ("main", "catalog"):
        results = conn.execute(
            f"PRAGMA {schema}.integrity_check"
        )
        if any(row[0] != "ok" for row in results):
            raise ValueError(f"{schema} 数据库完整性检查失败")

    if conn.execute(
        "PRAGMA main.foreign_key_check"
    ).fetchone() is not None:
        raise ValueError("标注引用了不存在的查询")

    checks = {
        "标注引用了不存在的商品": """
            SELECT 1
            FROM judgments j
            LEFT JOIN catalog.products p
                ON j.locale = p.locale
                AND j.product_id = p.product_id
            WHERE p.product_id IS NULL
            LIMIT 1
        """,
        "发现非法标签": """
            SELECT 1
            FROM judgments
            WHERE esci_label NOT IN ('E', 'S', 'C', 'I')
            LIMIT 1
        """,
        "官方来源与实际分区不一致": """
            SELECT 1
            FROM queries
            WHERE NOT (
                (
                    source_split = 'train'
                    AND partition IN ('train', 'validation')
                )
                OR (
                    source_split = 'test'
                    AND partition = 'test'
                )
            )
            LIMIT 1
        """,
    }

    for message, sql in checks.items():
        if conn.execute(sql).fetchone() is not None:
            raise ValueError(message)

    for table in ("queries", "judgments", "catalog.products"):
        row = conn.execute(
            f"SELECT 1 FROM {table} WHERE locale <> ? LIMIT 1",
            (locale,),
        ).fetchone()

        if row is not None:
            raise ValueError(f"{table} 含有非目标市场的数据")


def _summarize_partitions(conn) -> dict:
    """功能三：按查询统计，再汇总成分区报告。"""
    sizes = {p: [] for p in PARTITIONS}
    groups = {p: set() for p in PARTITIONS}
    without_exact = {p: 0 for p in PARTITIONS}
    owners = {}

    rows = conn.execute("""
        SELECT
            q.locale,
            q.query_id,
            q.text,
            q.normalized_group,
            q.partition,
            COUNT(j.product_id) AS candidate_count,
            SUM(
                CASE WHEN j.esci_label = 'E' THEN 1 ELSE 0 END
            ) AS exact_count
        FROM queries q
        LEFT JOIN judgments j
            ON q.locale = j.locale
            AND q.query_id = j.query_id
        GROUP BY q.locale, q.query_id
    """)

    for row in rows:
        part = row["partition"]

        if part not in PARTITIONS or row["candidate_count"] == 0:
            raise ValueError("发现非法分区或没有标注的查询")

        normalized = " ".join(row["text"].casefold().split())

        if not normalized or normalized != row["normalized_group"]:
            raise ValueError(
                f"查询文本组记录不正确：{row['query_id']}"
            )

        key = (row["locale"], normalized)

        if key in owners and owners[key] != part:
            raise ValueError(f"查询文本组跨分区：{key}")

        owners[key] = part
        groups[part].add(key)
        sizes[part].append(row["candidate_count"])
        without_exact[part] += int(row["exact_count"] == 0)

    label_counts = {
        p: {label: 0 for label in LABELS}
        for p in PARTITIONS
    }

    for row in conn.execute("""
        SELECT q.partition, j.esci_label, COUNT(*) AS n
        FROM judgments j
        JOIN queries q
            ON j.locale = q.locale
            AND j.query_id = q.query_id
        GROUP BY q.partition, j.esci_label
    """):
        label_counts[row["partition"]][row["esci_label"]] = row["n"]

    result = {}

    for part in PARTITIONS:
        values = sizes[part]

        if not values:
            raise ValueError(f"分区没有查询：{part}")

        total = sum(values)

        result[part] = {
            "queries": len(values),
            "query_groups": len(groups[part]),
            "judgments": total,
            "label_counts": label_counts[part],
            "label_ratios": {
                label: count / total
                for label, count in label_counts[part].items()
            },
            "judged_candidates_per_query": {
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "median": median(values),
            },
            "queries_without_exact": without_exact[part],
        }

    return result


def audit_dataset(dataset_dir: str | Path) -> dict:
    """功能四：组织审计，返回报告；硬性检查失败则抛错。"""
    directory = Path(dataset_dir).resolve()
    manifest = _load_verified_manifest(directory)

    with open_database(
        directory / "labels.sqlite", read_only=True,
    ) as conn:
        uri = (
            (directory / "catalog.sqlite").as_uri()
            + "?mode=ro"
        )
        conn.execute(
            "ATTACH DATABASE ? AS catalog",
            (uri,),
        )

        _check_relations(
            conn,
            manifest["identity"]["project"]["locale"],
        )
        partitions = _summarize_partitions(conn)

        products = conn.execute(
            "SELECT COUNT(*) FROM catalog.products"
        ).fetchone()[0]

        judged_products = conn.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT locale, product_id
                FROM judgments
            )
        """).fetchone()[0]

    basic = manifest["basic_integrity"]

    if (
        basic["status"] != "passed"
        or basic["products"] != products
        or basic["judgments"] != sum(
            p["judgments"] for p in partitions.values()
        )
        or basic["queries_by_partition"] != {
            name: p["queries"]
            for name, p in partitions.items()
        }
    ):
        raise ValueError("实际数量与准备阶段清单不一致")

    scope = manifest["identity"]["selection"]["catalog_scope"]

    if scope not in {
        "sampled_judged_union",
        "full_locale_catalog",
    }:
        raise ValueError("不支持的商品目录范围")

    if (
        scope == "sampled_judged_union"
        and products != judged_products
    ):
        raise ValueError(
            "开发商品目录不等于已选标注涉及的商品集合"
        )

    notes = ["未标注的查询—商品对不是已知负例。"]

    if scope == "sampled_judged_union":
        notes.append(
            "当前是样本商品目录；其结果不能代表全目录检索效果。"
        )

    if any(
        p["queries_without_exact"]
        for p in partitions.values()
    ):
        notes.append(
            "部分查询没有 E 标签；记录这一现象，不自动删除查询。"
        )

    return {
        "audit_version": "audit-v1",
        "dataset_id": manifest["dataset_id"],
        "dataset_dir": str(directory),
        "status": "passed",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_scope": scope,
        "products": products,
        "products_with_any_judgment": judged_products,
        "products_without_any_judgment": products - judged_products,
        "partitions": partitions,
        "notes": notes,
    }


def save_audit_report(
    report: dict,
    output_path: str | Path,
) -> Path:
    """功能五：报告保存在数据集外，拒绝覆盖已有文件。"""
    path = Path(output_path).resolve()

    if path.is_relative_to(
        Path(report["dataset_dir"]).resolve()
    ):
        raise ValueError("报告必须保存在数据集目录之外")

    content = (
        json.dumps(report, ensure_ascii=False, indent=2)
        + "\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)

    return path