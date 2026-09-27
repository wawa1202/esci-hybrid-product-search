"""分批读取 ESCI 原始数据，返回供清洗模块使用的字典。"""

from collections.abc import Generator
from contextlib import closing
from pathlib import Path

import pyarrow.parquet as pq

EXAMPLE_COLUMNS = [
    "query_id",
    "query",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "split",
]

PRODUCT_COLUMNS = [
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]


def _iter_rows(
    path: Path,
    columns: list[str],
    *,
    batch_size: int,
) -> Generator[dict[str, object], None, None]:
    """读取指定列，每批转换为普通 Python 字典。"""

    if type(batch_size) is not int:
        raise TypeError("batch_size 必须是整数")
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0")

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    with closing(pq.ParquetFile(path, pre_buffer=False)) as parquet:
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"数据文件缺少字段：{sorted(missing)}")

        batches = parquet.iter_batches(
            batch_size=batch_size,
            columns=columns,
            use_threads=False,
        )

        for batch in batches:
            yield from batch.to_pylist()


def iter_example_rows(
    path: Path,
    *,
    locale: str,
    source_split: str,
    batch_size: int = 4096,
) -> Generator[dict[str, object], None, None]:
    """读取指定市场、指定官方划分的 Task 1 标注。"""

    if locale not in {"us", "es", "jp"}:
        raise ValueError("locale 必须是 us、es 或 jp")
    if source_split not in {"train", "test"}:
        raise ValueError("source_split 必须是 train 或 test")

    with closing(
        _iter_rows(path, EXAMPLE_COLUMNS, batch_size=batch_size)
    ) as rows:
        for row in rows:
            if row["product_locale"] != locale:
                continue
            if row["split"] != source_split:
                continue
            if row["small_version"] != 1:
                continue
            yield row


def iter_product_rows(
    path: Path,
    *,
    locale: str,
    batch_size: int = 4096,
) -> Generator[dict[str, object], None, None]:
    """读取指定市场的商品，不根据标签裁剪商品目录。"""

    if locale not in {"us", "es", "jp"}:
        raise ValueError("locale 必须是 us、es 或 jp")

    with closing(
        _iter_rows(path, PRODUCT_COLUMNS, batch_size=batch_size)
    ) as rows:
        for row in rows:
            if row["product_locale"] == locale:
                yield row