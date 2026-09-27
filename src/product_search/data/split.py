"""从官方训练数据中，按查询文本组划分训练集和验证集。"""

import random
from dataclasses import dataclass

from product_search.data.build_dataset import Dataset


@dataclass
class TrainingSplit:
    """划分结果，以及本次划分使用的参数。"""

    train: Dataset
    validation: Dataset
    seed: int
    validation_fraction: float


def _select_queries(
    dataset: Dataset,
    query_keys: set[tuple[str, str]],
) -> Dataset:
    """保留选中的查询及其全部标注。"""

    queries = [
        query
        for query in dataset.queries
        if (query.locale, query.query_id) in query_keys
    ]

    judgments = [
        judgment
        for judgment in dataset.judgments
        if (judgment.locale, judgment.query_id) in query_keys
    ]

    return Dataset(
        source_split=dataset.source_split,
        queries=queries,
        products=list(dataset.products),
        judgments=judgments,
    )


def split_train_validation(
    dataset: Dataset,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> TrainingSplit:
    """按查询组划分，避免同一查询出现在两个集合中。"""

    if dataset.source_split != "train":
        raise ValueError("只能从官方 train 中划分验证集")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 和 1 之间")

    if type(seed) is not int:
        raise TypeError("seed 必须是整数")

    # 组键是“市场 + 规范化查询文本”。
    # 每个组保存它包含的查询身份。
    groups: dict[
        tuple[str, str],
        set[tuple[str, str]],
    ] = {}

    all_query_keys: set[tuple[str, str]] = set()

    for query in dataset.queries:
        query_key = (query.locale, query.query_id)

        if query_key in all_query_keys:
            raise ValueError(f"查询身份重复：{query_key}")

        all_query_keys.add(query_key)

        normalized_text = " ".join(
            query.text.casefold().split()
        )
        group_key = (query.locale, normalized_text)

        groups.setdefault(group_key, set()).add(query_key)

    if len(groups) < 2:
        raise ValueError("至少需要两个不同的查询文本组")

    # 先排序，消除原始记录顺序对划分结果的影响。
    group_keys = sorted(groups)

    # 使用局部随机数生成器，不改变其他代码的随机状态。
    random.Random(seed).shuffle(group_keys)

    validation_group_count = round(
        len(group_keys) * validation_fraction
    )

    # 确保两个集合都至少包含一个组。
    validation_group_count = max(
        1,
        min(len(group_keys) - 1, validation_group_count),
    )

    validation_query_keys: set[tuple[str, str]] = set()

    for group_key in group_keys[:validation_group_count]:
        validation_query_keys.update(groups[group_key])

    train_query_keys = (
        all_query_keys - validation_query_keys
    )

    return TrainingSplit(
        train=_select_queries(dataset, train_query_keys),
        validation=_select_queries(
            dataset,
            validation_query_keys,
        ),
        seed=seed,
        validation_fraction=validation_fraction,
    )