"""统计已组装的数据集，不修改或筛选数据。"""

from collections import Counter

from product_search.data.build_dataset import Dataset


def summarize_dataset(dataset: Dataset) -> dict[str, object]:
    """生成查询、商品、标注和候选分布的概况报告。"""

    if not dataset.queries or not dataset.judgments:
        raise ValueError("数据集必须包含查询和标注")

    # 标签分别出现了多少次。
    label_counts = Counter(
        judgment.esci_label
        for judgment in dataset.judgments
    )

    # 每个查询对应多少条商品标注。
    candidates_by_query = Counter(
        (judgment.locale, judgment.query_id)
        for judgment in dataset.judgments
    )

    # 哪些查询至少拥有一个 Exact 商品。
    queries_with_exact = {
        (judgment.locale, judgment.query_id)
        for judgment in dataset.judgments
        if judgment.esci_label == "E"
    }

    # 哪些商品至少出现在一条标注中。
    judged_product_keys = {
        (judgment.locale, judgment.product_id)
        for judgment in dataset.judgments
    }

    all_product_keys = {
        (product.locale, product.product_id)
        for product in dataset.products
    }

    # 每个市场包含多少个查询。
    locale_query_counts = Counter(
        query.locale
        for query in dataset.queries
    )

    candidate_counts = [
        candidates_by_query[(query.locale, query.query_id)]
        for query in dataset.queries
    ]

    query_count = len(dataset.queries)
    judgment_count = len(dataset.judgments)

    return {
        "source_split": dataset.source_split,
        "query_count": query_count,
        "product_count": len(dataset.products),
        "judgment_count": judgment_count,
        "judged_product_count": len(judged_product_keys),
        "unjudged_product_count": len(
            all_product_keys - judged_product_keys
        ),
        "label_counts": {
            label: label_counts[label]
            for label in ("E", "S", "C", "I")
        },
        "label_ratios": {
            label: label_counts[label] / judgment_count
            for label in ("E", "S", "C", "I")
        },
        "locale_query_counts": dict(locale_query_counts),
        "candidates_per_query": {
            "min": min(candidate_counts),
            "max": max(candidate_counts),
            "mean": sum(candidate_counts) / query_count,
        },
        "queries_with_exact": len(queries_with_exact),
        "queries_without_exact": (
            query_count - len(queries_with_exact)
        ),
    }