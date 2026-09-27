"""区分给定标注排序与目录检索的评估指标。"""

import math

from product_search.schemas import Judgment, Query, SearchResult
from product_search.validation import (
    validate_judgment,
    validate_query,
    validate_search_result,
)


GAINS = {'I': 0, 'C': 1, 'S': 3, 'E': 7}


def _validate(
    query: Query, results: list[SearchResult], judgments: list[Judgment], k: int,
) -> tuple[list[SearchResult], dict[str, str]]:
    """校验全部记录，按显式排名返回副本。"""
    if type(k) is not int:
        raise TypeError('k 必须是整数')
    if k < 1:
        raise ValueError('k 必须大于 0')
    validate_query(query)
    key = (query.locale, query.query_id)
    labels = {}
    for judgment in judgments:
        validate_judgment(judgment)
        if (judgment.locale, judgment.query_id) != key:
            raise ValueError('标注必须属于同一查询和市场')
        if judgment.product_id in labels:
            if labels[judgment.product_id] != judgment.esci_label:
                raise ValueError('同一商品存在冲突标签')
            raise ValueError('标注商品重复')
        labels[judgment.product_id] = judgment.esci_label

    seen = set()
    for result in results:
        validate_search_result(result)
        if (result.locale, result.query_id) != key:
            raise ValueError('结果必须属于同一查询和市场')
        if result.product_id in seen:
            raise ValueError('结果商品重复')
        seen.add(result.product_id)
    ordered = sorted(results, key=lambda result: result.rank)
    if any(result.rank != rank for rank, result in enumerate(ordered, 1)):
        raise ValueError('rank 必须从 1 开始连续且不重复')
    return ordered, labels


def ranking_metrics(
    query: Query, results: list[SearchResult], judgments: list[Judgment], *, k: int = 10,
) -> dict:
    """计算 judged-ranking；无正 gain/E 时对应指标为 None。"""
    ordered, labels = _validate(query, results, judgments, k)
    if any(result.product_id not in labels for result in ordered):
        raise ValueError('judged-ranking 不接受未知候选')
    gains = [GAINS[labels[result.product_id]] for result in ordered[:k]]
    ideal = sorted((GAINS[label] for label in labels.values()), reverse=True)[:k]
    dcg = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    idcg = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    mrr = None
    if 'E' in labels.values():
        mrr = next((1.0 / result.rank for result in ordered[:k]
                    if labels[result.product_id] == 'E'), 0.0)
    return {'ndcg': dcg / idcg if idcg else None, 'mrr_e': mrr}


def retrieval_metrics(
    query: Query, results: list[SearchResult], judgments: list[Judgment], *, k: int = 100,
) -> dict:
    """计算已知标注召回；覆盖率分母为实际 top-k 数，空分母为 None。"""
    ordered, labels = _validate(query, results, judgments, k)
    retrieved = {result.product_id for result in ordered[:k]}
    exact = {pid for pid, label in labels.items() if label == 'E'}
    relevant = {pid for pid, label in labels.items() if label in {'E', 'S'}}
    return {
        'known_e_recall': len(retrieved & exact) / len(exact) if exact else None,
        'known_es_recall': len(retrieved & relevant) / len(relevant) if relevant else None,
        'judged_fraction': len(retrieved & labels.keys()) / len(retrieved) if retrieved else None,
    }
