"""RRF 只合并排名，不直接相加 BM25 与余弦分数。"""

from collections import defaultdict
from collections.abc import Iterable

from product_search.contracts import Candidate, Hit


def fuse_hits(lists: Iterable[Iterable[Hit]], *, constant: int = 60, k: int | None = None) -> list[Candidate]:
    if type(constant) is not int or constant < 1:
        raise ValueError('RRF constant 必须是正整数')
    if k is not None and (type(k) is not int or k < 1):
        raise ValueError('k 必须是正整数')
    grouped = defaultdict(list)
    query_key = None
    sources = set()
    for values in lists:
        values = list(values)
        if not values:
            continue
        source = values[0].source
        if source in sources:
            raise ValueError('同一召回源只能提供一个列表')
        sources.add(source)
        seen = set()
        for expected_rank, hit in enumerate(values, 1):
            query_key = query_key or hit.query_key
            if hit.query_key != query_key or hit.source != source:
                raise ValueError('不能混合不同查询或不同召回源')
            if hit.rank != expected_rank or hit.product_key in seen:
                raise ValueError('每路结果必须排名连续且商品唯一')
            seen.add(hit.product_key)
            grouped[hit.product_key].append(hit)
    scores = {key: sum(1.0 / (constant + h.rank) for h in hits) for key, hits in grouped.items()}
    ordered = sorted(grouped, key=lambda key: (-scores[key], key))
    if k is not None:
        ordered = ordered[:k]
    return [Candidate(query_key, key, tuple(grouped[key]), rank, float(scores[key]))
            for rank, key in enumerate(ordered, 1)]
