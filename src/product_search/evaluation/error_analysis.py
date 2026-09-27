"""只依据给定指标、排名与标签生成可核验错误记录。"""

import math


PIPELINES = ('bm25', 'dense', 'rrf', 'ltr')
METRICS = ('ndcg', 'mrr_e')
COMPARISONS = (('dense', 'bm25'), ('rrf', 'bm25'), ('ltr', 'bm25'), ('ltr', 'rrf'))


def _identifier(value: object, field: str) -> None:
    """检查有内容的字符串。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} 必须是非空字符串')


def _validate_row(row: dict) -> None:
    """先完整校验单个查询，避免截断隐藏非法记录。"""
    required = {'query_id', 'text', 'metrics', 'rankings', 'labels'}
    if not isinstance(row, dict) or not required <= row.keys():
        raise ValueError('错误分析行缺少必要字段')
    _identifier(row['query_id'], 'query_id')
    _identifier(row['text'], 'text')
    for field in ('metrics', 'rankings', 'labels'):
        if not isinstance(row[field], dict):
            raise TypeError(f'{field} 必须是字典')
    for pid, label in row['labels'].items():
        _identifier(pid, 'product_id')
        if not isinstance(label, str) or label not in {'E', 'S', 'C', 'I'}:
            raise ValueError('标签必须是 E、S、C 或 I')
    available = set(row['metrics'])
    if available != set(row['rankings']) or not {'bm25', 'dense', 'rrf'} <= available or not available <= set(PIPELINES):
        raise ValueError('指标与排名的方法必须一致，至少含 bm25/dense/rrf')
    for pipeline in (name for name in PIPELINES if name in available):
        metrics = row['metrics'][pipeline]
        if not isinstance(metrics, dict) or not set(METRICS) <= metrics.keys():
            raise ValueError('每路指标必须包含 ndcg 和 mrr_e')
        for metric in METRICS:
            value = metrics[metric]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError('指标必须是数值或 None')
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('ndcg 和 mrr_e 必须在 [0, 1] 内')
        ranking = row['rankings'][pipeline]
        if not isinstance(ranking, list):
            raise TypeError('排名必须是商品 ID 列表')
        for pid in ranking:
            _identifier(pid, 'product_id')
        if len(set(ranking)) != len(ranking):
            raise ValueError('排名中商品重复')


def analyze_errors(rows: list[dict], *, limit: int = 30) -> list[dict]:
    """返回失败查询、类别、差值与证据；先按最差 NDCG 差值再按 ID 排序。"""
    if type(limit) is not int:
        raise TypeError('limit 必须是整数')
    if limit < 0:
        raise ValueError('limit 必须非负')
    seen = set()
    errors = []
    for row in rows:
        _validate_row(row)
        query_id = row['query_id']
        if query_id in seen:
            raise ValueError('query_id 重复')
        seen.add(query_id)
        labels = row['labels']
        exact = {pid for pid, label in labels.items() if label == 'E'}
        categories = []
        evidence = {}
        for pipeline in (name for name in PIPELINES if name in row['rankings']):
            ranking = row['rankings'][pipeline]
            exact_ranks = [rank for rank, pid in enumerate(ranking, 1)
                           if labels.get(pid) == 'E']
            first_e = exact_ranks[0] if exact_ranks else None
            unknown = [pid for pid in ranking if pid not in labels]
            missing = sorted(exact - set(ranking))
            preceding_i = ([pid for pid in ranking[:exact_ranks[-1] - 1]
                            if labels.get(pid) == 'I'] if exact_ranks else [])
            evidence[pipeline] = {
                'first_e_rank': first_e,
                'unknown_products': unknown,
                'missing_known_e': missing,
                'i_before_e': preceding_i,
            }
            for category, applies in (
                ('empty_ranking', not ranking),
                ('unknown_candidates', bool(unknown)),
                ('known_e_missing', bool(missing)),
                ('exact_below_irrelevant', bool(preceding_i)),
            ):
                if applies:
                    categories.append(f'{pipeline}:{category}')
        deltas = {}
        for right, left in COMPARISONS:
            if right not in row['metrics'] or left not in row['metrics']:
                continue
            pair = {}
            for metric in METRICS:
                a, b = row['metrics'][left][metric], row['metrics'][right][metric]
                delta = b - a if a is not None and b is not None else None
                pair[metric] = delta
                if delta is not None and delta < 0:
                    categories.append(f'{right}:{metric}_regression_vs_{left}')
            deltas[f'{right}_minus_{left}'] = pair
        if categories:
            errors.append({'query_id': query_id, 'text': row['text'],
                           'categories': categories, 'deltas': deltas, 'evidence': evidence})

    def priority(error: dict) -> tuple[float, str]:
        differences = [pair['ndcg'] for pair in error['deltas'].values()
                       if pair['ndcg'] is not None]
        return min([0.0, *differences]), error['query_id']

    return sorted(errors, key=priority)[:limit]
