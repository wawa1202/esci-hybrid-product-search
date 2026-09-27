"""对同一查询集合做可复现的配对 bootstrap。"""

import math
import random
from statistics import mean


def _metric(value: float | None) -> None:
    """允许未定义值，其余必须是有限数值。"""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('指标必须是数值或 None')
    if not math.isfinite(value):
        raise ValueError('指标必须有限')


def _percentile(values: list[float], fraction: float) -> float:
    """对已排序样本做线性分位数插值。"""
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return (1 - weight) * values[lower] + weight * values[upper]


def paired_bootstrap(
    left: dict[str, float | None], right: dict[str, float | None], *,
    iterations: int = 1000, seed: int = 42,
) -> dict:
    """报告 right-left；任一侧为 None 的查询整对排除。"""
    if type(iterations) is not int or type(seed) is not int:
        raise TypeError('iterations 和 seed 必须是整数')
    if iterations < 1:
        raise ValueError('iterations 必须大于 0')
    if left.keys() != right.keys():
        raise ValueError('两侧 query key sets 必须相同')
    if any(not isinstance(key, str) or not key.strip() for key in left):
        raise ValueError('query key 必须是非空字符串')

    differences = []
    for key in sorted(left):
        a, b = left[key], right[key]
        _metric(a)
        _metric(b)
        if a is None or b is None:
            continue
        delta = b - a
        if not math.isfinite(delta):
            raise ValueError('指标差值溢出')
        differences.append(delta)

    n = len(differences)
    result = {
        'mean_delta': None,
        'ci95': [None, None],
        'wins': sum(delta > 0 for delta in differences),
        'losses': sum(delta < 0 for delta in differences),
        'ties': sum(delta == 0 for delta in differences),
        'n': n,
        'excluded': len(left) - n,
    }
    if not n:
        return result
    rng = random.Random(seed)
    means = sorted(mean(differences[rng.randrange(n)] for _ in range(n))
                   for _ in range(iterations))
    result['mean_delta'] = mean(differences)
    result['ci95'] = [_percentile(means, 0.025), _percentile(means, 0.975)]
    return result
