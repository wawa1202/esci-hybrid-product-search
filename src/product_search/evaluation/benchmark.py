"""汇总调用方提供的阶段耗时，不运行真实检索。"""

import math


def _percentile(values: list[float], fraction: float) -> float:
    """对已排序样本做线性分位数插值。"""
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return values[lower] + (values[upper] - values[lower]) * weight


def summarize_timings(samples: list[dict[str, float]]) -> dict:
    """输入单位为毫秒；各阶段独立计数，缺失阶段不补零。"""
    stages: dict[str, list[float]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise TypeError('每个耗时样本必须是字典')
        for stage, value in sample.items():
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError('阶段名必须是非空字符串')
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError('耗时必须是数值')
            if not math.isfinite(value) or value < 0:
                raise ValueError('耗时必须有限且非负')
            stages.setdefault(stage, []).append(float(value))
    result = {}
    for stage in sorted(stages):
        values = sorted(stages[stage])
        result[stage] = {
            'count': len(values),
            'p50_ms': _percentile(values, 0.5),
            'p95_ms': _percentile(values, 0.95),
            'p99_ms': _percentile(values, 0.99),
        }
    return result
