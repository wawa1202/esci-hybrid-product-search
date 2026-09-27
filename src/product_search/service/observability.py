"""记录分阶段耗时；日志不写查询原文和商品详情。"""

import json
import logging
from contextlib import contextmanager
from time import perf_counter

logger = logging.getLogger('product_search')


@contextmanager
def timed(timings: dict, name: str):
    started = perf_counter()
    try:
        yield
    finally:
        timings[name] = round((perf_counter() - started) * 1000, 3)


def log_search(request_id, response):
    logger.info(json.dumps({
        'event': 'search', 'request_id': request_id,
        'release_id': response['release_id'], 'pipeline': response['pipeline'],
        'result_count': len(response['results']), 'timings_ms': response['timings_ms'],
    }, ensure_ascii=False))
