"""两种评估口径分别输出；所有方法在相同查询与候选集上比较。"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from product_search.artifacts import read_json, sha256, verify_dataset, write_json
from product_search.contracts import Hit, get_query_key
from product_search.data.store import get_product, iter_judgments, iter_queries, open_database
from product_search.evaluation.benchmark import summarize_timings
from product_search.evaluation.compare import paired_bootstrap
from product_search.evaluation.error_analysis import analyze_errors
from product_search.evaluation.metrics import ranking_metrics, retrieval_metrics
from product_search.retrieval.fusion import fuse_hits
from product_search.schemas import SearchResult
from product_search.service.search import SearchEngine


def _results(query, scores):
    """既用于离线候选打分，也用于把融合结果统一成指标输入。"""
    return [SearchResult(query.query_id, key[1], query.locale, i, float(score))
            for i, (key, score) in enumerate(scores, 1)]


def _ordered(scores):
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _aggregate(rows):
    output = {}
    for method, values in rows.items():
        metrics = {}
        for name in sorted({name for row in values for name in row}):
            if name in {'k', 'query_id', 'locale'}:
                continue
            valid = [row[name] for row in values if row.get(name) is not None and isinstance(row[name], (float, int))]
            metrics[name] = {'mean': mean(valid) if valid else None, 'defined': len(valid), 'undefined': len(values) - len(valid)}
        output[method] = metrics
    return output


def evaluate(settings, dataset_dir, release_dir, output_dir, *, partition='validation', allow_test=False,
             embedder=None) -> Path:
    if partition not in {'validation', 'test'}:
        raise ValueError('评估仅允许 validation 或明确冻结后的 test')
    if partition == 'test' and not allow_test:
        raise ValueError('测试集必须显式 --allow-test，且先 freeze')
    dataset_dir, release_dir, output_dir = map(Path, (dataset_dir, release_dir, output_dir))
    manifest = verify_dataset(dataset_dir)
    # 新建目录拒绝覆盖，保留每一次评估证据。
    output_dir.mkdir(parents=True, exist_ok=False)
    with SearchEngine(settings, release_dir, embedder=embedder) as engine:
        if manifest['dataset_id'] != engine.manifest['dataset_id']:
            raise ValueError('评估数据与发布包不同')
        if partition == 'test':
            from product_search.evaluation.freeze import claim_test_run
            claim_test_run(settings, release_dir, output_dir)
        per_query, retrieval_rows, timing_rows = [], [], []
        rank_aggregate, recall_aggregate = defaultdict(list), defaultdict(list)
        with open_database(dataset_dir / 'labels.sqlite', read_only=True) as labels:
            with (output_dir / 'retrieval-results.jsonl').open('x', encoding='utf-8') as retrieval_stream:
                for stored in iter_queries(labels, partition=partition):
                    query = stored.query
                    judgments = list(iter_judgments(labels, get_query_key(query)))
                    keys = [(j.locale, j.product_id) for j in judgments]
                    products = [get_product(engine.catalog, key) for key in keys]
                    if any(p is None for p in products):
                        raise ValueError('人工候选引用了不存在的商品')
                    vector = engine.embedder.encode_queries([query])[0]
                    bm25_scores = engine.bm25.score_pairs(query, keys)
                    dense_scores = engine.vector.score_pairs(vector, keys)
                    candidate_hits = {}
                    for source, scores in [('bm25', bm25_scores), ('dense', dense_scores)]:
                        candidate_hits[source] = [Hit(get_query_key(query), key, source, i, float(score))
                                                 for i, (key, score) in enumerate(_ordered(scores), 1)]
                    fused = fuse_hits(candidate_hits.values(), constant=settings.retrieval.rrf_constant)
                    rankings = {source: _results(query, [(h.product_key, h.score) for h in hits])
                                for source, hits in candidate_hits.items()}
                    rankings['rrf'] = _results(query, [(c.product_key, c.rrf_score) for c in fused])
                    if engine.ranker:
                        rankings['ltr'] = engine.ranker.predict(query, products, bm25_scores=bm25_scores,
                                                                dense_scores=dense_scores)
                    row = {'query_id': query.query_id, 'text': query.text, 'locale': query.locale,
                           'metrics': {}, 'rankings': {}, 'labels': {j.product_id: j.esci_label for j in judgments}}
                    for method, results in rankings.items():
                        metrics = ranking_metrics(query, results, judgments, k=settings.evaluation.ranking_k)
                        row['metrics'][method] = metrics
                        row['rankings'][method] = [r.product_id for r in results]
                        rank_aggregate[method].append(metrics)
                    per_query.append(row)
                    # 商品库召回与上述人工候选排序彻底分开。
                    start = perf_counter()
                    bm_hits = engine.bm25.search(query, settings.bm25.top_k)
                    bm_end = perf_counter()
                    dense_hits = engine.vector.search(query, vector, k=settings.retrieval.dense_top_k,
                                                      block_size=settings.retrieval.vector_block_size)
                    dense_end = perf_counter()
                    fused = fuse_hits([bm_hits, dense_hits], constant=settings.retrieval.rrf_constant,
                                      k=settings.retrieval.rerank_top_k)
                    recalled = {'bm25': _results(query, [(h.product_key, h.score) for h in bm_hits]),
                                'dense': _results(query, [(h.product_key, h.score) for h in dense_hits]),
                                'rrf': _results(query, [(c.product_key, c.rrf_score) for c in fused])}
                    record = {'query_id': query.query_id, 'metrics': {}, 'rankings': {}}
                    for method, results in recalled.items():
                        metrics = retrieval_metrics(query, results, judgments, k=settings.evaluation.retrieval_k)
                        metrics['empty_rate'] = float(not results)
                        record['metrics'][method] = metrics
                        record['rankings'][method] = [r.product_id for r in results]
                        recall_aggregate[method].append(metrics)
                    retrieval_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    retrieval_rows.append(record)
                    timing_rows.append({'bm25': (bm_end - start) * 1000, 'dense': (dense_end - bm_end) * 1000})
                    if len(per_query) % 50 == 0:
                        print(f'评估 {partition}: {len(per_query)} queries', flush=True)
        comparisons = {}
        for method in ('dense', 'rrf', 'ltr'):
            if method not in rank_aggregate:
                continue
            comparisons[f'{method}_minus_bm25'] = paired_bootstrap(
                {r['query_id']: r['metrics']['bm25']['ndcg'] for r in per_query},
                {r['query_id']: r['metrics'][method]['ndcg'] for r in per_query},
                iterations=settings.evaluation.bootstrap_iterations, seed=settings.project.seed)
        report = {'release_id': engine.manifest['release_id'], 'dataset_id': manifest['dataset_id'],
                  'partition': partition, 'queries': len(per_query), 'catalog_scope': settings.data.catalog_scope,
                  'ranking_k': settings.evaluation.ranking_k, 'retrieval_k': settings.evaluation.retrieval_k,
                  'ranking': _aggregate(rank_aggregate), 'retrieval': _aggregate(recall_aggregate),
                  'comparisons': comparisons,
                  'limitations': ['人工候选集排序指标不能代表全商品库端到端效果',
                                 '未标注商品是未知，不是 I；known recall 只对已知相关标签计算',
                                 'dev 是人工候选商品并集，不能宣称完整 Amazon 商品库效果']}
        write_json(output_dir / 'ranking-results.json', report)
        with (output_dir / 'ranking-per-query.jsonl').open('x', encoding='utf-8') as stream:
            for row in per_query:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        with (output_dir / 'error-analysis.jsonl').open('x', encoding='utf-8') as stream:
            for row in analyze_errors(per_query, limit=min(30, len(per_query))):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        write_json(output_dir / 'retrieval-timings.json', {'scope': '检索耗时，不含查询编码/HTTP/排序',
                                                        'timings': summarize_timings(timing_rows)})
    return output_dir
