import math
import unittest
from dataclasses import replace

from product_search.evaluation import metrics
from product_search.schemas import Judgment, Query, SearchResult


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = metrics
        self.query = Query('q', 'phone', 'us')

    def judgments(self, *labels):
        return [Judgment('q', str(i), 'us', label) for i, label in enumerate(labels)]

    def results(self, *ids):
        return [SearchResult('q', str(pid), 'us', i, float(-i))
                for i, pid in enumerate(ids, 1)]

    def test_roadmap_hand_calculation(self):
        actual = self.metrics.ranking_metrics(
            self.query, self.results(0, 1), self.judgments('I', 'E'), k=2)
        self.assertAlmostEqual(actual['ndcg'], 1 / math.log2(3))
        self.assertEqual(actual['mrr_e'], 0.5)

    def test_gains_and_ideal_use_all_judgments(self):
        labels = self.judgments('I', 'C', 'S', 'E')
        actual = self.metrics.ranking_metrics(self.query, self.results(1, 2), labels, k=2)
        self.assertAlmostEqual(actual['ndcg'],
                               (1 + 3 / math.log2(3)) / (7 + 3 / math.log2(3)))
        self.assertEqual(actual['mrr_e'], 0.0)

    def test_cutoff_and_rank_are_authoritative(self):
        labels = self.judgments('I', 'E')
        result = self.results(0, 1)
        self.assertEqual(self.metrics.ranking_metrics(self.query, result[::-1], labels, k=2),
                         self.metrics.ranking_metrics(self.query, result, labels, k=2))
        self.assertEqual(self.metrics.ranking_metrics(self.query, result, labels, k=1),
                         {'ndcg': 0.0, 'mrr_e': 0.0})

    def test_undefined_differs_from_no_retrieved_positive(self):
        for labels in ([], self.judgments('I')):
            self.assertEqual(self.metrics.ranking_metrics(self.query, [], labels),
                             {'ndcg': None, 'mrr_e': None})
        self.assertEqual(self.metrics.ranking_metrics(self.query, [], self.judgments('E')),
                         {'ndcg': 0.0, 'mrr_e': 0.0})
        actual = self.metrics.ranking_metrics(self.query, self.results(0), self.judgments('C'))
        self.assertEqual(actual, {'ndcg': 1.0, 'mrr_e': None})

    def test_unknown_rejected_even_below_ranking_cutoff(self):
        with self.assertRaises(ValueError):
            self.metrics.ranking_metrics(self.query, self.results(0, 'unknown'),
                                         self.judgments('E'), k=1)

    def test_retrieval_keeps_unknown_and_uses_returned_count(self):
        labels = self.judgments('E', 'E', 'S', 'I')
        actual = self.metrics.retrieval_metrics(self.query, self.results(0, 'unknown', 2), labels)
        self.assertEqual(actual['known_e_recall'], 0.5)
        self.assertAlmostEqual(actual['known_es_recall'], 2 / 3)
        self.assertAlmostEqual(actual['judged_fraction'], 2 / 3)
        actual = self.metrics.retrieval_metrics(self.query, self.results('unknown', 0), labels, k=1)
        self.assertEqual(actual, {'known_e_recall': 0.0, 'known_es_recall': 0.0,
                                 'judged_fraction': 0.0})

    def test_retrieval_empty_denominators(self):
        actual = self.metrics.retrieval_metrics(self.query, [], [])
        self.assertEqual(actual, {'known_e_recall': None, 'known_es_recall': None,
                                 'judged_fraction': None})
        actual = self.metrics.retrieval_metrics(self.query, [], self.judgments('E'))
        self.assertEqual(actual, {'known_e_recall': 0.0, 'known_es_recall': 0.0,
                                 'judged_fraction': None})
        actual = self.metrics.retrieval_metrics(self.query, self.results(0), self.judgments('S'))
        self.assertEqual(actual, {'known_e_recall': None, 'known_es_recall': 1.0,
                                 'judged_fraction': 1.0})

    def test_default_cutoffs(self):
        labels = self.judgments(*(['I'] * 100 + ['E']))
        results = self.results(*range(101))
        self.assertEqual(self.metrics.ranking_metrics(self.query, results, labels)['mrr_e'], 0.0)
        self.assertEqual(self.metrics.retrieval_metrics(self.query, results, labels)['known_e_recall'], 0.0)

    def test_rejects_invalid_results_and_judgments_in_both_modes(self):
        good_result = self.results(0)[0]
        good_label = self.judgments('E')[0]
        cases = [
            ([good_result, replace(good_result, rank=2)], [good_label]),
            ([replace(good_result, rank=2)], [good_label]),
            ([good_result, replace(good_result, product_id='1')], [good_label]),
            ([replace(good_result, rank=True)], [good_label]),
            ([replace(good_result, query_id='other')], [good_label]),
            ([replace(good_result, locale='jp')], [good_label]),
            ([replace(good_result, score=float('nan'))], [good_label]),
            ([replace(good_result, score=float('inf'))], [good_label]),
            ([replace(good_result, score=-float('inf'))], [good_label]),
            ([replace(good_result, score=1)], [good_label]),
            ([good_result], [good_label, good_label]),
            ([good_result], [good_label, replace(good_label, esci_label='I')]),
            ([good_result], [replace(good_label, locale='jp')]),
            ([good_result], [replace(good_label, query_id='other')]),
            ([good_result], [replace(good_label, esci_label='X')]),
            ([replace(good_result, product_id='')], [good_label]),
        ]
        for function in (self.metrics.ranking_metrics, self.metrics.retrieval_metrics):
            for results, labels in cases:
                with self.subTest(function=function.__name__, results=results, labels=labels):
                    with self.assertRaises((ValueError, TypeError)):
                        function(self.query, results, labels, k=1)
            for k in (0, -1, True, 1.5):
                with self.subTest(k=k), self.assertRaises((ValueError, TypeError)):
                    function(self.query, [], [], k=k)
            with self.assertRaises(ValueError):
                function(replace(self.query, locale='xx'), [], [])

    def test_does_not_mutate_inputs(self):
        results = self.results(0, 1)[::-1]
        labels = self.judgments('I', 'E')
        before = (list(results), list(labels))
        self.metrics.ranking_metrics(self.query, results, labels)
        self.metrics.retrieval_metrics(self.query, results, labels)
        self.assertEqual((results, labels), before)
