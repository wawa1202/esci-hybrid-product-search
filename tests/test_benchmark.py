import unittest

from product_search.evaluation.benchmark import summarize_timings


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.summarize = summarize_timings

    def test_linear_percentiles_and_stage_counts_in_milliseconds(self):
        result = self.summarize([{'bm25': 0.0, 'dense': 20.0}, {'bm25': 10.0},
                                 {'bm25': 20.0}, {'bm25': 30.0, 'dense': 40.0}])
        self.assertEqual(result['bm25']['count'], 4)
        self.assertEqual(result['dense']['count'], 2)
        for stage, expected in [('bm25', (15.0, 28.5, 29.7)), ('dense', (30.0, 39.0, 39.8))]:
            for key, value in zip(('p50_ms', 'p95_ms', 'p99_ms'), expected):
                self.assertAlmostEqual(result[stage][key], value)

    def test_empty_missing_stages_and_singletons(self):
        self.assertEqual(self.summarize([]), {})
        self.assertEqual(self.summarize([{}, {}]), {})
        self.assertEqual(self.summarize([{}, {'total': 0.0}]),
                         {'total': {'count': 1, 'p50_ms': 0.0, 'p95_ms': 0.0, 'p99_ms': 0.0}})

    def test_rejects_nonfinite_negative_and_non_numeric(self):
        for value in (-0.1, float('nan'), float('inf'), -float('inf'), None, '2', True):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.summarize([{'bm25': value}])
        for stage in ('', ' ', 12):
            with self.assertRaises((ValueError, TypeError)):
                self.summarize([{stage: 1.0}])

    def test_order_and_inputs_are_preserved(self):
        samples = [{'stage': 3.0}, {'stage': 1.0}, {'stage': 2.0}]
        before = [dict(row) for row in samples]
        self.assertEqual(self.summarize(samples), self.summarize(samples[::-1]))
        self.assertEqual(samples, before)
