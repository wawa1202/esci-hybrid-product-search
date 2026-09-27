import copy
import unittest

from product_search.evaluation.error_analysis import analyze_errors


def row(query_id='q'):
    return {'query_id': query_id, 'text': 'phone',
            'metrics': {name: {'ndcg': 1.0, 'mrr_e': 1.0}
                        for name in ('bm25', 'dense', 'rrf', 'ltr')},
            'rankings': {name: ['e', 'i'] for name in ('bm25', 'dense', 'rrf', 'ltr')},
            'labels': {'e': 'E', 'i': 'I'}}


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.analyze = analyze_errors

    def test_irrelevant_ahead_of_later_exact_is_still_an_inversion(self):
        data = row()
        data['labels']['e2'] = 'E'
        for name in data['rankings']:
            data['rankings'][name] = ['e', 'e2', 'i']
        data['rankings']['ltr'] = ['e', 'i', 'e2']
        result = self.analyze([data])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['categories'], ['ltr:exact_below_irrelevant'])
        self.assertEqual(result[0]['evidence']['ltr']['first_e_rank'], 1)
        self.assertEqual(result[0]['evidence']['ltr']['i_before_e'], ['i'])

    def test_reports_verifiable_rank_failures_and_deltas(self):
        data = row()
        data['rankings']['bm25'] = []
        data['rankings']['dense'] = ['unknown', 'i']
        data['rankings']['ltr'] = ['i', 'e']
        data['metrics']['bm25'] = {'ndcg': 0.0, 'mrr_e': 0.0}
        data['metrics']['dense'] = {'ndcg': None, 'mrr_e': None}
        data['metrics']['ltr'] = {'ndcg': 0.63, 'mrr_e': 0.5}
        result = self.analyze([data])[0]
        self.assertEqual(result['query_id'], 'q')
        self.assertEqual(result['text'], 'phone')
        self.assertTrue({'bm25:empty_ranking', 'dense:unknown_candidates',
                         'dense:known_e_missing', 'ltr:exact_below_irrelevant',
                         'ltr:ndcg_regression_vs_rrf', 'ltr:mrr_e_regression_vs_rrf'}
                        <= set(result['categories']))
        self.assertAlmostEqual(result['deltas']['ltr_minus_rrf']['ndcg'], -0.37)
        self.assertEqual(result['deltas']['ltr_minus_rrf']['mrr_e'], -0.5)
        self.assertIsNone(result['deltas']['dense_minus_bm25']['ndcg'])
        self.assertEqual(result['evidence']['ltr']['first_e_rank'], 2)
        self.assertEqual(result['evidence']['ltr']['i_before_e'], ['i'])
        self.assertEqual(result['evidence']['dense']['unknown_products'], ['unknown'])
        self.assertEqual(result['evidence']['dense']['missing_known_e'], ['e'])

    def test_unknown_is_not_irrelevant_and_no_semantic_diagnosis(self):
        data = row()
        data['rankings']['ltr'] = ['unknown', 'e']
        result = self.analyze([data])[0]
        self.assertEqual(result['categories'], ['ltr:unknown_candidates'])
        self.assertEqual(result['evidence']['ltr']['i_before_e'], [])
        self.assertEqual(result['text'], data['text'])

    def test_no_failures_and_undefined_metrics(self):
        self.assertEqual(self.analyze([row()]), [])
        data = row()
        data['labels'] = {'i': 'I'}
        for name in data['rankings']:
            data['rankings'][name] = ['i']
            data['metrics'][name] = {'ndcg': None, 'mrr_e': None}
        self.assertEqual(self.analyze([data]), [])

    def test_limit_prioritizes_regression_and_has_stable_ties(self):
        rows = [row('b'), row('a'), row('worst')]
        for data in rows:
            data['metrics']['ltr']['ndcg'] = 0.75
        rows[-1]['metrics']['ltr']['ndcg'] = 0.1
        self.assertEqual([x['query_id'] for x in self.analyze(rows, limit=2)], ['worst', 'a'])
        self.assertEqual(self.analyze(rows, limit=0), [])
        self.assertEqual(self.analyze(rows), self.analyze(rows[::-1]))
        self.assertEqual(len(self.analyze([dict(copy.deepcopy(rows[0]), query_id=str(i))
                                          for i in range(31)])), 30)

    def test_rejects_duplicate_queries_rankings_and_invalid_labels(self):
        with self.assertRaises(ValueError):
            self.analyze([row(), row()])
        data = row()
        data['rankings']['ltr'] = ['e', 'e']
        with self.assertRaises(ValueError):
            self.analyze([data])
        data = row()
        data['labels']['e'] = 'X'
        with self.assertRaises(ValueError):
            self.analyze([data])

    def test_rejects_bad_metrics_fields_and_limits_even_beyond_limit(self):
        for value in (float('nan'), float('inf'), -0.1, 1.1, True, '1'):
            data = row('bad')
            data['metrics']['ltr']['ndcg'] = value
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.analyze([row(), data], limit=0)
        for field in ('metrics', 'rankings', 'labels', 'query_id', 'text'):
            data = row()
            del data[field]
            with self.assertRaises(ValueError):
                self.analyze([data])
        for limit in (-1, True, 0.5):
            with self.assertRaises((ValueError, TypeError)):
                self.analyze([], limit=limit)

    def test_does_not_mutate_inputs(self):
        data = row()
        data['rankings']['ltr'] = []
        before = copy.deepcopy(data)
        self.analyze([data])
        self.assertEqual(data, before)
