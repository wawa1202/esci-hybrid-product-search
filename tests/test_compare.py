import unittest

from product_search.evaluation.compare import paired_bootstrap


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.compare = paired_bootstrap

    def test_pairing_direction_counts_and_exclusions(self):
        left = {'a': 0.25, 'b': 0.5, 'c': 0.5, 'd': None, 'e': 1.0}
        right = {'e': None, 'd': 0.0, 'c': 0.5, 'b': 0.25, 'a': 0.75}
        actual = self.compare(left, right)
        self.assertAlmostEqual(actual['mean_delta'], 1 / 12)
        self.assertEqual({k: actual[k] for k in ('wins', 'losses', 'ties', 'n', 'excluded')},
                         {'wins': 1, 'losses': 1, 'ties': 1, 'n': 3, 'excluded': 2})
        self.assertLessEqual(actual['ci95'][0], actual['ci95'][1])

    def test_constant_differences_and_single_pair(self):
        for left, right in [({'a': 0.0}, {'a': 0.25}),
                            ({'a': 0.0, 'b': 0.5}, {'a': 0.25, 'b': 0.75})]:
            actual = self.compare(left, right)
            self.assertEqual(actual['mean_delta'], 0.25)
            self.assertEqual(tuple(actual['ci95']), (0.25, 0.25))

    def test_percentile_bootstrap_reproduces_reference(self):
        import random

        differences = [-0.5, 0.25, 0.75]
        rng = random.Random(19)
        means = sorted(sum(differences[rng.randrange(3)] for _ in range(3)) / 3
                       for _ in range(41))
        actual = self.compare({'c': 0.0, 'b': 0.0, 'a': 0.5},
                              {'b': 0.25, 'a': 0.0, 'c': 0.75}, iterations=41, seed=19)
        self.assertEqual(tuple(actual['ci95']), (means[1], means[39]))

    def test_seed_and_dictionary_order_do_not_change_result(self):
        left = {'c': 0.5, 'a': 0.0, 'b': 0.25}
        right = {'a': 0.75, 'b': 0.0, 'c': 0.5}
        actual = self.compare(left, right, seed=12, iterations=63)
        self.assertEqual(actual, self.compare(dict(reversed(list(left.items()))),
                                             dict(reversed(list(right.items()))),
                                             seed=12, iterations=63))

    def test_empty_and_all_undefined(self):
        for left, right, excluded in [({}, {}, 0), ({'q': None}, {'q': 0.5}, 1)]:
            self.assertEqual(self.compare(left, right),
                             {'mean_delta': None, 'ci95': [None, None], 'wins': 0,
                              'losses': 0, 'ties': 0, 'n': 0, 'excluded': excluded})

    def test_rejects_mismatched_queries(self):
        with self.assertRaises(ValueError):
            self.compare({'a': None}, {'b': None})

    def test_rejects_invalid_values_even_in_excluded_pairs(self):
        for value in (float('nan'), float('inf'), -float('inf'), True, '0.5'):
            for left, right in [({'q': None}, {'q': value}), ({'q': value}, {'q': None})]:
                with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                    self.compare(left, right)

    def test_rejects_bad_parameters_and_overflow(self):
        for iterations in (0, -1, True, 1.5):
            with self.assertRaises((TypeError, ValueError)):
                self.compare({}, {}, iterations=iterations)
        with self.assertRaises((TypeError, ValueError)):
            self.compare({}, {}, seed=True)
        with self.assertRaises(ValueError):
            self.compare({'q': -1e308}, {'q': 1e308})

    def test_large_finite_equal_deltas_do_not_overflow_mean(self):
        actual = self.compare({'a': 0.0, 'b': 0.0}, {'a': 1e308, 'b': 1e308}, iterations=3)
        self.assertEqual(actual['mean_delta'], 1e308)
        self.assertEqual(actual['ci95'], [1e308, 1e308])
