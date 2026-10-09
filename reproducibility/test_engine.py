"""Contract tests for scientific counting and clarification logic."""
from itertools import combinations, permutations
import unittest
import pandas as pd
from engine import Spec, Result, PandasEngine, SQLiteEngine, divergence, stable, sensitive, ClarificationPlanner


class CountingContracts(unittest.TestCase):
    def setUp(self):
        # p1 has a primary negative specimen and a metastatic positive specimen;
        # p2 is profiled primary positive; p3 is unprofiled. Duplicate records
        # and a silent mutation deliberately challenge a naive counting engine.
        self.samples = pd.DataFrame([
            ("x", "a", "p1", True, True),
            ("x", "b", "p1", False, True),
            ("x", "c", "p2", True, True),
            ("x", "d", "p3", True, False),
        ], columns=["cohort", "sample", "patient", "primary", "profiled"])
        self.mutations = pd.DataFrame([
            ("x", "b", "TP53", "Missense_Mutation"),
            ("x", "b", "TP53", "Missense_Mutation"),
            ("x", "c", "TP53", "Nonsense_Mutation"),
            ("x", "a", "TP53", "Silent"),
            ("x", "a", "TP53", "Intron"),
            ("x", "a", "TP53", "Splice_Region"),
            ("x", "d", "TP53", "Nonsense_Mutation"),
        ], columns=["cohort", "sample", "gene", "mutation_type"])
        self.engines = [PandasEngine(self.samples, self.mutations),
                        SQLiteEngine(self.samples, self.mutations)]

    def check(self, result, **kwargs):
        for engine in self.engines:
            self.assertEqual(engine.execute(Spec("x", "TP53", **kwargs)), result)

    def test_patient_union_and_duplicate_deduplication(self):
        self.check(Result(2, 2, 2))

    def test_sample_unit(self):
        self.check(Result(2, 3, 3), analysis_unit="sample")

    def test_primary_context_does_not_inherit_metastatic_mutation(self):
        self.check(Result(1, 2, 2), sample_context="primary")

    def test_unprofiled_positive_cannot_enter_numerator(self):
        self.check(Result(2, 3, 2), assay_denominator="cohort")

    def test_truncating_scope_and_silent_exclusion(self):
        self.check(Result(1, 2, 2), mutation_scope="truncating")

    def test_missingness_bound(self):
        value = self.engines[0].execute(Spec("x", "TP53", assay_denominator="cohort"))
        self.assertAlmostEqual(value.prevalence, 2 / 3)
        self.assertAlmostEqual(value.missingness_upper_bound, 1.0)

    def test_zero_mutation_gene(self):
        for engine in self.engines:
            self.assertEqual(engine.execute(Spec("x", "VHL")), Result(0, 2, 2))

    def test_empty_denominator_is_undefined(self):
        for engine in self.engines:
            result = engine.execute(Spec("absent", "TP53"))
            self.assertEqual(result, Result(0, 0, 0))
            self.assertIsNone(result.prevalence)

    def test_invalid_enum_fails(self):
        with self.assertRaises(ValueError):
            Spec("x", "TP53", mutation_scope="all_mutations")


class DecisionContracts(unittest.TestCase):
    def test_zero_pair_stable(self):
        self.assertFalse(sensitive(0.0, 0.0))

    def test_absolute_or_branch(self):
        # Relative difference is under 10%, but absolute difference exceeds 5 pp.
        self.assertLess(divergence(0.80, 0.87)[1], 0.10)
        self.assertTrue(sensitive(0.80, 0.87))

    def test_zero_vs_positive_sensitive(self):
        self.assertTrue(sensitive(0.0, 0.001))

    def test_extrema_equal_all_pairs(self):
        groups = [[0.0, 0.0, 0.0], [0.2, 0.201, 0.21], [0.8, 0.84, 0.87],
                  [0.0, 0.00001, 0.4], [0.03, 0.2, 0.8]]
        for values in groups:
            self.assertEqual(not stable(values), any(sensitive(a, b) for a, b in combinations(values, 2)))

    def test_undefined_candidate_cannot_certify_stability(self):
        self.assertFalse(stable([0.1, None]))

    def test_planner_uniform_expectation_and_stable_terminal_states(self):
        values = [0.1 + 0.30 * bits[3] for bits in __import__('engine').ASSIGNMENTS]
        planner = ClarificationPlanner(values)
        self.assertEqual(planner.solve(planner.root)[0], 1.0)
        self.assertEqual(planner.solve(planner.root)[1], 3)
        self.assertTrue(all(planner.walk(i)[0] == 1 for i in range(16)))
        self.assertLessEqual(planner.solve(planner.root)[0],
                             min(sum(planner.walk(i, order)[0] for i in range(16)) / 16
                                 for order in permutations(range(4))))

    def test_joint_sensitivity_can_be_missed_by_one_axis_checks(self):
        # Each single change shifts 0.02 from the default 0.50 (stable), while
        # two simultaneous changes shift 0.04 (stable) and four shift 0.08.
        values = [0.5 + 0.02 * sum(bits) for bits in __import__('engine').ASSIGNMENTS]
        self.assertFalse(any(sensitive(values[0], values[1 << b]) for b in range(4)))
        self.assertFalse(stable(values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
