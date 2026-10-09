"""Contracts for the actual agent interface, rejection behavior and replay."""
import copy
import unittest
from guard_demo import guard, verify


class GuardContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.interval = guard({"cohort": "kirc", "gene": "VHL", "fields": {
            "assay_denominator": "cohort", "sample_context": "all", "mutation_scope": "broad"}})

    def test_interval_instead_of_unidentified_point(self):
        card = self.interval["certificate"]
        self.assertEqual(card["status"], "interval_answer")
        self.assertTrue(card["semantic_certificate"]["stable"])
        self.assertIsNone(card["answer"]["minimax_point"])
        self.assertEqual(card["answer"]["interval"], [166/512, 276/512])

    def test_missing_fields_require_clarification(self):
        card = guard({"cohort": "kirc", "gene": "VHL"})["certificate"]
        self.assertEqual(card["status"], "clarification_required")
        self.assertIsNone(card["answer"])
        self.assertIsNotNone(card["next_prompt"])

    def test_identified_profiled_fraction(self):
        card = guard({"cohort": "kirc", "gene": "VHL", "fields": {
            "analysis_unit": "patient", "assay_denominator": "profiled", "sample_context": "all", "mutation_scope": "broad"}})["certificate"]
        self.assertEqual(card["status"], "point_answer_with_certificate")
        self.assertEqual(card["answer"]["minimax_point"], 166/402)

    def test_full_replay(self):
        self.assertTrue(verify(self.interval, replay=True))

    def test_modified_answer_rejected(self):
        receipt = copy.deepcopy(self.interval)
        receipt["certificate"]["answer"]["interval"][1] = .4
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify(receipt)

    def test_modified_binding_rejected(self):
        receipt = copy.deepcopy(self.interval)
        receipt["bindings"]["snapshot_content_sha256"] = "0"*64
        with self.assertRaises(ValueError):
            verify(receipt)
        receipt = copy.deepcopy(self.interval)
        receipt["provenance"]["@graph"][0]["@id"] = "urn:changed-source"
        with self.assertRaises(ValueError):
            verify(receipt)

    def test_unsupported_fields_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported scientific field"):
            guard({"cohort": "kirc", "gene": "VHL", "fields": {"unit": "patient"}})

    def test_unsupported_field_value_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported value"):
            guard({"cohort": "kirc", "gene": "VHL", "fields": {"analysis_unit": "tumor"}})

    def test_nonfinite_budget_rejected(self):
        with self.assertRaises(ValueError):
            guard({"cohort": "kirc", "gene": "VHL", "point_error_budget_pp": float("nan")})


if __name__ == "__main__":
    unittest.main(verbosity=2)
