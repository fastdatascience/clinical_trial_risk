import unittest

from processors.phase_extractor_rule_based import PhaseExtractorRuleBased

extractor = PhaseExtractorRuleBased("../models/phase_rf_classifier.pkl.bz2")


class TestPhaseExtractor(unittest.TestCase):

    def test_empty(self):
        output = extractor.process([["this", "is", "a", "blah", "trial"]])
        self.assertEqual(0, output["prediction"])
        self.assertDictEqual({}, output["pages"])

    def test_simple_phase_i_present(self):
        output = extractor.process([["this", "is", "a", "phase", "i", "trial"]])
        self.assertEqual(1, output["prediction"])
        self.assertDictEqual({"Phase 1": [0]}, output["pages"])

    def test_simple_phase_i_ii_present(self):
        output = extractor.process([["this", "is", "a", "phase", "i", "ii", "trial"]])
        self.assertEqual(1.5, output["prediction"])
        self.assertDictEqual({"Phase 1.5": [0]}, output["pages"])

    def test_viral_decay(self):
        output = extractor.process([["this", "is", "a", "phase", "i", "viral", "decay"]])
        self.assertEqual(0, output["prediction"])
        self.assertDictEqual({}, output["pages"])
