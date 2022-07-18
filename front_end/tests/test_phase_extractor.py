import unittest

from processors.phase_extractor import PhaseExtractor

extractor = PhaseExtractor()


class TestPhaseExtractor(unittest.TestCase):

    def test_empty(self):
        output = extractor.process([["this", "is", "a", "blah", "trial"]])
        self.assertEqual(0, output["prediction"])
        self.assertDictEqual({}, output["pages"])

    def test_simple_phase_i_present(self):
        output = extractor.process([["this", "is", "a", "phase", "i", "trial"]])
        self.assertEqual(1, output["prediction"])
        self.assertDictEqual({"Phase 1": 0}, output["pages"])

    def test_simple_phase_i_ii_present(self):
        output = extractor.process([["this", "is", "a", "phase", "i", "ii", "trial"]])
        self.assertEqual(1.5, output["prediction"])
        self.assertDictEqual({"Phase 1.5": 0}, output["pages"])
