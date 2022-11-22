import unittest

from processors.effect_estimate_extractor import EffectEstimateExtractor

extractor = EffectEstimateExtractor("../models/effect_estimate_classifier.pkl.bz2")


class TestEffectEstimateExtractor(unittest.TestCase):

    def test_empty(self):
        output = extractor.process([["blah", "blah"]])
        self.assertEqual(0, output["prediction"])
        # self.assertDictEqual({}, output["pages"])

    def test_simple_effect_estimate_present(self):
        output = extractor.process(["the study will have 80% power to detect a 0.42 log 10 increase in HIV-1 RNA by SCA using a two-sided Wilcoxon rank sum test at 5% level".split()])
        self.assertEqual(1, output["prediction"])
        # self.assertDictEqual({"0.42": 0}, output["pages"])
