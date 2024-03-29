import unittest

from processors.num_arms_extractor import NumArmsExtractor

extractor = NumArmsExtractor()


class TestNumArmsExtractor(unittest.TestCase):

    def test_empty(self):
        output = extractor.process([["blah", "blah"]])
        self.assertEqual(None, output["prediction"])
        self.assertDictEqual({}, output["pages"])

    def test_simple_num_arms_present_word(self):
        output = extractor.process([["this", "study", "has", "four", "experimental", "arms"]])
        self.assertEqual(4, output["prediction"])
        self.assertDictEqual({"four experimental arms": [0]}, output["pages"])

    def test_simple_num_arms_present_numeric(self):
        output = extractor.process([["this", "study", "has", "4", "experimental", "arms"]])
        self.assertEqual(4, output["prediction"])
        self.assertDictEqual({"4": [0]}, output["pages"])
