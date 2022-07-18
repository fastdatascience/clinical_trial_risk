import unittest

from processors.num_subjects_extractor import NumSubjectsExtractor

extractor = NumSubjectsExtractor("../num_subjects_classifier.pkl.bz2")


class TestNumSubjectsExtractor(unittest.TestCase):

    def test_empty(self):
        output = extractor.process([["blah", "blah"]])
        self.assertEqual(None, output["prediction"])
        self.assertDictEqual({}, output["pages"])

    def test_simple_num_subjects_present(self):
        output = extractor.process([["we", "will", "recruit", "1000", "subjects"]])
        self.assertEqual(1000, output["prediction"])
        self.assertDictEqual({"1000": 0}, output["pages"])
