import unittest

from processors.num_subjects_extractor import NumSubjectsExtractor

extractor = NumSubjectsExtractor("../models/num_subjects_classifier.pkl.bz2")


class TestNumSubjectsExtractor(unittest.TestCase):

    def test_empty(self):
        output = extractor.process([["blah", "blah"]])
        self.assertEqual(0, output["prediction"])
        self.assertDictEqual({}, output["pages"])

    def test_simple_num_subjects_present(self):
        output = extractor.process([["we", "will", "recruit", "1000", "subjects"]])
        self.assertEqual(1000, output["prediction"])
        self.assertDictEqual({"1000": [0]}, output["pages"])

    def test_2500(self):
        output = extractor.process(["""Target enrollment is 2500 participants .""".split()])
        self.assertEqual(2500, output["prediction"])
        self.assertDictEqual({"2500": [0]}, output["pages"])

    def test_exclude_contents_page(self):
        output = extractor.process(["""Study population 31 6.1 subject inclusion criteria""".split()])
        self.assertEqual(0, output["prediction"])
        self.assertDictEqual({}, output["pages"])
