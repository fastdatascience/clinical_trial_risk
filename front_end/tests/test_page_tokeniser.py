import unittest

from util.page_tokeniser import tokenise_pages


class TestTok(unittest.TestCase):

    def test_fraction(self):
        self.assertListEqual([["with", "effect", "size", "=", "0.3"]],
                             tokenise_pages(["with effect size=0.3"]))

    def test_percentage(self):
        self.assertListEqual([["effect", "size", "be", "0.4%", "based", "on"]],
                             tokenise_pages(["effect size be 0.4% based on"]))

    def test_3dp(self):
        self.assertListEqual([["The", "estimated", "effect", "size", "is", "therefore", "set", "to", "0.325", "HAZ"]],
                             tokenise_pages(["The estimated effect size is therefore set to 0.325 HAZ"]))

    def test_95percent_ci(self):
        self.assertListEqual([["Primary", "analysis", "is", "to", "calculate", "the", "2", "sided", "95%CI", "for",
                               "the", "difference", "in", "proportion"]],
                             tokenise_pages([
                                                "Primary analysis is to calculate the 2-sided 95%CI for the difference in proportion"]))

    def test_five_fold(self):
        self.assertListEqual(
            [["study", "has", "80%", "power", "to", "detect", "a", "five", "fold", "increase", "in", "this", "SAE"]],
            tokenise_pages(["study has 80% power to detect a five-fold increase in this SAE."]))

    def test_delta(self):
        self.assertListEqual(
            [["y", "identifying", "ATP", "for", "a", "regimen", "is", "at", "least", "80%", "when", "δ", "=", "24%"]],
            tokenise_pages(["y identifying ATP for a regimen is at least 80% when δ=24%"]))

    def test_phase(self):
        self.assertListEqual([["Phase", "I", "II"]],
                             tokenise_pages(["Phase I/II"]))

    def test_commas(self):
        self.assertListEqual([["sample", "size", "of", "1,200"]],
                             tokenise_pages(["sample size of 1,200"]))
