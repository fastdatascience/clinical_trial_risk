import unittest

from util.get_phase_tertiles import get_tertile
from util.tertile_provider import DefaultSampleSizeTertileProvider

tertile_provider = DefaultSampleSizeTertileProvider("../sample_size_tertiles.csv")


class TestTertileFinder(unittest.TestCase):

    def test_phase_1(self):
        self.assertEqual([40, 130],
                         get_tertile(tertile_provider, "HIV", 1))

    def test_phase_1_5(self):
        self.assertEqual([40, 80],
                         get_tertile(tertile_provider, "TB", 1.5))
