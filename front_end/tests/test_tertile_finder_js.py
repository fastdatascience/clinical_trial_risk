import json
import re
import unittest

import js2py

from util.clientside_callbacks import JAVASCRIPT_FUNCTION_TO_CALCULATE_SAMPLE_SIZE_TERTILE
from util.tertile_provider import DefaultSampleSizeTertileProvider

tertile_provider = DefaultSampleSizeTertileProvider("../sample_size_tertiles.csv")


def get_tertile(num_subjects, condition, phase, data, columns):
    result = js2py.eval_js(
        re.sub("function", "function xxx",
               JAVASCRIPT_FUNCTION_TO_CALCULATE_SAMPLE_SIZE_TERTILE) + f";\nxxx({num_subjects}, '{condition}', {phase}, {data}, {columns});\n")
    return json.loads(str(result[2]))


class TestTertileFinderJs(unittest.TestCase):

    def test_phase_1(self):
        result = get_tertile(1, "HIV", 1, tertile_provider.DF_TERTILES_DATA_FOR_DASH,
                             tertile_provider.DF_TERTILES_COLUMNS_FOR_DASH)
        self.assertEqual([1, 0, 40, 130],
                         result)

    def test_phase_1_5(self):
        result = get_tertile(1, "TB", 1.5, tertile_provider.DF_TERTILES_DATA_FOR_DASH,
                             tertile_provider.DF_TERTILES_COLUMNS_FOR_DASH)

        self.assertEqual([1, 0, 40, 80],
                         result)
