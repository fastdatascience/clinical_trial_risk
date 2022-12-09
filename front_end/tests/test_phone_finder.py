import unittest

from util.phone_number_finder import find_phone_numbers


class TestPhaseExtractor(unittest.TestCase):

    def test_empty(self):
        output = find_phone_numbers("this is a trial trial")
        self.assertEqual(0, len(output))

    def test_simple_uk(self):
        output = find_phone_numbers("Please call +44 48415 15445 1"
                                    )
        self.assertEqual(1, len(output))
