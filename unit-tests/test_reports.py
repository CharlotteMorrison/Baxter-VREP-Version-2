import unittest
from td3.reports import Reports


class TestReports(unittest.TestCase):

    def setUp(self):
        pass

    def test_report_step(self):
        report = Reports()
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()
