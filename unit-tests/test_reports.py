import unittest
from td3.reports import Report


class TestReport(unittest.TestCase):
    def setUp(self):
        self.report = Report()

    def test_report_step(self):
        for i in range(99):
            self.report.error_list.append([i, i, 0.1, 0.1, 0.1, False, 0.1])

        # self.assertEqual(self.report., )


