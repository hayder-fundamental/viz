# TODO(HE): fix this make package.
import sys;
sys.path.append("../")
import unittest

import pandas as pd

import core


class MockRun:
    id = "foo"


class TestDownloader(unittest.TestCase):
    def setUp(self):
        self.run = MockRun()
        self.downloader = core.HistoryDownloader(_login=False)

    def test_read_write_cache(self):
        data = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [float("nan"), 0],
                "c": ["foo", "bar"],
                "d": ["baz", "0"],
            }
        )
        self.downloader.write_cache(self.run, data)
        read = self.downloader.read_cache(self.run)
        pd.testing.assert_frame_equal(read, data)
