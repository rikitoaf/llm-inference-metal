import unittest

from ratelimit import SlidingWindow, TokenBucket


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestRateLimit(unittest.TestCase):
    def test_bucket_burst(self):
        c = Clock()
        b = TokenBucket(1, 3, c)
        self.assertEqual([b.allow() for _ in range(4)], [True, True, True, False])

    def test_window_limit(self):
        c = Clock()
        w = SlidingWindow(2, 10, c)
        self.assertEqual([w.allow("u") for _ in range(3)], [True, True, False])


if __name__ == "__main__":
    unittest.main()
