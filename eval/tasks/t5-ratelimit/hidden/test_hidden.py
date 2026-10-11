import unittest

from ratelimit import SlidingWindow, TokenBucket


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class Hidden(unittest.TestCase):
    def test_bucket_burst(self):
        c = Clock()
        b = TokenBucket(1, 3, c)
        self.assertEqual([b.allow() for _ in range(4)], [True, True, True, False])

    def test_bucket_fractional_refill(self):
        c = Clock()
        b = TokenBucket(0.5, 1, c)
        self.assertTrue(b.allow())
        c.t += 1
        self.assertFalse(b.allow())
        c.t += 1
        self.assertTrue(b.allow())

    def test_bucket_small_steps_with_calls(self):
        c = Clock()
        b = TokenBucket(1, 1, c)
        self.assertTrue(b.allow())
        results = []
        for _ in range(4):
            c.t += 0.25
            results.append(b.allow())
        self.assertEqual(results, [False, False, False, True])

    def test_bucket_cap(self):
        c = Clock()
        b = TokenBucket(10, 2, c)
        c.t += 100
        self.assertEqual([b.allow() for _ in range(3)], [True, True, False])

    def test_bucket_n(self):
        c = Clock()
        b = TokenBucket(1, 5, c)
        self.assertTrue(b.allow(3))
        self.assertFalse(b.allow(3))
        self.assertTrue(b.allow(2))

    def test_bucket_failed_takes_nothing(self):
        c = Clock()
        b = TokenBucket(1, 2, c)
        self.assertTrue(b.allow(2))
        c.t += 1
        self.assertFalse(b.allow(2))
        self.assertTrue(b.allow(1))

    def test_bucket_validation(self):
        c = Clock()
        for args in [(0, 1), (-1, 1), (1, 0)]:
            with self.assertRaises(ValueError):
                TokenBucket(args[0], args[1], c)
        b = TokenBucket(1, 2, c)
        for n in [0, 3, -1]:
            with self.assertRaises(ValueError):
                b.allow(n)

    def test_window_limit(self):
        c = Clock()
        w = SlidingWindow(2, 10, c)
        self.assertEqual([w.allow("u") for _ in range(3)], [True, True, False])

    def test_window_keys_independent(self):
        c = Clock()
        w = SlidingWindow(1, 10, c)
        self.assertTrue(w.allow("a"))
        self.assertTrue(w.allow("b"))
        self.assertFalse(w.allow("a"))

    def test_window_boundary_half_open(self):
        c = Clock()
        w = SlidingWindow(1, 10, c)
        self.assertTrue(w.allow("u"))
        c.t += 9.999
        self.assertFalse(w.allow("u"))
        c.t = 110.0
        self.assertTrue(w.allow("u"))

    def test_window_rejected_not_recorded(self):
        c = Clock()
        w = SlidingWindow(2, 10, c)
        self.assertTrue(w.allow("u"))      # t=100
        c.t = 105
        self.assertTrue(w.allow("u"))      # t=105
        self.assertFalse(w.allow("u"))     # rejected at 105, must not count
        c.t = 110
        self.assertTrue(w.allow("u"))      # 100 left the window; only 105 remains

    def test_window_validation(self):
        c = Clock()
        for args in [(0, 10), (1, 0), (-1, 5), (1.5, 10)]:
            with self.assertRaises(ValueError):
                SlidingWindow(args[0], args[1], c)


if __name__ == "__main__":
    unittest.main()
