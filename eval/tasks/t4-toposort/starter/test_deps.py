import unittest

from deps import CycleError, resolve


class TestDeps(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(resolve({"app": ["lib", "log"], "lib": ["log"]}), ["log", "lib", "app"])

    def test_cycle_raises(self):
        with self.assertRaises(CycleError):
            resolve({"a": ["b"], "b": ["a"]})


if __name__ == "__main__":
    unittest.main()
