import unittest

from semver import compare, satisfies


class TestSemver(unittest.TestCase):
    def test_compare_basic(self):
        self.assertEqual(compare("1.2.3", "1.2.4"), -1)
        self.assertEqual(compare("2.0.0", "1.9.9"), 1)
        self.assertEqual(compare("1.0.0", "1.0.0"), 0)

    def test_caret(self):
        self.assertTrue(satisfies("1.5.0", "^1.2.3"))
        self.assertFalse(satisfies("2.0.0", "^1.2.3"))


if __name__ == "__main__":
    unittest.main()
