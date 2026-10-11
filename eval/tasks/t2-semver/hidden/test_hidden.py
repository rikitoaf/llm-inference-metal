import unittest

from semver import compare, satisfies


class Hidden(unittest.TestCase):
    def test_compare_basic(self):
        self.assertEqual(compare("1.2.3", "1.2.4"), -1)
        self.assertEqual(compare("2.0.0", "1.9.9"), 1)
        self.assertEqual(compare("1.0.0", "1.0.0"), 0)

    def test_numeric_not_lexical(self):
        self.assertEqual(compare("1.10.0", "1.9.0"), 1)

    def test_prerelease_chain(self):
        chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2",
                 "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
        for lo, hi in zip(chain, chain[1:]):
            self.assertEqual(compare(lo, hi), -1, (lo, hi))
            self.assertEqual(compare(hi, lo), 1, (hi, lo))

    def test_build_metadata_ignored(self):
        self.assertEqual(compare("1.0.0+build.5", "1.0.0"), 0)
        self.assertEqual(compare("1.0.0-rc.1+x", "1.0.0-rc.1"), 0)

    def test_invalid(self):
        for bad in ["1.2", "1.2.x", "v1.2.3", "", "1.2.3.4", "a.b.c"]:
            with self.assertRaises(ValueError, msg=bad):
                compare(bad, "1.0.0")

    def test_caret(self):
        self.assertTrue(satisfies("1.5.0", "^1.2.3"))
        self.assertFalse(satisfies("2.0.0", "^1.2.3"))
        self.assertFalse(satisfies("1.2.2", "^1.2.3"))

    def test_caret_zero(self):
        self.assertTrue(satisfies("0.2.9", "^0.2.3"))
        self.assertFalse(satisfies("0.3.0", "^0.2.3"))
        self.assertTrue(satisfies("0.0.3", "^0.0.3"))
        self.assertFalse(satisfies("0.0.4", "^0.0.3"))

    def test_tilde(self):
        self.assertTrue(satisfies("1.2.9", "~1.2.3"))
        self.assertFalse(satisfies("1.3.0", "~1.2.3"))
        self.assertTrue(satisfies("1.2.0", "~1.2"))
        self.assertFalse(satisfies("1.3.0", "~1.2"))

    def test_operators(self):
        self.assertTrue(satisfies("1.2.3", "1.2.3"))
        self.assertTrue(satisfies("1.2.3", "=1.2.3"))
        self.assertFalse(satisfies("1.2.4", "=1.2.3"))
        self.assertTrue(satisfies("1.2.4", ">1.2.3"))
        self.assertFalse(satisfies("1.2.3", ">1.2.3"))
        self.assertTrue(satisfies("1.2.3", "<=1.2.3"))
        self.assertFalse(satisfies("1.2.3", "<1.2.3"))

    def test_and_range(self):
        self.assertTrue(satisfies("1.5.0", ">=1.2.0 <2.0.0"))
        self.assertFalse(satisfies("2.0.0", ">=1.2.0 <2.0.0"))
        self.assertFalse(satisfies("1.1.9", ">=1.2.0 <2.0.0"))

    def test_prerelease_excluded(self):
        self.assertFalse(satisfies("1.3.0-beta", ">=1.2.0"))
        self.assertFalse(satisfies("1.3.0-beta", "^1.2.3"))

    def test_prerelease_same_tuple(self):
        self.assertTrue(satisfies("1.2.4-rc.2", ">=1.2.4-rc.1 <1.3.0"))
        self.assertFalse(satisfies("1.2.5-rc.2", ">=1.2.4-rc.1 <1.3.0"))

    def test_invalid_range(self):
        with self.assertRaises(ValueError):
            satisfies("1.0.0", ">>1.0.0")


if __name__ == "__main__":
    unittest.main()
