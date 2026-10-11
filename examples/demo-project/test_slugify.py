import unittest

from slugify import slugify


class SlugifyTest(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_collapses_separators(self):
        self.assertEqual(slugify("Hello,   World!!"), "hello-world")

    def test_accents(self):
        self.assertEqual(slugify("Crème Brûlée"), "creme-brulee")

    def test_max_len_does_not_end_with_hyphen(self):
        self.assertEqual(slugify("one two three", max_len=8), "one-two")


if __name__ == "__main__":
    unittest.main()
