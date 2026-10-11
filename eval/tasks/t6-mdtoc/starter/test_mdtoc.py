import unittest

from mdtoc import toc


class TestToc(unittest.TestCase):
    def test_basic(self):
        md = "# Title\n\n## Install\n\ntext\n\n## Usage\n"
        self.assertEqual(toc(md), "- [Title](#title)\n  - [Install](#install)\n  - [Usage](#usage)")


if __name__ == "__main__":
    unittest.main()
