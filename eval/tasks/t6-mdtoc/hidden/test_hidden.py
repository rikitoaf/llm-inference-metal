import unittest

from mdtoc import toc


class Hidden(unittest.TestCase):
    def test_basic(self):
        md = "# Title\n\n## Install\n\ntext\n\n## Usage\n"
        self.assertEqual(toc(md), "- [Title](#title)\n  - [Install](#install)\n  - [Usage](#usage)")

    def test_empty(self):
        self.assertEqual(toc("no headings here\n"), "")

    def test_not_a_heading(self):
        self.assertEqual(toc("#hashtag\n####### seven\n # indented? no\n"), "")

    def test_indent_relative_to_min(self):
        self.assertEqual(toc("## A\n### B\n## C\n"), "- [A](#a)\n  - [B](#b)\n- [C](#c)")

    def test_max_level(self):
        self.assertEqual(toc("# A\n## B\n### C\n#### D\n", max_level=2), "- [A](#a)\n  - [B](#b)")

    def test_slug_punctuation(self):
        self.assertEqual(toc("## What's new? (v2.0)\n"), "- [What's new? (v2.0)](#whats-new-v20)")

    def test_slug_keeps_hyphen_underscore(self):
        self.assertEqual(toc("## snake_case and kebab-case\n"),
                         "- [snake_case and kebab-case](#snake_case-and-kebab-case)")

    def test_duplicates(self):
        md = "# Notes\n## Notes\n## Notes\n"
        self.assertEqual(toc(md), "- [Notes](#notes)\n  - [Notes](#notes-1)\n  - [Notes](#notes-2)")

    def test_duplicates_count_hidden_levels(self):
        md = "# API\n#### API\n## API\n"
        self.assertEqual(toc(md), "- [API](#api)\n  - [API](#api-2)")

    def test_closing_hashes(self):
        self.assertEqual(toc("## Setup ##\n"), "- [Setup](#setup)")

    def test_inline_formatting(self):
        md = "## Use `pip` and **uv** with *care* via [docs](https://x.y)\n"
        self.assertEqual(toc(md), "- [Use pip and uv with care via docs](#use-pip-and-uv-with-care-via-docs)")

    def test_fences_ignored(self):
        md = "# Real\n```bash\n# not a heading\n```\n~~~\n## also not\n~~~\n## Also real\n"
        self.assertEqual(toc(md), "- [Real](#real)\n  - [Also real](#also-real)")

    def test_fence_types_do_not_cross(self):
        md = "# A\n```\n~~~\n# hidden\n```\n# B\n"
        self.assertEqual(toc(md), "- [A](#a)\n- [B](#b)")

    def test_multiple_spaces(self):
        self.assertEqual(toc("##   Spaced   Out  \n"), "- [Spaced   Out](#spaced---out)")


if __name__ == "__main__":
    unittest.main()
