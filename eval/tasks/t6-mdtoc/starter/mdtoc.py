"""Markdown table of contents.

toc(markdown, max_level=3) -> str
    One line per ATX heading (1 to 6 '#' characters at the start of a line followed by a space)
    whose level is <= max_level, in document order, formatted as
        '<indent>- [<text>](#<slug>)'
    and joined with '\\n', with no trailing newline ('' when there are no such headings).

    indent  two spaces per level below the smallest heading level that appears in the TOC.
    text    the heading text with any closing '#' sequence and surrounding whitespace removed,
            and inline formatting stripped: `code` -> code, **bold** -> bold, *em* -> em,
            [label](url) -> label.
    slug    computed from text: lowercase; remove every character that is not a letter, digit,
            space, hyphen or underscore; then replace each space with '-'. A slug that was already
            used earlier in the document gets '-1' appended, the next repeat '-2', and so on
            (GitHub style). Every heading of every level counts towards repeats, including
            headings deeper than max_level that are not listed.

    Lines inside fenced code blocks are ignored. A fence opens with a line starting with ``` or
    ~~~ and closes with the next line starting with the same three characters.
"""


def toc(markdown, max_level=3):
    raise NotImplementedError
