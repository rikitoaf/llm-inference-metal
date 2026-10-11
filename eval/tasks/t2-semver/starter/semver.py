"""Semantic version comparison: a subset of semver 2.0.0 and npm-style ranges.

compare(a, b) -> -1, 0 or 1
    Versions are MAJOR.MINOR.PATCH (non-negative integers, no leading 'v') with an optional
    pre-release after '-', e.g. 1.2.3-alpha.1. Build metadata after '+' is ignored, so
    1.0.0+build.5 equals 1.0.0. A pre-release sorts before its release (1.0.0-alpha < 1.0.0).
    Pre-release identifiers are compared left to right: numeric identifiers numerically,
    alphanumeric ones in ASCII order, a numeric identifier lower than an alphanumeric one, and
    when all shared identifiers are equal the shorter list is lower:
    1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-alpha.beta < 1.0.0-beta < 1.0.0-beta.2 < 1.0.0-beta.11
    < 1.0.0-rc.1 < 1.0.0.
    Anything else (e.g. '1.2', '1.2.x', 'v1.2.3', '') raises ValueError.

satisfies(version, range_) -> bool
    range_ is one or more comparators separated by spaces; all of them must hold.
    Comparators:
        '1.2.3' or '=1.2.3'   equal
        '>1.2.3' '>=1.2.3' '<1.2.3' '<=1.2.3'
        '~1.2.3'  means >=1.2.3 <1.3.0
        '~1.2'    means >=1.2.0 <1.3.0
        '^1.2.3'  means >=1.2.3 <2.0.0
        '^0.2.3'  means >=0.2.3 <0.3.0
        '^0.0.3'  means >=0.0.3 <0.0.4
    A version with a pre-release satisfies a range only if at least one comparator in the range
    itself has a pre-release with the same MAJOR.MINOR.PATCH as the version (as in npm), so
    1.3.0-beta does not satisfy '>=1.2.0', but 1.2.4-rc.2 satisfies '>=1.2.4-rc.1 <1.3.0'.
    Invalid versions or comparators raise ValueError.
"""


def compare(a, b):
    raise NotImplementedError


def satisfies(version, range_):
    raise NotImplementedError
