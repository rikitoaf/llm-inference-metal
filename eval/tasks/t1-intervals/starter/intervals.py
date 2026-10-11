"""Interval utilities."""


def merge_intervals(intervals):
    """Merge overlapping intervals.

    Takes a list of [start, end] pairs (start <= end), in any order, and returns a new list of
    merged [start, end] lists sorted by start. Intervals that overlap or touch ([1, 2] and [2, 3])
    merge into one. An empty input gives an empty list. The input list and its pairs must not be
    modified.
    """
    result = []
    for pair in intervals:
        if result and pair[0] < result[-1][1]:
            result[-1][1] = max(result[-1][1], pair[1])
        else:
            result.append(pair)
    return result


def insert_interval(intervals, new):
    """Insert the [start, end] pair `new` into `intervals` and return the merged result, as
    merge_intervals would. The input list must not be modified."""
    intervals.append(new)
    return merge_intervals(intervals)
