import copy
import unittest

from intervals import insert_interval, merge_intervals


class Hidden(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10]]), [[1, 6], [8, 10]])

    def test_touching(self):
        self.assertEqual(merge_intervals([[1, 2], [2, 3]]), [[1, 3]])

    def test_unsorted(self):
        self.assertEqual(merge_intervals([[8, 10], [1, 3], [2, 6]]), [[1, 6], [8, 10]])

    def test_nested(self):
        self.assertEqual(merge_intervals([[1, 10], [2, 3], [4, 5]]), [[1, 10]])

    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])

    def test_points(self):
        self.assertEqual(merge_intervals([[5, 5], [5, 5], [1, 1]]), [[1, 1], [5, 5]])

    def test_merge_does_not_mutate(self):
        data = [[1, 3], [2, 6]]
        before = copy.deepcopy(data)
        merge_intervals(data)
        self.assertEqual(data, before)

    def test_result_pairs_are_new(self):
        data = [[1, 3]]
        out = merge_intervals(data)
        out[0][1] = 99
        self.assertEqual(data, [[1, 3]])

    def test_insert(self):
        self.assertEqual(insert_interval([[1, 2], [3, 5], [6, 7], [8, 10]], [4, 8]), [[1, 2], [3, 10]])

    def test_insert_does_not_mutate(self):
        data = [[1, 2], [5, 6]]
        insert_interval(data, [3, 4])
        self.assertEqual(data, [[1, 2], [5, 6]])

    def test_insert_into_empty(self):
        self.assertEqual(insert_interval([], [2, 3]), [[2, 3]])


if __name__ == "__main__":
    unittest.main()
