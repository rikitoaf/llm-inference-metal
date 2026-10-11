import unittest

from deps import CycleError, resolve


class Hidden(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(resolve({"app": ["lib", "log"], "lib": ["log"]}), ["log", "lib", "app"])

    def test_empty(self):
        self.assertEqual(resolve({}), [])

    def test_only_mentioned(self):
        self.assertEqual(resolve({"a": ["z", "m"]}), ["m", "z", "a"])

    def test_alphabetical_tiebreak(self):
        deps = {"d": [], "c": [], "b": ["d"], "a": ["d"]}
        self.assertEqual(resolve(deps), ["c", "d", "a", "b"])

    def test_ready_order_not_global_sort(self):
        # smallest *ready* package first: 'b' waits for 'z' although it sorts before 'y'
        deps = {"b": ["z"], "y": [], "z": []}
        self.assertEqual(resolve(deps), ["y", "z", "b"])

    def test_duplicates(self):
        self.assertEqual(resolve({"a": ["b", "b"], "b": []}), ["b", "a"])

    def test_diamond(self):
        deps = {"top": ["left", "right"], "left": ["base"], "right": ["base"], "base": []}
        self.assertEqual(resolve(deps), ["base", "left", "right", "top"])

    def test_cycle_path(self):
        with self.assertRaises(CycleError) as cm:
            resolve({"b": ["c"], "c": ["a"], "a": ["b"]})
        self.assertEqual(cm.exception.cycle, ["a", "b", "c", "a"])

    def test_self_cycle(self):
        with self.assertRaises(CycleError) as cm:
            resolve({"x": ["x"], "a": []})
        self.assertEqual(cm.exception.cycle, ["x", "x"])

    def test_cycle_behind_dependency(self):
        with self.assertRaises(CycleError) as cm:
            resolve({"app": ["q"], "q": ["r"], "r": ["s"], "s": ["q"]})
        self.assertEqual(cm.exception.cycle, ["q", "r", "s", "q"])

    def test_two_cycles(self):
        with self.assertRaises(CycleError) as cm:
            resolve({"m": ["n"], "n": ["m"], "d": ["e"], "e": ["f"], "f": ["d"]})
        self.assertEqual(cm.exception.cycle, ["d", "e", "f", "d"])

    def test_message(self):
        with self.assertRaises(CycleError) as cm:
            resolve({"a": ["b"], "b": ["a"]})
        self.assertEqual(str(cm.exception), "dependency cycle: a -> b -> a")


if __name__ == "__main__":
    unittest.main()
