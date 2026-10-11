"""Dependency ordering.

resolve(deps) -> list[str]
    deps maps each package name to a list of the packages it depends on. Return every package,
    both the keys and names that only appear as dependencies, exactly once, in an order where
    each package comes after all of its dependencies. Whenever several packages are ready at the
    same time, the alphabetically smallest one comes first, so the result is deterministic:
    resolve({'app': ['lib', 'log'], 'lib': ['log']}) == ['log', 'lib', 'app'].
    Duplicate entries in a dependency list are allowed and ignored.

    A cycle (including a package depending on itself) raises CycleError. Its .cycle attribute is
    the cycle as a list that starts and ends with the same package, where each package depends on
    the next one, starting from the alphabetically smallest package of the cycle:
    {'b': ['c'], 'c': ['a'], 'a': ['b']} gives ['a', 'b', 'c', 'a']; {'x': ['x']} gives
    ['x', 'x']. If there are several separate cycles, report the one containing the
    alphabetically smallest package that is on any cycle.
"""


class CycleError(Exception):
    def __init__(self, cycle):
        super().__init__("dependency cycle: " + " -> ".join(cycle))
        self.cycle = cycle


def resolve(deps):
    raise NotImplementedError
