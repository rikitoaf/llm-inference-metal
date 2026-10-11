import heapq


class CycleError(Exception):
    def __init__(self, cycle):
        super().__init__("dependency cycle: " + " -> ".join(cycle))
        self.cycle = cycle


def resolve(deps):
    graph = {k: set(v) for k, v in deps.items()}
    for vs in list(graph.values()):
        for v in vs:
            graph.setdefault(v, set())
    pending = {k: len(v) for k, v in graph.items()}
    users = {k: [] for k in graph}
    for k, vs in graph.items():
        for v in vs:
            users[v].append(k)
    ready = [k for k, n in pending.items() if n == 0]
    heapq.heapify(ready)
    out = []
    while ready:
        k = heapq.heappop(ready)
        out.append(k)
        for u in users[k]:
            pending[u] -= 1
            if pending[u] == 0:
                heapq.heappush(ready, u)
    if len(out) == len(graph):
        return out
    left = {k for k in graph if k not in set(out)}
    # nodes on a cycle: those that can reach themselves within `left`
    def reaches(start):
        seen, stack = set(), [start]
        while stack:
            n = stack.pop()
            for d in graph[n]:
                if d == start:
                    return True
                if d in left and d not in seen:
                    seen.add(d)
                    stack.append(d)
        return False
    start = min(k for k in left if reaches(k))
    path, n = [start], start
    while True:
        nxt = min(d for d in graph[n] if d in left and (d == start or reaches(d)))
        path.append(nxt)
        if nxt == start:
            raise CycleError(path)
        n = nxt
