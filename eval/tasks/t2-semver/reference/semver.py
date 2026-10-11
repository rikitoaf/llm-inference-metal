import re

_V = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z.-]+)?$")


def _parse(v):
    m = _V.match(v or "")
    if not m:
        raise ValueError(v)
    pre = tuple(m[4].split(".")) if m[4] else ()
    return (int(m[1]), int(m[2]), int(m[3])), pre


def _cmp_pre(a, b):
    if a == b:
        return 0
    if not a:
        return 1
    if not b:
        return -1
    for x, y in zip(a, b):
        if x == y:
            continue
        xn, yn = x.isdigit(), y.isdigit()
        if xn and yn:
            return -1 if int(x) < int(y) else 1
        if xn != yn:
            return -1 if xn else 1
        return -1 if x < y else 1
    return -1 if len(a) < len(b) else 1


def _cmp(pa, pb):
    if pa[0] != pb[0]:
        return -1 if pa[0] < pb[0] else 1
    return _cmp_pre(pa[1], pb[1])


def compare(a, b):
    return _cmp(_parse(a), _parse(b))


def _comparators(token):
    m = re.match(r"^(>=|<=|>|<|=|~|\^)?(.*)$", token)
    op, rest = m[1] or "=", m[2]
    if op == "~" and re.match(r"^\d+\.\d+$", rest):
        rest += ".0"
    base = _parse(rest)
    (ma, mi, pa), pre = base
    if op == "~":
        return [(">=", base), ("<", ((ma, mi + 1, 0), ("0",)))]
    if op == "^":
        if ma > 0:
            hi = (ma + 1, 0, 0)
        elif mi > 0:
            hi = (0, mi + 1, 0)
        else:
            hi = (0, 0, pa + 1)
        return [(">=", base), ("<", (hi, ("0",)))]
    return [(op, base)]


def satisfies(version, range_):
    v = _parse(version)
    comps = [c for tok in range_.split() for c in _comparators(tok)]
    if not comps:
        raise ValueError(range_)
    for op, c in comps:
        r = _cmp(v, c)
        if not {"=": r == 0, ">": r > 0, ">=": r >= 0, "<": r < 0, "<=": r <= 0}[op]:
            return False
    if v[1]:
        return any(c[1] and c[1] != ("0",) and c[0] == v[0] for _, c in comps)
    return True
