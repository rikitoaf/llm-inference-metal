from collections import defaultdict, deque


class TokenBucket:
    def __init__(self, rate, capacity, clock):
        if not rate > 0 or not capacity > 0:
            raise ValueError("rate and capacity must be positive")
        self.rate, self.capacity, self.clock = rate, capacity, clock
        self.tokens = float(capacity)
        self.last = clock()

    def allow(self, n=1):
        if not isinstance(n, int) or n < 1 or n > self.capacity:
            raise ValueError(n)
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens + 1e-9 >= n:
            self.tokens -= n
            return True
        return False


class SlidingWindow:
    def __init__(self, limit, window, clock):
        if not isinstance(limit, int) or limit <= 0 or not window > 0:
            raise ValueError("bad limit/window")
        self.limit, self.window, self.clock = limit, window, clock
        self.calls = defaultdict(deque)

    def allow(self, key):
        now = self.clock()
        q = self.calls[key]
        while q and q[0] <= now - self.window:
            q.popleft()
        if len(q) < self.limit:
            q.append(now)
            return True
        return False
