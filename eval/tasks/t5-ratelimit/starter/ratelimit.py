"""Rate limiters. Both take an injectable clock: a zero-argument function returning seconds."""


class TokenBucket:
    """Token bucket.

    Starts full with `capacity` tokens and refills continuously at `rate` tokens per second, never
    above `capacity`; fractional tokens accumulate (a rate of 0.5 gives one token every 2 s).
    allow(n=1) takes n tokens and returns True if at least n are available; otherwise it returns
    False and takes nothing. rate and capacity must be positive numbers and n an int in
    1..capacity, otherwise ValueError.
    """

    def __init__(self, rate, capacity, clock):
        self.rate = rate
        self.capacity = capacity
        self.clock = clock
        self.tokens = capacity
        self.last = clock()

    def allow(self, n=1):
        now = self.clock()
        self.tokens += int((now - self.last) * self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


class SlidingWindow:
    """Sliding-window log limiter.

    Allows at most `limit` calls per key within any `window` seconds. allow(key) returns True and
    records the call if fewer than `limit` calls for that key were recorded in the half-open
    interval (now - window, now]; otherwise it returns False and records nothing. Keys are
    independent. limit must be a positive int and window a positive number, otherwise ValueError.
    """

    def __init__(self, limit, window, clock):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.calls = []

    def allow(self, key):
        now = self.clock()
        self.calls.append(now)
        recent = [t for t in self.calls if t >= now - self.window]
        return len(recent) <= self.limit
