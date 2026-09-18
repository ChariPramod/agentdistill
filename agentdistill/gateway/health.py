"""Rolling health of the traffic the gateway is actually serving.

Two rates, over a sliding window: how often the student failed and the teacher answered instead (fallback), and
how often a request could not be placed in a cluster (unassigned). Either one high means the gateway is serving
every request "successfully" while doing something nobody chose -- billing the teacher, or routing blind. The
point is to say so on the first `/healthz` anyone makes, not after the invoice.

In memory and per process, deliberately: this is about the last few minutes of this gateway, and it must be fed
on every request, including the fallback path, or it reports healthy while everything escalates.
"""

from __future__ import annotations

import time
from collections import deque

#: Below this many requests in the window, a rate is noise and raises nothing.
MIN_REQUESTS = 20


class HealthTracker:
    def __init__(self, window_s: int = 300, fallback_alert: float = 0.2, unassigned_alert: float = 0.5):
        self.window_s, self.fallback_alert, self.unassigned_alert = window_s, fallback_alert, unassigned_alert
        self.events: deque[tuple[float, bool, bool]] = deque()     # (ts, fallback, unassigned)

    def record(self, fallback: bool, unassigned: bool, now: float | None = None) -> None:
        t = now if now is not None else time.time()
        self.events.append((t, fallback, unassigned))
        cut = t - self.window_s
        while self.events and self.events[0][0] < cut:
            self.events.popleft()

    def snapshot(self, now: float | None = None) -> dict:
        t = now if now is not None else time.time()
        cut = t - self.window_s
        rows = [e for e in self.events if e[0] >= cut]
        n = len(rows)
        fb = sum(1 for e in rows if e[1]) / n if n else 0.0
        un = sum(1 for e in rows if e[2]) / n if n else 0.0
        problems = []
        if n >= MIN_REQUESTS and fb > self.fallback_alert:
            problems.append(f"student fallback rate {fb:.0%} over {self.window_s}s")
        if n >= MIN_REQUESTS and un > self.unassigned_alert:
            problems.append(f"unassigned cluster rate {un:.0%}; cluster model may be missing")
        return {"ok": not problems, "requests": n, "fallback_rate": round(fb, 3), "unassigned_rate": round(un, 3),
                "problems": problems}
