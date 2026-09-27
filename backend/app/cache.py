"""Small thread-safe TTL + LRU cache for API responses."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Hashable


class TTLCache:
    def __init__(self, maxsize: int = 2048, ttl_s: float = 24 * 3600):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            ts, value = item
            if time.time() - ts > self.ttl_s:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def get_or_set(self, key: Hashable, fn: Callable[[], Any]) -> tuple[Any, bool]:
        v = self.get(key)
        if v is not None:
            return v, True
        v = fn()
        self.set(key, v)
        return v, False

    def stats(self) -> dict:
        return {"size": len(self._data), "maxsize": self.maxsize, "hits": self.hits, "misses": self.misses}
