from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

from config.settings import settings


def _key(messages: list[dict], params: dict) -> str:
    payload = json.dumps({"m": messages, "p": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LRUCache:
    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, messages: list[dict], params: dict) -> Any | None:
        k = _key(messages, params)
        with self._lock:
            if k not in self._store:
                return None
            self._store.move_to_end(k)
            return self._store[k]

    def set(self, messages: list[dict], params: dict, value: Any) -> None:
        k = _key(messages, params)
        with self._lock:
            self._store[k] = value
            self._store.move_to_end(k)
            while len(self._store) > self._max:
                self._store.popitem(last=False)


cache = LRUCache(max_entries=settings.cache_max_entries)
