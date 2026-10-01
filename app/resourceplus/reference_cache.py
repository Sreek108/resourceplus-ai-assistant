from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Awaitable, Callable

from app.config import get_settings
from app.identity import current_request_identity
from app.resourceplus.attendance import get_exception_reasons
from app.resourceplus.leave import get_day_types


@dataclass(frozen=True)
class _CacheEntry:
    value: Any
    expires_at: float


class ReferenceDataCache:
    """Small process-local cache for non-employee ResourcePlus master data."""

    def __init__(self, *, ttl_seconds: float, max_entries: int = 32) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str, int], _CacheEntry] = OrderedDict()
        self._lock = RLock()

    async def get_or_load(
        self,
        key: tuple[str, str, int],
        loader: Callable[[], Awaitable[Any]],
        *,
        force_refresh: bool = False,
    ) -> Any:
        now = monotonic()
        if not force_refresh:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and now < entry.expires_at:
                    self._entries.move_to_end(key)
                    return deepcopy(entry.value)
                if entry is not None:
                    self._entries.pop(key, None)

        value = await loader()
        with self._lock:
            self._entries[key] = _CacheEntry(
                value=deepcopy(value),
                expires_at=monotonic() + self.ttl_seconds,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return deepcopy(value)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_settings = get_settings()
reference_data_cache = ReferenceDataCache(
    ttl_seconds=_settings.reference_data_cache_ttl_seconds,
    max_entries=_settings.reference_data_cache_max_entries,
)


async def cached_day_types(lang: int, *, force_refresh: bool = False) -> Any:
    identity = current_request_identity()
    return await reference_data_cache.get_or_load(
        ("day_types", identity.instance.casefold(), lang),
        lambda: get_day_types(lang=lang),
        force_refresh=force_refresh,
    )


async def cached_exception_reasons(lang: int, *, force_refresh: bool = False) -> Any:
    identity = current_request_identity()
    return await reference_data_cache.get_or_load(
        ("exception_reasons", identity.instance.casefold(), lang),
        lambda: get_exception_reasons(lang=lang),
        force_refresh=force_refresh,
    )
