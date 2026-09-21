import logging
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.config import get_settings
from app.audit import record_resourceplus_call
from app.observability import current_voice_trace
from app.telemetry import (
    CORRELATION_HEADER,
    current_trace_id,
    emit_structured_event,
    record_resourceplus_metrics,
)


logger = logging.getLogger(__name__)


class ResourcePlusError(Exception):
    """Base error for ResourcePlus integration failures."""


class ResourcePlusTimeoutError(ResourcePlusError):
    """ResourcePlus did not respond before the configured timeout."""


class ResourcePlusConnectionError(ResourcePlusError):
    """ResourcePlus could not be reached."""


class ResourcePlusConfigurationError(ResourcePlusError):
    """Required backend-owned ResourcePlus configuration is missing."""


class ResourcePlusHTTPError(ResourcePlusError):
    """ResourcePlus returned a non-success HTTP response."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"ResourcePlus returned HTTP {status_code}.")


class ResourcePlusInvalidResponseError(ResourcePlusError):
    """ResourcePlus returned a response that was not valid JSON."""


class ResourcePlusClient:
    """Small async HTTP client restricted to backend-defined relative routes."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = self._normalize_base_url(base_url or settings.rp_base_url)
        self.timeout = httpx.Timeout(timeout_seconds or settings.rp_timeout_seconds)
        self.transport = transport

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        candidate = base_url.strip()
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RP_BASE_URL must be an absolute HTTP(S) URL.")
        return candidate.rstrip("/") + "/"

    def _build_url(self, route: str) -> str:
        parsed = urlsplit(route)
        if parsed.scheme or parsed.netloc or ".." in parsed.path.split("/"):
            raise ValueError("ResourcePlus routes must be safe relative paths.")
        return self.base_url + route.lstrip("/")

    async def _request(
        self,
        method: str,
        route: str,
        *,
        params: dict[str, str | int] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        url = self._build_url(route)
        safe_endpoint = urlsplit(route).path.lstrip("/")
        started_at = time.perf_counter()
        status: str | int = "error"
        error_category: str | None = None
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                transport=self.transport,
            ) as client:
                response = await client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=(
                        {CORRELATION_HEADER: trace_id}
                        if (trace_id := current_trace_id())
                        else None
                    ),
                )
                status = response.status_code
                if response.is_error:
                    error_category = "resourceplus_error"
        except httpx.TimeoutException as exc:
            status = "timeout"
            error_category = "resourceplus_timeout"
            logger.warning(
                "ResourcePlus request timed out for endpoint %s",
                safe_endpoint,
            )
            raise ResourcePlusTimeoutError(
                "ResourcePlus took too long to respond. Please try again."
            ) from exc
        except httpx.RequestError as exc:
            status = "connection_error"
            error_category = "resourceplus_error"
            logger.warning(
                "ResourcePlus connection failed for endpoint %s: %s",
                safe_endpoint,
                type(exc).__name__,
            )
            raise ResourcePlusConnectionError(
                "ResourcePlus is currently unreachable. Please try again."
            ) from exc
        finally:
            duration = time.perf_counter() - started_at
            trace = current_voice_trace()
            if trace is not None:
                trace.add_duration("resourceplus", duration)
            record_resourceplus_call(
                method=method,
                endpoint=safe_endpoint,
                status=status,
                duration=duration,
            )
            record_resourceplus_metrics(
                method=method,
                status=status,
                duration_ms=duration * 1000,
            )
            emit_structured_event(
                component="resourceplus_api",
                event="request_completed",
                endpoint=safe_endpoint,
                duration_ms=duration * 1000,
                status=status,
                error_category=error_category,
                error_owner="resourceplus_api" if error_category else None,
                error_stage="resourceplus" if error_category else None,
                retryable=True if error_category else None,
            )
            logger.info(
                "RP_API method=%s endpoint=%s duration=%.3fs status=%s",
                method.upper(),
                safe_endpoint,
                duration,
                status,
            )

        if response.is_error:
            emit_structured_event(
                level="WARNING",
                component="resourceplus_api",
                event="upstream_http_error",
                endpoint=safe_endpoint,
                status=response.status_code,
                error_category="resourceplus_error",
                error_owner="resourceplus_api",
                error_stage="resourceplus",
                retryable=response.status_code >= 500,
            )
            logger.warning(
                "ResourcePlus endpoint %s returned HTTP %s",
                safe_endpoint,
                response.status_code,
            )
            raise ResourcePlusHTTPError(response.status_code)

        try:
            return response.json()
        except ValueError as exc:
            logger.warning(
                "ResourcePlus endpoint %s returned invalid JSON",
                safe_endpoint,
            )
            raise ResourcePlusInvalidResponseError(
                "ResourcePlus returned an unreadable response. Please try again."
            ) from exc

    async def get(
        self,
        route: str,
        *,
        params: dict[str, str | int],
    ) -> Any:
        return await self._request("GET", route, params=params)

    async def post(
        self,
        route: str,
        *,
        params: dict[str, str | int],
        json_body: dict[str, Any],
    ) -> Any:
        return await self._request(
            "POST",
            route,
            params=params,
            json_body=json_body,
        )
