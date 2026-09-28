import logging
import re
import time
from dataclasses import dataclass
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

_DIAGNOSTIC_ENVIRONMENTS = frozenset({"local", "dev", "development", "test", "uat"})
_CORRELATION_HEADERS = (
    "x-correlation-id",
    "x-request-id",
    "request-id",
    "trace-id",
)
_SAFE_DIAGNOSTIC_ID = re.compile(r"^[A-Za-z0-9_./:-]{1,128}$")
_SENSITIVE_TEXT = re.compile(
    r"(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|https?://|"
    r"authorization|bearer|password|api[_ -]?key|secret|token)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResourcePlusUpstreamDiagnostic:
    """Allowlisted metadata extracted from an upstream error response."""

    code: str | None = None
    message: str | None = None
    correlation_id: str | None = None
    validation_fields: tuple[str, ...] = ()


def _safe_diagnostic_id(value: object) -> str | None:
    candidate = str(value).strip() if isinstance(value, (str, int)) else ""
    return candidate if _SAFE_DIAGNOSTIC_ID.fullmatch(candidate) else None


def _safe_diagnostic_message(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = " ".join(value.split())
    if not candidate or len(candidate) > 240 or _SENSITIVE_TEXT.search(candidate):
        return None
    if any(ord(character) < 32 for character in candidate):
        return None
    return candidate


def _first_mapping_value(payload: dict[str, Any], *names: str) -> object:
    normalized = {str(key).casefold(): value for key, value in payload.items()}
    for name in names:
        if name.casefold() in normalized:
            return normalized[name.casefold()]
    return None


def _extract_upstream_diagnostic(
    response: httpx.Response,
) -> ResourcePlusUpstreamDiagnostic:
    """Extract only short allowlisted fields; never retain or log the raw body."""

    payload: dict[str, Any] = {}
    try:
        parsed = response.json()
        if isinstance(parsed, dict):
            payload = parsed
    except ValueError:
        pass

    nested_error = _first_mapping_value(payload, "error")
    nested = nested_error if isinstance(nested_error, dict) else {}
    code = _safe_diagnostic_id(
        _first_mapping_value(payload, "errorCode", "code")
        or _first_mapping_value(nested, "errorCode", "code")
    )
    message = _safe_diagnostic_message(
        _first_mapping_value(payload, "message", "title", "detail")
        or _first_mapping_value(nested, "message", "title", "detail")
        or (nested_error if isinstance(nested_error, str) else None)
    )

    validation = _first_mapping_value(payload, "errors", "validationErrors")
    if not isinstance(validation, dict):
        validation = _first_mapping_value(nested, "errors", "validationErrors")
    validation_fields = tuple(
        field
        for field in (
            _safe_diagnostic_id(key)
            for key in (validation.keys() if isinstance(validation, dict) else ())
        )
        if field is not None
    )[:12]

    correlation_id = next(
        (
            safe_value
            for header in _CORRELATION_HEADERS
            if (safe_value := _safe_diagnostic_id(response.headers.get(header)))
            is not None
        ),
        None,
    )
    return ResourcePlusUpstreamDiagnostic(
        code=code,
        message=message,
        correlation_id=correlation_id,
        validation_fields=validation_fields,
    )


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

    def __init__(
        self,
        status_code: int,
        *,
        endpoint: str | None = None,
        diagnostic: ResourcePlusUpstreamDiagnostic | None = None,
    ) -> None:
        self.status_code = status_code
        self.endpoint = endpoint
        self.diagnostic = diagnostic or ResourcePlusUpstreamDiagnostic()
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
            diagnostic = _extract_upstream_diagnostic(response)
            expose_diagnostic = (
                get_settings().app_environment.casefold() in _DIAGNOSTIC_ENVIRONMENTS
            )
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
                upstream_error_code=(
                    diagnostic.code if expose_diagnostic else None
                ),
                upstream_error_message=(
                    diagnostic.message if expose_diagnostic else None
                ),
                upstream_correlation_id=(
                    diagnostic.correlation_id if expose_diagnostic else None
                ),
                validation_fields=(
                    diagnostic.validation_fields if expose_diagnostic else ()
                ),
            )
            logger.warning(
                "ResourcePlus endpoint %s returned HTTP %s",
                safe_endpoint,
                response.status_code,
            )
            raise ResourcePlusHTTPError(
                response.status_code,
                endpoint=safe_endpoint,
                diagnostic=diagnostic,
            )

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
