from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from app.config import get_settings


_LOCAL_FALLBACK_ENVIRONMENTS = frozenset({"local", "test"})


class RequestIdentityError(ValueError):
    """The demo/UAT request identity is absent, partial, or malformed."""


@dataclass(frozen=True, repr=False, eq=False)
class RequestIdentity:
    """Backend-owned identity context for one ResourcePlus request."""

    email: str
    instance: str

    def __post_init__(self) -> None:
        email = self.email.strip() if isinstance(self.email, str) else ""
        instance = self.instance.strip() if isinstance(self.instance, str) else ""
        if not email or len(email) > 320 or "@" not in email:
            raise RequestIdentityError("A valid demo user email is required.")
        if not instance or len(instance) > 128:
            raise RequestIdentityError("A valid demo ResourcePlus instance is required.")
        object.__setattr__(self, "email", email)
        object.__setattr__(self, "instance", instance)

    @property
    def ownership_key(self) -> tuple[str, str]:
        """Compare owners case-insensitively without changing outbound identity."""

        return self.email.casefold(), self.instance

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RequestIdentity):
            return NotImplemented
        return self.ownership_key == other.ownership_key

    def __hash__(self) -> int:
        return hash(self.ownership_key)


_request_identity: ContextVar[RequestIdentity | None] = ContextVar(
    "resourceplus_request_identity",
    default=None,
)


def resolve_request_identity(
    email: str | None,
    instance: str | None,
) -> RequestIdentity:
    """Resolve an all-request or all-default identity pair; never mix the two."""

    if (email is None) != (instance is None):
        raise RequestIdentityError(
            "Demo user email and ResourcePlus instance must be supplied together."
        )
    if email is not None and instance is not None:
        return RequestIdentity(email=email, instance=instance)

    settings = get_settings()
    environment = settings.app_environment.strip().casefold()
    if environment not in _LOCAL_FALLBACK_ENVIRONMENTS:
        raise RequestIdentityError(
            "Demo user email and ResourcePlus instance are required in this environment."
        )
    default_email = settings.rp_default_email
    default_instance = settings.rp_instance
    if not default_email or not default_instance:
        raise RequestIdentityError(
            "Demo identity is required because the local fallback is incomplete."
        )
    return RequestIdentity(email=default_email, instance=default_instance)


def current_request_identity() -> RequestIdentity:
    identity = _request_identity.get()
    return identity if identity is not None else resolve_request_identity(None, None)


def bind_request_identity(identity: RequestIdentity) -> Token[RequestIdentity | None]:
    return _request_identity.set(identity)


def reset_request_identity(token: Token[RequestIdentity | None]) -> None:
    _request_identity.reset(token)


def resourceplus_identity(
    *,
    email: str | None = None,
    instance: str | None = None,
) -> RequestIdentity:
    """Use an explicit complete pair for tests, otherwise the current request pair."""

    if email is None and instance is None:
        return current_request_identity()
    return resolve_request_identity(email, instance)
