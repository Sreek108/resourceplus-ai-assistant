from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Callable, Protocol
from uuid import uuid4

from app.config import get_settings
from app.identity import RequestIdentity, current_request_identity


HistoryItem = dict[str, str]


@dataclass(frozen=True)
class PendingAction:
    confirmation_id: str
    action_type: str
    validated_arguments: dict[str, object]
    summary: str
    language: str
    created_at: datetime
    expires_at: datetime
    owner: RequestIdentity = field(repr=False)


@dataclass(frozen=True)
class ExceptionalEntryDraft:
    """Short-lived selection context that cannot execute a ResourcePlus write."""

    attendance_date: str
    entry_type: str
    suggested_entry_time: str
    language: str
    created_at: datetime
    expires_at: datetime
    reason_options: tuple[str, ...] = ()


@dataclass
class _SessionState:
    session_id: str
    owner: RequestIdentity = field(repr=False)
    history: list[HistoryItem] = field(default_factory=list)
    pending_action: PendingAction | None = None
    exceptional_entry_draft: ExceptionalEntryDraft | None = None
    exceptional_entry_history_start: int | None = None
    expired_pending_language: str | None = None
    expired_pending_action_type: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PendingActionExpired(Exception):
    pass


class PendingActionMismatch(Exception):
    pass


class SessionIdentityMismatch(Exception):
    pass


class SessionStore(Protocol):
    def ensure_session(self, session_id: str | None = None) -> str: ...

    def create_pending_action(
        self,
        session_id: str,
        *,
        action_type: str,
        validated_arguments: dict[str, object],
        summary: str,
        language: str,
    ) -> PendingAction: ...

    def get_pending_action(
        self,
        session_id: str,
    ) -> tuple[PendingAction | None, bool]: ...

    def create_exceptional_entry_draft(
        self,
        session_id: str,
        *,
        attendance_date: str,
        entry_type: str,
        suggested_entry_time: str,
        language: str,
        reason_options: list[str] | tuple[str, ...] | None = None,
    ) -> ExceptionalEntryDraft: ...

    def get_exceptional_entry_draft(
        self,
        session_id: str,
    ) -> ExceptionalEntryDraft | None: ...

    def clear_exceptional_entry_draft(
        self,
        session_id: str,
    ) -> ExceptionalEntryDraft | None: ...

    def get_expired_pending_language(self, session_id: str) -> str | None: ...

    def get_expired_pending_action_type(self, session_id: str) -> str | None: ...

    def consume_pending_action(
        self,
        session_id: str,
        confirmation_id: str | None = None,
    ) -> PendingAction: ...

    def discard_pending_action(self, session_id: str) -> PendingAction | None: ...

    def get_history(self, session_id: str) -> list[HistoryItem]: ...

    def append_history(self, session_id: str, role: str, content: str) -> None: ...


class InMemorySessionStore:
    """Replaceable, process-local session and confirmation storage for the demo."""

    def __init__(
        self,
        *,
        confirmation_ttl_seconds: int = 300,
        session_ttl_seconds: int = 1_800,
        now: Callable[[], datetime] | None = None,
        history_limit: int = 12,
    ) -> None:
        self.confirmation_ttl = timedelta(seconds=confirmation_ttl_seconds)
        self.session_ttl = timedelta(seconds=session_ttl_seconds)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.history_limit = history_limit
        self._sessions: dict[str, _SessionState] = {}
        self._lock = RLock()

    @staticmethod
    def _clone_action(action: PendingAction) -> PendingAction:
        return PendingAction(
            confirmation_id=action.confirmation_id,
            action_type=action.action_type,
            validated_arguments=deepcopy(action.validated_arguments),
            summary=action.summary,
            language=action.language,
            created_at=action.created_at,
            expires_at=action.expires_at,
            owner=action.owner,
        )

    @staticmethod
    def _clone_draft(draft: ExceptionalEntryDraft) -> ExceptionalEntryDraft:
        return ExceptionalEntryDraft(
            attendance_date=draft.attendance_date,
            entry_type=draft.entry_type,
            suggested_entry_time=draft.suggested_entry_time,
            language=draft.language,
            created_at=draft.created_at,
            expires_at=draft.expires_at,
            reason_options=draft.reason_options,
        )

    def _remove_stale_sessions(self, current: datetime) -> None:
        stale = [
            session_id
            for session_id, state in self._sessions.items()
            if current - state.updated_at > self.session_ttl
        ]
        for session_id in stale:
            self._sessions.pop(session_id, None)

    def _owned_state(self, session_id: str) -> _SessionState | None:
        state = self._sessions.get(session_id)
        if state is not None and state.owner != current_request_identity():
            raise SessionIdentityMismatch(
                "The conversation does not belong to the current demo identity."
            )
        return state

    @staticmethod
    def _abandon_exceptional_entry_draft(state: _SessionState) -> None:
        history_start = state.exceptional_entry_history_start
        if history_start is not None:
            del state.history[history_start:]
        state.exceptional_entry_draft = None
        state.exceptional_entry_history_start = None

    def ensure_session(self, session_id: str | None = None) -> str:
        with self._lock:
            current = self._now()
            self._remove_stale_sessions(current)
            resolved = session_id or str(uuid4())
            identity = current_request_identity()
            state = self._sessions.get(resolved)
            if state is None:
                state = _SessionState(
                    session_id=resolved,
                    owner=identity,
                    updated_at=current,
                )
                self._sessions[resolved] = state
            elif state.owner != identity:
                raise SessionIdentityMismatch(
                    "The conversation does not belong to the current demo identity."
                )
            else:
                state.updated_at = current
            return resolved

    def create_pending_action(
        self,
        session_id: str,
        *,
        action_type: str,
        validated_arguments: dict[str, object],
        summary: str,
        language: str,
    ) -> PendingAction:
        with self._lock:
            self.ensure_session(session_id)
            current = self._now()
            action = PendingAction(
                confirmation_id=str(uuid4()),
                action_type=action_type,
                validated_arguments=deepcopy(validated_arguments),
                summary=summary,
                language=language,
                created_at=current,
                expires_at=current + self.confirmation_ttl,
                owner=current_request_identity(),
            )
            state = self._sessions[session_id]
            state.pending_action = action
            state.exceptional_entry_draft = None
            state.exceptional_entry_history_start = None
            state.expired_pending_language = None
            state.expired_pending_action_type = None
            state.updated_at = current
            return self._clone_action(action)

    def create_exceptional_entry_draft(
        self,
        session_id: str,
        *,
        attendance_date: str,
        entry_type: str,
        suggested_entry_time: str,
        language: str,
        reason_options: list[str] | tuple[str, ...] | None = None,
    ) -> ExceptionalEntryDraft:
        if entry_type not in {"IN", "OUT"}:
            raise ValueError("Exceptional-entry draft direction must be IN or OUT.")
        if language not in {"en", "ar"}:
            raise ValueError("Exceptional-entry draft language must be en or ar.")
        if not attendance_date or not suggested_entry_time:
            raise ValueError("Exceptional-entry draft selection is incomplete.")
        normalized_reason_options = tuple(
            option.strip()
            for option in (reason_options or ())
            if isinstance(option, str) and option.strip()
        )
        with self._lock:
            self.ensure_session(session_id)
            current = self._now()
            draft = ExceptionalEntryDraft(
                attendance_date=attendance_date,
                entry_type=entry_type,
                suggested_entry_time=suggested_entry_time,
                language=language,
                created_at=current,
                expires_at=current + self.confirmation_ttl,
                reason_options=normalized_reason_options,
            )
            state = self._sessions[session_id]
            if state.exceptional_entry_draft is None:
                state.exceptional_entry_history_start = len(state.history)
            state.exceptional_entry_draft = draft
            state.updated_at = current
            return self._clone_draft(draft)

    def get_exceptional_entry_draft(
        self,
        session_id: str,
    ) -> ExceptionalEntryDraft | None:
        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.exceptional_entry_draft is None:
                return None
            current = self._now()
            if current >= state.exceptional_entry_draft.expires_at:
                self._abandon_exceptional_entry_draft(state)
                state.updated_at = current
                return None
            return self._clone_draft(state.exceptional_entry_draft)

    def clear_exceptional_entry_draft(
        self,
        session_id: str,
    ) -> ExceptionalEntryDraft | None:
        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.exceptional_entry_draft is None:
                return None
            draft = self._clone_draft(state.exceptional_entry_draft)
            self._abandon_exceptional_entry_draft(state)
            state.updated_at = self._now()
            return draft

    def get_pending_action(
        self,
        session_id: str,
    ) -> tuple[PendingAction | None, bool]:
        """Return (action, expired). Expired actions are discarded immediately."""

        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.pending_action is None:
                return None, False
            current = self._now()
            if current >= state.pending_action.expires_at:
                state.expired_pending_language = state.pending_action.language
                state.expired_pending_action_type = state.pending_action.action_type
                state.pending_action = None
                state.updated_at = current
                return None, True
            return self._clone_action(state.pending_action), False

    def get_expired_pending_language(self, session_id: str) -> str | None:
        with self._lock:
            state = self._owned_state(session_id)
            return state.expired_pending_language if state else None

    def get_expired_pending_action_type(self, session_id: str) -> str | None:
        with self._lock:
            state = self._owned_state(session_id)
            return state.expired_pending_action_type if state else None

    def consume_pending_action(
        self,
        session_id: str,
        confirmation_id: str | None = None,
    ) -> PendingAction:
        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.pending_action is None:
                raise PendingActionMismatch("There is no pending action to confirm.")
            current = self._now()
            action = state.pending_action
            if action.owner != current_request_identity():
                raise PendingActionMismatch(
                    "The pending action does not belong to the current demo identity."
                )
            if current >= action.expires_at:
                state.pending_action = None
                state.updated_at = current
                raise PendingActionExpired("The pending confirmation has expired.")
            if confirmation_id and confirmation_id != action.confirmation_id:
                raise PendingActionMismatch("The confirmation ID does not match.")
            state.pending_action = None
            state.expired_pending_language = None
            state.expired_pending_action_type = None
            state.updated_at = current
            return self._clone_action(action)

    def discard_pending_action(self, session_id: str) -> PendingAction | None:
        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.pending_action is None:
                return None
            action = self._clone_action(state.pending_action)
            state.pending_action = None
            state.expired_pending_language = None
            state.expired_pending_action_type = None
            state.updated_at = self._now()
            return action

    def get_history(self, session_id: str) -> list[HistoryItem]:
        with self._lock:
            state = self._owned_state(session_id)
            return deepcopy(state.history) if state else []

    def append_history(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self.ensure_session(session_id)
            state = self._sessions[session_id]
            state.history.append({"role": role, "content": content})
            overflow = max(0, len(state.history) - self.history_limit)
            if overflow:
                state.history = state.history[overflow:]
                if state.exceptional_entry_history_start is not None:
                    state.exceptional_entry_history_start = max(
                        0,
                        state.exceptional_entry_history_start - overflow,
                    )
            state.updated_at = self._now()

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


_settings = get_settings()
session_store = InMemorySessionStore(
    confirmation_ttl_seconds=_settings.confirmation_ttl_seconds,
    session_ttl_seconds=_settings.session_ttl_seconds,
)
