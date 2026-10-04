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


@dataclass(frozen=True)
class ConversationDraft:
    """Short-lived semantic slots; this state can never execute a write itself."""

    intent: str
    slots: dict[str, str]
    validated_slots: tuple[str, ...]
    language: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class ApprovalCandidate:
    """Identity-bound pending request facts; IDs are never rendered to users."""

    ordinal: int
    request_id: str = field(repr=False)
    request_type: str = field(repr=False)
    employee_name: str
    category: str
    detail: str
    request_date: str
    status: str = "Pending"


@dataclass(frozen=True)
class RequestCorrelation:
    """Safe session evidence for reconciling ambiguous request statuses."""

    category: str
    request_date: str
    detail: str
    state: str


@dataclass(frozen=True)
class TrustedResultContext:
    """Non-executable provenance for the latest successful ResourcePlus result."""

    message: str
    tools_used: tuple[str, ...]
    language: str
    created_at: datetime
    correction_dates: tuple[str, ...] = ()
    attendance_period: tuple[str, str] | None = None
    attendance_period_label: str | None = None
    attendance_period_source: str | None = None
    discussed_date: str | None = None
    recent_request_category: str | None = None
    recent_request_date: str | None = None
    recent_request_detail: str | None = None
    recent_request_state: str | None = None
    request_correlations: tuple[RequestCorrelation, ...] = ()
    approval_candidates: tuple[ApprovalCandidate, ...] = ()


@dataclass
class _SessionState:
    session_id: str
    owner: RequestIdentity = field(repr=False)
    history: list[HistoryItem] = field(default_factory=list)
    pending_action: PendingAction | None = None
    exceptional_entry_draft: ExceptionalEntryDraft | None = None
    conversation_draft: ConversationDraft | None = None
    trusted_result: TrustedResultContext | None = None
    exceptional_entry_history_start: int | None = None
    expired_pending_language: str | None = None
    expired_pending_action_type: str | None = None
    last_confident_language: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PendingActionExpired(Exception):
    pass


class PendingActionMismatch(Exception):
    pass


class SessionIdentityMismatch(Exception):
    pass


class SessionStore(Protocol):
    def ensure_session(self, session_id: str | None = None) -> str: ...

    def get_last_confident_language(self, session_id: str) -> str | None: ...

    def set_last_confident_language(
        self,
        session_id: str,
        language: str | None,
    ) -> None: ...

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

    def save_conversation_draft(
        self,
        session_id: str,
        *,
        intent: str,
        slots: dict[str, str],
        validated_slots: tuple[str, ...] = (),
        language: str,
    ) -> ConversationDraft: ...

    def get_conversation_draft(self, session_id: str) -> ConversationDraft | None: ...

    def clear_conversation_draft(self, session_id: str) -> ConversationDraft | None: ...

    def save_trusted_result(
        self,
        session_id: str,
        *,
        message: str,
        tools_used: list[str] | tuple[str, ...],
        language: str,
        correction_dates: tuple[str, ...] | None = None,
        attendance_period: tuple[str, str] | None = None,
        attendance_period_label: str | None = None,
        attendance_period_source: str | None = None,
        discussed_date: str | None = None,
        recent_request_category: str | None = None,
        recent_request_date: str | None = None,
        recent_request_detail: str | None = None,
        recent_request_state: str | None = None,
        approval_candidates: tuple[ApprovalCandidate, ...] | None = None,
    ) -> TrustedResultContext: ...

    def get_trusted_result(self, session_id: str) -> TrustedResultContext | None: ...

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

    @staticmethod
    def _clone_conversation_draft(draft: ConversationDraft) -> ConversationDraft:
        return ConversationDraft(
            intent=draft.intent,
            slots=deepcopy(draft.slots),
            validated_slots=draft.validated_slots,
            language=draft.language,
            created_at=draft.created_at,
            expires_at=draft.expires_at,
        )

    @staticmethod
    def _clone_trusted_result(context: TrustedResultContext) -> TrustedResultContext:
        return TrustedResultContext(
            message=context.message,
            tools_used=context.tools_used,
            language=context.language,
            created_at=context.created_at,
            correction_dates=context.correction_dates,
            attendance_period=context.attendance_period,
            attendance_period_label=context.attendance_period_label,
            attendance_period_source=context.attendance_period_source,
            discussed_date=context.discussed_date,
            recent_request_category=context.recent_request_category,
            recent_request_date=context.recent_request_date,
            recent_request_detail=context.recent_request_detail,
            recent_request_state=context.recent_request_state,
            request_correlations=context.request_correlations,
            approval_candidates=context.approval_candidates,
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

    def get_last_confident_language(self, session_id: str) -> str | None:
        with self._lock:
            state = self._owned_state(session_id)
            return state.last_confident_language if state is not None else None

    def set_last_confident_language(
        self,
        session_id: str,
        language: str | None,
    ) -> None:
        if language is not None and language not in {"en", "ar"}:
            raise ValueError("Conversation language must be en, ar, or None.")
        with self._lock:
            self.ensure_session(session_id)
            state = self._sessions[session_id]
            state.last_confident_language = language
            state.updated_at = self._now()

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
            state.conversation_draft = None
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

    def save_conversation_draft(
        self,
        session_id: str,
        *,
        intent: str,
        slots: dict[str, str],
        validated_slots: tuple[str, ...] = (),
        language: str,
    ) -> ConversationDraft:
        if not intent.strip() or language not in {"en", "ar"}:
            raise ValueError("Conversation draft intent and language are required.")
        with self._lock:
            self.ensure_session(session_id)
            current = self._now()
            draft = ConversationDraft(
                intent=intent,
                slots=deepcopy(slots),
                validated_slots=tuple(dict.fromkeys(validated_slots)),
                language=language,
                created_at=current,
                expires_at=current + self.confirmation_ttl,
            )
            state = self._sessions[session_id]
            state.conversation_draft = draft
            state.updated_at = current
            return self._clone_conversation_draft(draft)

    def get_conversation_draft(self, session_id: str) -> ConversationDraft | None:
        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.conversation_draft is None:
                return None
            current = self._now()
            if current >= state.conversation_draft.expires_at:
                state.conversation_draft = None
                state.updated_at = current
                return None
            return self._clone_conversation_draft(state.conversation_draft)

    def clear_conversation_draft(self, session_id: str) -> ConversationDraft | None:
        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.conversation_draft is None:
                return None
            draft = self._clone_conversation_draft(state.conversation_draft)
            state.conversation_draft = None
            state.updated_at = self._now()
            return draft

    def save_trusted_result(
        self,
        session_id: str,
        *,
        message: str,
        tools_used: list[str] | tuple[str, ...],
        language: str,
        correction_dates: tuple[str, ...] | None = None,
        attendance_period: tuple[str, str] | None = None,
        attendance_period_label: str | None = None,
        attendance_period_source: str | None = None,
        discussed_date: str | None = None,
        recent_request_category: str | None = None,
        recent_request_date: str | None = None,
        recent_request_detail: str | None = None,
        recent_request_state: str | None = None,
        approval_candidates: tuple[ApprovalCandidate, ...] | None = None,
    ) -> TrustedResultContext:
        if not message.strip() or not tools_used or language not in {"en", "ar"}:
            raise ValueError("Trusted result provenance is incomplete.")
        with self._lock:
            self.ensure_session(session_id)
            current = self._now()
            previous = self._sessions[session_id].trusted_result
            correlations = list(previous.request_correlations if previous is not None else ())
            correlation_date = recent_request_date
            correlation_state = recent_request_state
            if correlation_date is not None and correlation_state is not None:
                correlation_category = recent_request_category or "attendance_correction"
                correlation_detail = recent_request_detail or ""
                correlation = RequestCorrelation(
                    category=correlation_category,
                    request_date=correlation_date,
                    detail=correlation_detail,
                    state=correlation_state,
                )
                correlations = [
                    item for item in correlations
                    if not (
                        item.category == correlation.category
                        and item.request_date == correlation.request_date
                        and item.detail.casefold() == correlation.detail.casefold()
                    )
                ]
                correlations.append(correlation)
                correlations = correlations[-20:]
            context = TrustedResultContext(
                message=message.strip(),
                tools_used=tuple(dict.fromkeys(tools_used)),
                language=language,
                created_at=current,
                correction_dates=(
                    correction_dates
                    if correction_dates is not None
                    else previous.correction_dates if previous is not None else ()
                ),
                attendance_period=(
                    attendance_period
                    if attendance_period is not None
                    else previous.attendance_period if previous is not None else None
                ),
                attendance_period_label=(
                    attendance_period_label
                    if attendance_period_label is not None
                    else previous.attendance_period_label if previous is not None else None
                ),
                attendance_period_source=(
                    attendance_period_source
                    if attendance_period_source is not None
                    else previous.attendance_period_source if previous is not None else None
                ),
                discussed_date=(
                    discussed_date
                    if discussed_date is not None
                    else previous.discussed_date if previous is not None else None
                ),
                recent_request_category=(
                    recent_request_category
                    if recent_request_category is not None
                    else previous.recent_request_category if previous is not None else None
                ),
                recent_request_date=(
                    recent_request_date
                    if recent_request_date is not None
                    else previous.recent_request_date if previous is not None else None
                ),
                recent_request_detail=(
                    recent_request_detail
                    if recent_request_detail is not None
                    else previous.recent_request_detail if previous is not None else None
                ),
                recent_request_state=(
                    recent_request_state
                    if recent_request_state is not None
                    else previous.recent_request_state if previous is not None else None
                ),
                request_correlations=tuple(correlations),
                approval_candidates=(
                    approval_candidates
                    if approval_candidates is not None
                    else previous.approval_candidates if previous is not None else ()
                ),
            )
            state = self._sessions[session_id]
            state.trusted_result = context
            state.updated_at = current
            return self._clone_trusted_result(context)

    def get_trusted_result(self, session_id: str) -> TrustedResultContext | None:
        with self._lock:
            state = self._owned_state(session_id)
            if state is None or state.trusted_result is None:
                return None
            return self._clone_trusted_result(state.trusted_result)

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
