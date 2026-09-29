import base64
import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.audit import (
    persist_audit,
    reset_interaction_audit,
    safe_session_reference,
    start_interaction_audit,
)
from app.config import get_settings
from app.identity import RequestIdentityError, resolve_request_identity
from app.models.schemas import ChatRequest, VoiceChatResponse
from app.observability import (
    SAFE_VOICE_ERROR_CATEGORIES,
    current_voice_trace,
    mark_voice_route_complete,
    measure_stage,
    reset_voice_trace,
    start_voice_trace,
)
from app.services.chat import process_chat
from app.speech import (
    SpeechConfigurationError,
    SpeechInputError,
    SpeechRecognitionError,
    SpeechSynthesisError,
    markdown_to_speech_text,
    synthesize_speech,
    transcribe_audio,
    StreamingSpeechRecognizer,
)
from app.telemetry import (
    TRACE_HEADER,
    classify_error,
    emit_structured_event,
    metrics,
    record_websocket_failure,
    reset_trace,
    start_trace,
    validate_trace_id,
)


router = APIRouter(prefix="/api/voice", tags=["voice"])
MAX_AUDIO_BYTES = 10 * 1024 * 1024
latency_logger = logging.getLogger("app.voice_latency")


def _capture_voice_trace(audit) -> None:
    trace = current_voice_trace()
    if trace is not None:
        audit.latencies = {
            key: round(value * 1000, 3)
            for key, value in trace.snapshot().items()
        }
        audit.model_requests = trace.model_requests


def _safe_error_category(exc: Exception) -> str:
    category = getattr(exc, "safe_category", None)
    return (
        category
        if isinstance(category, str) and category in SAFE_VOICE_ERROR_CATEGORIES
        else "unknown_safe_category"
    )


def _mark_voice_error(category: str) -> None:
    trace = current_voice_trace()
    if trace is not None:
        trace.set_error_category(category)


def _safe_stream_error(exc: Exception) -> tuple[str, str, str, int, int]:
    category = _safe_error_category(exc)
    if isinstance(exc, SpeechInputError):
        if category in {"no_audio", "no_recognized_speech"}:
            return (
                "no_speech",
                "I didn't catch that. Hold the mic and try again.",
                category,
                204,
                1000,
            )
        return (
            "invalid_stream_state",
            "The voice recording could not be completed. Please try again.",
            "invalid_stream_state",
            400,
            1008,
        )
    if isinstance(exc, SpeechConfigurationError):
        return "voice_unavailable", "Voice is temporarily unavailable.", category, 503, 1011
    if isinstance(exc, SpeechRecognitionError):
        return (
            "speech_recognition_failed",
            "I couldn't recognize that voice message.",
            category,
            502,
            1011,
        )
    if isinstance(exc, SpeechSynthesisError):
        return (
            "speech_synthesis_failed",
            "I understood you, but couldn't create a spoken reply.",
            category,
            502,
            1011,
        )
    return (
        "voice_processing_failed",
        "I couldn't process that voice message.",
        "unknown_safe_category",
        500,
        1011,
    )


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    confirmation_id: str | None = Form(default=None),
    email: str | None = Form(default=None),
    instance: str | None = Form(default=None),
) -> VoiceChatResponse:
    try:
        request_identity = resolve_request_identity(email, instance)
    except RequestIdentityError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_identity", "message": str(exc)},
        ) from exc
    audit, audit_token = start_interaction_audit(
        input_mode="voice",
        input_source="stt",
        session_id=session_id,
    )
    with measure_stage("audio_receive"):
        payload = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(payload) > MAX_AUDIO_BYTES:
        audit.error_category = "fallback_invalid_audio"
        audit.result_status = "error"
        _capture_voice_trace(audit)
        await persist_audit(audit)
        reset_interaction_audit(audit_token)
        raise HTTPException(status_code=413, detail="Audio upload exceeds 10 MB.")
    try:
        recognized = await transcribe_audio(
            payload,
            content_type=audio.content_type,
        )
        audit.user_text = recognized.transcript
        audit.raw_detected_locale = recognized.detected_locale
        audit.resolved_language = recognized.detected_language
        with measure_stage("agent"):
            chat_response = await process_chat(
                ChatRequest(
                    message=recognized.transcript,
                    session_id=session_id,
                    email=request_identity.email,
                    instance=request_identity.instance,
                    confirmation_id=confirmation_id,
                ),
                detected_language=recognized.detected_language,
            )
        audit.session_reference = audit.session_reference or safe_session_reference(
            chat_response.session_id
        )
        audit.response_language = chat_response.language
        audit.display_message = chat_response.message
        audit.speech_message = chat_response.speech_message
        audit.tools_used = list(dict.fromkeys([*audit.tools_used, *chat_response.tools_used]))
        audit.success = chat_response.success
        audit.result_status = "success" if chat_response.success else "failed"
        audit.confirmation_required = chat_response.requires_confirmation
        speech_source = chat_response.speech_message
        if not speech_source:
            raise SpeechSynthesisError(
                "The assistant did not provide a dedicated spoken response."
            )
        with measure_stage("speech_normalization"):
            speech_text = markdown_to_speech_text(speech_source)
        audit.tts_text = speech_text
        audit.tts_requested = True
        audit.tts_generated = False
        with measure_stage("tts"):
            spoken = await synthesize_speech(
                speech_text,
                language=chat_response.language,
            )
        audit.tts_generated = True
        audit.tts_locale = spoken.locale
        audit.tts_voice = spoken.voice_name
    except SpeechInputError as exc:
        category = _safe_error_category(exc)
        _mark_voice_error(category)
        audit.error_category = category
        audit.result_status = "no_speech" if category in {
            "no_audio",
            "no_recognized_speech",
            "fallback_stt_no_match",
        } else "error"
        _capture_voice_trace(audit)
        await persist_audit(audit)
        reset_interaction_audit(audit_token)
        raise HTTPException(
            status_code=400,
            detail={"code": "no_speech", "message": str(exc)},
        ) from exc
    except SpeechConfigurationError as exc:
        category = _safe_error_category(exc)
        _mark_voice_error(category)
        audit.error_category = category
        audit.result_status = "error"
        _capture_voice_trace(audit)
        await persist_audit(audit)
        reset_interaction_audit(audit_token)
        raise HTTPException(
            status_code=503,
            detail={"code": "voice_unavailable", "message": str(exc)},
        ) from exc
    except SpeechRecognitionError as exc:
        category = _safe_error_category(exc)
        _mark_voice_error(category)
        audit.error_category = category
        audit.result_status = "error"
        _capture_voice_trace(audit)
        await persist_audit(audit)
        reset_interaction_audit(audit_token)
        raise HTTPException(
            status_code=502,
            detail={"code": "speech_recognition_failed", "message": str(exc)},
        ) from exc
    except SpeechSynthesisError as exc:
        category = _safe_error_category(exc)
        _mark_voice_error(category)
        audit.error_category = category
        audit.result_status = "error"
        _capture_voice_trace(audit)
        await persist_audit(audit)
        reset_interaction_audit(audit_token)
        raise HTTPException(
            status_code=502,
            detail={"code": "speech_synthesis_failed", "message": str(exc)},
        ) from exc
    except Exception as exc:
        _mark_voice_error("unknown_safe_category")
        audit.error_category = "unknown_safe_category"
        audit.result_status = "error"
        _capture_voice_trace(audit)
        await persist_audit(audit)
        reset_interaction_audit(audit_token)
        raise

    with measure_stage("response_build"):
        response = VoiceChatResponse(
            **chat_response.model_dump(),
            transcript=recognized.transcript,
            detected_language=recognized.detected_language,
            detected_locale=recognized.detected_locale,
            audio_base64=base64.b64encode(spoken.data).decode("ascii"),
            audio_mime_type=spoken.mime_type,
        )
    _capture_voice_trace(audit)
    with measure_stage("audit_persist"):
        await persist_audit(audit)
    mark_voice_route_complete()
    reset_interaction_audit(audit_token)
    return response


@router.websocket("/stream")
async def voice_stream(websocket: WebSocket) -> None:
    try:
        trace_id, request_trace_token = start_trace(websocket.headers.get(TRACE_HEADER))
    except ValueError:
        await websocket.close(code=1008)
        return
    trace, trace_token = start_voice_trace()
    recognizer: StreamingSpeechRecognizer | None = None
    audit = None
    audit_token = None
    audit_persisted = False
    completed = False
    status_code = 500
    origin = websocket.headers.get("origin")
    if origin and origin not in get_settings().cors_origins:
        status_code = 403
        trace.set_error_category("invalid_stream_state")
        record_websocket_failure("invalid_stream_state")
        metrics.increment(
            "resourceplus_assistant_requests_total",
            mode="voice",
            status="failure",
        )
        emit_structured_event(
            level="WARNING",
            component="websocket",
            event="origin_rejected",
            status=403,
            error_category="invalid_stream_state",
            error_owner="validation",
            error_stage="websocket_request",
            retryable=False,
        )
        await websocket.close(code=1008)
        latency_logger.info(trace.format_summary(status_code=status_code))
        reset_voice_trace(trace_token)
        reset_trace(request_trace_token)
        return
    await websocket.accept()
    try:
        start_message = await websocket.receive_text()
        try:
            metadata = json.loads(start_message)
        except json.JSONDecodeError as exc:
            raise SpeechInputError(
                "The stream start message is invalid.",
                safe_category="invalid_stream_state",
            ) from exc
        if not isinstance(metadata, dict) or metadata.get("type") != "start":
            raise SpeechInputError(
                "The stream must begin with a start message.",
                safe_category="invalid_stream_state",
            )
        if metadata.get("sample_rate") != 16_000:
            raise SpeechInputError(
                "Streaming audio must use 16 kHz PCM.",
                safe_category="invalid_stream_state",
            )
        metadata_trace_id = metadata.get("trace_id")
        if metadata_trace_id is not None:
            if not isinstance(metadata_trace_id, str):
                raise SpeechInputError(
                    "The trace reference is invalid.",
                    safe_category="invalid_stream_state",
                )
            try:
                validated_metadata_trace = validate_trace_id(metadata_trace_id)
            except ValueError as exc:
                raise SpeechInputError(
                    "The trace reference is invalid.",
                    safe_category="invalid_stream_state",
                ) from exc
            header_trace_id = websocket.headers.get(TRACE_HEADER)
            if header_trace_id and validated_metadata_trace != trace_id:
                raise SpeechInputError(
                    "The trace references do not match.",
                    safe_category="invalid_stream_state",
                )
            if not header_trace_id:
                reset_trace(request_trace_token)
                trace_id, request_trace_token = start_trace(validated_metadata_trace)
        session_id = metadata.get("session_id") or None
        confirmation_id = metadata.get("confirmation_id") or None
        if session_id is not None and (
            not isinstance(session_id, str) or not 1 <= len(session_id) <= 128
        ):
            raise SpeechInputError(
                "The session reference is invalid.",
                safe_category="invalid_stream_state",
            )
        if confirmation_id is not None and (
            not isinstance(confirmation_id, str)
            or not 1 <= len(confirmation_id) <= 128
        ):
            raise SpeechInputError(
                "The confirmation reference is invalid.",
                safe_category="invalid_stream_state",
            )
        try:
            request_identity = resolve_request_identity(
                metadata.get("email"),
                metadata.get("instance"),
            )
        except RequestIdentityError as exc:
            raise SpeechInputError(
                "The demo identity must contain a valid email and instance pair.",
                safe_category="invalid_stream_state",
            ) from exc
        debug = metadata.get("debug") is True
        audit, audit_token = start_interaction_audit(
            input_mode="voice",
            input_source="stt",
            session_id=session_id,
        )
        recognizer = StreamingSpeechRecognizer()
        await recognizer.start()
        await websocket.send_json({"type": "ready"})

        received_bytes = 0
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            chunk = message.get("bytes")
            if chunk is not None:
                trace.mark_audio_chunk()
                received_bytes += len(chunk)
                if received_bytes > MAX_AUDIO_BYTES:
                    raise SpeechInputError(
                        "The streamed audio is too large.",
                        safe_category="invalid_stream_state",
                    )
                recognizer.write(chunk)
                continue
            control_text = message.get("text")
            if control_text is None:
                continue
            try:
                control = json.loads(control_text)
            except json.JSONDecodeError as exc:
                raise SpeechInputError(
                    "The stream control message is invalid.",
                    safe_category="invalid_stream_state",
                ) from exc
            control_type = control.get("type") if isinstance(control, dict) else None
            if control_type == "cancel":
                status_code = 499
                trace.set_error_category("client_aborted")
                audit.error_category = "client_aborted"
                audit.result_status = "aborted"
                _capture_voice_trace(audit)
                await persist_audit(audit)
                audit_persisted = True
                await recognizer.cancel()
                await websocket.close(code=1000)
                return
            if control_type == "end":
                break
            raise SpeechInputError(
                "The stream control message is invalid.",
                safe_category="invalid_stream_state",
            )

        if received_bytes == 0:
            raise SpeechInputError(
                "No audio was received.",
                safe_category="no_audio",
            )
        trace.mark_release()
        with measure_stage("post_release_stt_finalize"):
            recognized = await recognizer.finish()
        completed = True

        audit.user_text = recognized.transcript
        audit.raw_detected_locale = recognized.detected_locale
        audit.resolved_language = recognized.detected_language
        with measure_stage("agent"):
            chat_response = await process_chat(
                ChatRequest(
                    message=recognized.transcript,
                    session_id=session_id,
                    email=request_identity.email,
                    instance=request_identity.instance,
                    confirmation_id=confirmation_id,
                ),
                detected_language=recognized.detected_language,
            )
        audit.session_reference = audit.session_reference or safe_session_reference(
            chat_response.session_id
        )
        audit.response_language = chat_response.language
        audit.display_message = chat_response.message
        audit.speech_message = chat_response.speech_message
        audit.tools_used = list(dict.fromkeys([*audit.tools_used, *chat_response.tools_used]))
        audit.success = chat_response.success
        audit.result_status = "success" if chat_response.success else "failed"
        audit.confirmation_required = chat_response.requires_confirmation
        speech_source = chat_response.speech_message
        if not speech_source:
            raise SpeechSynthesisError(
                "The assistant did not provide a dedicated spoken response."
            )
        with measure_stage("speech_normalization"):
            speech_text = markdown_to_speech_text(speech_source)
        audit.tts_text = speech_text
        audit.tts_requested = True
        audit.tts_generated = False
        with measure_stage("tts"):
            spoken = await synthesize_speech(
                speech_text,
                language=chat_response.language,
            )
        audit.tts_generated = True
        audit.tts_locale = spoken.locale
        audit.tts_voice = spoken.voice_name
        response = VoiceChatResponse(
            **chat_response.model_dump(),
            transcript=recognized.transcript,
            detected_language=recognized.detected_language,
            detected_locale=recognized.detected_locale,
            audio_base64=base64.b64encode(spoken.data).decode("ascii"),
            audio_mime_type=spoken.mime_type,
        )
        response_payload = {"type": "final", **response.model_dump()}
        if debug:
            response_payload["interaction_id"] = audit.interaction_id
            response_payload["trace_id"] = trace_id
            response_payload["timings"] = trace.snapshot()
        with measure_stage("response_send"):
            await websocket.send_json(response_payload)
        trace.mark_response_sent()
        status_code = 200
        _capture_voice_trace(audit)
        with measure_stage("audit_persist"):
            await persist_audit(audit)
        audit_persisted = True
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        if status_code != 200:
            status_code = 499
            trace.set_error_category("websocket_disconnected")
            record_websocket_failure("websocket_disconnected")
            if audit is not None and not audit_persisted:
                audit.error_category = "websocket_disconnected"
                audit.result_status = "aborted"
                _capture_voice_trace(audit)
                await persist_audit(audit)
                audit_persisted = True
    except Exception as exc:
        code, message, category, status_code, close_code = _safe_stream_error(exc)
        trace.set_error_category(category)
        if status_code >= 400:
            record_websocket_failure(category)
        if audit is not None:
            audit.error_category = category
            audit.result_status = "no_speech" if category in {
                "no_audio",
                "no_recognized_speech",
            } else "error"
            _capture_voice_trace(audit)
            await persist_audit(audit)
            audit_persisted = True
        try:
            await websocket.send_json(
                {"type": "error", "code": code, "message": message}
            )
            await websocket.close(code=close_code)
        except Exception:
            pass
    finally:
        if recognizer is not None and not completed:
            await recognizer.cancel()
        latency_logger.info(trace.format_summary(status_code=status_code))
        snapshot = trace.snapshot()
        if "audio_stream_duration" in snapshot:
            emit_structured_event(
                component="audio",
                event="stream_completed",
                duration_ms=snapshot["audio_stream_duration"] * 1000,
                status="success" if status_code in {200, 204} else "failure",
            )
        if "total_after_release" in snapshot:
            metrics.observe(
                "resourceplus_assistant_voice_after_release_duration_ms",
                snapshot["total_after_release"] * 1000,
                status=(
                    "success"
                    if status_code == 200
                    else "no_speech"
                    if status_code == 204
                    else "aborted"
                    if status_code == 499
                    else "failure"
                ),
            )
        metrics.increment(
            "resourceplus_assistant_requests_total",
            mode="voice",
            status="success" if status_code in {200, 204} else "failure",
        )
        emit_structured_event(
            level="INFO" if status_code in {200, 204, 499} else "WARNING",
            component="websocket",
            event="request_completed",
            duration_ms=snapshot.get("total", 0.0) * 1000,
            status=status_code,
            error_category=None if status_code == 200 else trace.error_category,
            error_owner=(
                None
                if status_code == 200
                else classify_error(trace.error_category)[0]
            ),
            error_stage=(
                None
                if status_code == 200
                else classify_error(trace.error_category)[1]
            ),
            retryable=None if status_code == 200 else status_code >= 500,
        )
        if audit_token is not None:
            reset_interaction_audit(audit_token)
        reset_voice_trace(trace_token)
        reset_trace(request_trace_token)
