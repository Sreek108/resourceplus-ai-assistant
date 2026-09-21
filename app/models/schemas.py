from pydantic import BaseModel, Field, PrivateAttr, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    lang: int | None = Field(default=None, ge=1)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    confirmation_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    success: bool
    message: str
    language: str
    tools_used: list[str] = Field(default_factory=list)
    session_id: str
    requires_confirmation: bool = False
    confirmation_id: str | None = None
    _speech_message: str | None = PrivateAttr(default=None)

    @property
    def speech_message(self) -> str | None:
        return self._speech_message

    def set_speech_message(self, message: str | None) -> "ChatResponse":
        self._speech_message = message
        return self


class VoiceChatResponse(ChatResponse):
    transcript: str
    detected_language: str
    detected_locale: str
    audio_base64: str
    audio_mime_type: str


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
