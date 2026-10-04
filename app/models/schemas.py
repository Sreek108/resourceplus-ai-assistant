from typing import Annotated, Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator


class ApprovalSelection(BaseModel):
    kind: Literal["pending_approval"] = "pending_approval"
    decision: Literal["approve", "reject"]
    ordinal: int = Field(ge=1, le=500)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    lang: int | None = Field(default=None, ge=1)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    email: str | None = None
    instance: str | None = None
    confirmation_id: str | None = Field(default=None, min_length=1, max_length=128)
    approval_selection: ApprovalSelection | None = None

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ReasonOption(BaseModel):
    label: str
    value: str


class BlockColumn(BaseModel):
    key: str
    label: str


class BlockItem(BaseModel):
    label: str
    value: str | int | float | bool | None


class BlockAction(BaseModel):
    label: str
    value: str
    style: Literal["primary", "secondary", "danger"] = "secondary"
    payload: ApprovalSelection | None = None


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    title: str
    columns: list[BlockColumn]
    rows: list[dict[str, str | int | float | bool | None]]
    row_actions: list[list[BlockAction]] = Field(default_factory=list)


class KeyValueBlock(BaseModel):
    type: Literal["key_value"] = "key_value"
    title: str
    items: list[BlockItem]


class StatCardsBlock(BaseModel):
    type: Literal["stat_cards"] = "stat_cards"
    title: str
    items: list[BlockItem]


class ListBlock(BaseModel):
    type: Literal["list"] = "list"
    title: str
    items: list[str]


class ActionsBlock(BaseModel):
    type: Literal["actions"] = "actions"
    title: str
    actions: list[BlockAction]


class ConfirmationBlock(BaseModel):
    type: Literal["confirmation"] = "confirmation"
    title: str
    summary: str
    actions: list[BlockAction] = Field(
        default_factory=lambda: [
            BlockAction(label="Confirm", value="confirm", style="primary"),
            BlockAction(label="Cancel", value="cancel", style="secondary"),
        ]
    )


class NoticeBlock(BaseModel):
    type: Literal["notice"] = "notice"
    title: str
    message: str
    level: Literal["info", "success", "warning", "error"] = "info"


ResponseBlock = Annotated[
    TableBlock
    | KeyValueBlock
    | StatCardsBlock
    | ListBlock
    | ActionsBlock
    | ConfirmationBlock
    | NoticeBlock,
    Field(discriminator="type"),
]


class ChatResponse(BaseModel):
    response_schema_version: Literal[2] = 2
    success: bool
    message: str
    display_message: str | None = None
    language: str
    tools_used: list[str] = Field(default_factory=list)
    session_id: str
    requires_confirmation: bool = False
    confirmation_id: str | None = None
    needs_reason: bool = False
    reason_options: list[ReasonOption] | None = None
    blocks: list[ResponseBlock] = Field(default_factory=list)
    _speech_message: str | None = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        if self.display_message is None:
            self.display_message = self.message
        self.blocks = [
            block
            for block in self.blocks
            if not (
                (block.type == "table" and not block.rows)
                or (block.type in {"key_value", "stat_cards", "list"} and not block.items)
                or (block.type == "actions" and not block.actions)
            )
        ]

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


class SpeechSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    language: Literal["en", "ar"]


class SpeechSynthesisResponse(BaseModel):
    audio_base64: str
    audio_mime_type: str
    language: Literal["en", "ar"]
    tts_locale: str
    tts_voice: str


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
