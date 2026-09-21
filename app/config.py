from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_RESOURCEPLUS_LANG = 1


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    rp_base_url: str = "https://app.resourceplus.app/Mobile/"
    rp_instance: str = "Universal"
    # TODO: Replace this POC identity with authenticated_user.email from the
    # authenticated ResourcePlus session before production use.
    rp_default_email: str = "saneesh.netsoftpro@gmail.com"
    rp_default_lang: int = Field(default=DEFAULT_RESOURCEPLUS_LANG, ge=1)
    rp_timeout_seconds: float = Field(default=20.0, gt=0)
    rp_manager_email: str | None = None
    confirmation_ttl_seconds: int = Field(default=300, ge=30, le=1_800)
    session_ttl_seconds: int = Field(default=1_800, ge=300, le=86_400)
    cors_allowed_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    uat_allowed_origins: str = ""

    openai_api_key: str | None = None
    openai_model: str | None = None

    azure_speech_key: str | None = None
    azure_speech_region: str | None = None
    azure_speech_en_locale: str = "en-US"
    azure_speech_ar_locale: str = "ar-SA"
    azure_speech_en_voice: str = "en-US-AvaNeural"
    azure_speech_ar_voice: str = "ar-SA-ZariyahNeural"

    ai_audit_enabled: bool = False
    ai_audit_store_content: bool = False
    ai_audit_retention_days: int = Field(default=7, ge=1, le=365)
    ai_audit_db_path: str = "data/assistant_audit.db"
    ai_audit_debug_endpoint_enabled: bool = False

    app_version: str = "0.2.0"
    git_commit: str = "unknown"
    deployment_id: str = "local"
    app_environment: str = "local"
    observability_exporter: str = "none"
    observability_endpoint: str | None = None
    ops_diagnostics_enabled: bool = False
    ops_diagnostics_token: str | None = None
    frontend_telemetry_rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)

    @field_validator(
        "openai_api_key",
        "openai_model",
        "rp_manager_email",
        "azure_speech_key",
        "azure_speech_region",
        "observability_endpoint",
        "ops_diagnostics_token",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "app_version",
        "git_commit",
        "deployment_id",
        "app_environment",
        "observability_exporter",
    )
    @classmethod
    def safe_deployment_metadata(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > 128 or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/"
            for character in normalized
        ):
            raise ValueError("Deployment metadata must contain only safe ASCII characters.")
        return normalized

    @property
    def cors_origins(self) -> list[str]:
        """Return the explicit local and temporary-UAT browser origin allowlist."""
        origins: list[str] = []
        for configured in (self.cors_allowed_origins, self.uat_allowed_origins):
            for origin in configured.split(","):
                normalized = origin.strip().rstrip("/")
                if not normalized:
                    continue
                parsed = urlsplit(normalized)
                if (
                    normalized == "*"
                    or parsed.scheme not in {"http", "https"}
                    or not parsed.netloc
                    or parsed.path
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError(
                        "Browser origins must be explicit HTTP(S) origins without paths."
                    )
                if normalized not in origins:
                    origins.append(normalized)
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
