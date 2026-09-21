from types import SimpleNamespace

import pytest

from app import audit as audit_module
from app.api import telemetry as telemetry_api_module


@pytest.fixture(autouse=True)
def disable_real_audit_database_during_tests(monkeypatch):
    """Never let normal API tests append synthetic turns to the configured UAT DB."""

    configured = audit_module.get_settings()
    safe_settings = SimpleNamespace(
        ai_audit_enabled=False,
        ai_audit_store_content=False,
        ai_audit_retention_days=configured.ai_audit_retention_days,
        ai_audit_db_path=configured.ai_audit_db_path,
        frontend_telemetry_rate_limit_per_minute=120,
    )
    monkeypatch.setattr(audit_module, "get_settings", lambda: safe_settings)
    monkeypatch.setattr(telemetry_api_module, "get_settings", lambda: safe_settings)
