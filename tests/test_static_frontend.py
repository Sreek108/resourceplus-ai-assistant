import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import FRONTEND_DIST, FRONTEND_INDEX, app


client = TestClient(app)


def _require_build() -> None:
    if not FRONTEND_INDEX.is_file():
        pytest.skip("Run `npm run build` in frontend/ before static frontend tests.")


def test_root_serves_production_react_index() -> None:
    _require_build()
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="root"></div>' in response.text


def test_built_asset_is_served_from_frontend_dist() -> None:
    _require_build()
    index = FRONTEND_INDEX.read_text(encoding="utf-8")
    match = re.search(r'(?:src|href)="(/assets/[^"]+)"', index)
    assert match is not None

    response = client.get(match.group(1))

    assert response.status_code == 200
    assert response.content


def test_spa_route_returns_index_without_intercepting_platform_routes() -> None:
    _require_build()
    index = client.get("/")
    spa = client.get("/employee/assistant")

    assert spa.status_code == 200
    assert spa.text == index.text
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_unknown_api_route_remains_json_404() -> None:
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "<html" not in response.text.casefold()


@pytest.mark.parametrize(
    "path",
    [
        "/.env",
        "/data/assistant_audit.db",
        "/app/main.py",
        "/tests/test_api.py",
    ],
)
def test_project_and_secret_paths_are_not_static(path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404
    assert "<div id=\"root\"></div>" not in response.text


def test_production_javascript_contains_no_configured_secrets() -> None:
    _require_build()
    settings = get_settings()
    sensitive_values = (
        settings.openai_api_key,
        settings.azure_speech_key,
        settings.rp_default_email,
        settings.rp_manager_email,
    )
    bundles = list((FRONTEND_DIST / "assets").glob("*.js"))
    assert bundles

    for bundle in bundles:
        content = bundle.read_bytes()
        for secret in sensitive_values:
            if secret and secret.encode("utf-8") in content:
                pytest.fail("A configured backend-only value was found in the frontend bundle.")

    if any(b"127.0.0.1:8001" in bundle.read_bytes() for bundle in bundles):
        pytest.fail("The production frontend contains the local development API target.")


def test_uat_origin_is_added_without_wildcarding_cors() -> None:
    settings = Settings(
        _env_file=None,
        cors_allowed_origins="http://127.0.0.1:5173",
        uat_allowed_origins="https://lead-uat.ngrok-free.app/",
    )

    assert settings.cors_origins == [
        "http://127.0.0.1:5173",
        "https://lead-uat.ngrok-free.app",
    ]
    assert "*" not in settings.cors_origins


def test_uat_origin_rejects_wildcards() -> None:
    settings = Settings(
        _env_file=None,
        cors_allowed_origins="http://127.0.0.1:5173",
        uat_allowed_origins="*",
    )

    with pytest.raises(ValueError, match="explicit HTTP"):
        _ = settings.cors_origins
