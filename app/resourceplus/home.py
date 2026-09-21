from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG, get_settings
from app.resourceplus.client import ResourcePlusClient


HOME_DATA_ROUTE = "api/Client/GetHomeData"


async def get_home_data(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    settings = get_settings()
    # TODO: Replace the configured POC email with authenticated_user.email.
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        HOME_DATA_ROUTE,
        params={
            "instanceName": settings.rp_instance,
            "Usremail": identity,
            "Lang": lang,
        },
    )

