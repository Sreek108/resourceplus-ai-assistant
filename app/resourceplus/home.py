from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG
from app.identity import resourceplus_identity
from app.resourceplus.client import ResourcePlusClient


HOME_DATA_ROUTE = "api/Client/GetHomeData"


async def get_home_data(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    instance_name: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    identity = resourceplus_identity(email=usr_email, instance=instance_name)
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        HOME_DATA_ROUTE,
        params={
            "instanceName": identity.instance,
            "Usremail": identity.email,
            "Lang": lang,
        },
    )
