from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.config import DEFAULT_RESOURCEPLUS_LANG
from app.resourceplus.attendance import get_attendance_summary
from app.resourceplus.employee import get_profile_data
from app.resourceplus.home import get_home_data


router = APIRouter(prefix="/api/debug/resourceplus", tags=["resourceplus-debug"])


@router.get("/attendance")
async def attendance_debug(
    from_date: date,
    to_date: date,
    lang: int = Query(default=DEFAULT_RESOURCEPLUS_LANG, ge=1),
) -> Any:
    if from_date > to_date:
        raise HTTPException(
            status_code=400,
            detail="from_date must be on or before to_date.",
        )
    return await get_attendance_summary(from_date, to_date, lang=lang)


@router.get("/home")
async def home_debug(
    lang: int = Query(default=DEFAULT_RESOURCEPLUS_LANG, ge=1),
) -> Any:
    return await get_home_data(lang=lang)


@router.get("/profile")
async def profile_debug(
    lang: int = Query(default=DEFAULT_RESOURCEPLUS_LANG, ge=1),
) -> Any:
    return await get_profile_data(lang=lang)

