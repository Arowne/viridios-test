from typing import List
from fastapi import status, APIRouter, Response
from app.models.input import Input
from app.utils.pricing_computation import pricing_computation

router = APIRouter()


@router.post("")
async def pricing(data: Input):
    pricings = pricing_computation(data)
    return pricings