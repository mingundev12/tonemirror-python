from fastapi import APIRouter, Depends

from database.db import get_db

from services.roi_service import ROIService
from schemas.roi_schema import (
    ROIDeleteResponse
)

router = APIRouter(
    prefix="/api/roi",
    tags=["ROI"]
)

@router.delete(
    "/{parent_file_id}",
    response_model=ROIDeleteResponse
)
def delete_roi(
    parent_file_id: int,
    db=Depends(get_db)
):
    
    service = ROIService(db)

    result = service.delete_roi(parent_file_id)

    return result