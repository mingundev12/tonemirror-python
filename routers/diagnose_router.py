# routers/diagnose_router.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.db import get_db

from schemas.preprocess_schema import PreprocessRequest
from schemas.diagnose_schema import DiagnoseResponse

from services.diagnose_service import DiagnoseService

router = APIRouter(
    prefix="/ai",
    tags=["Diagnosis"]
)

@router.post(
    "/diagnose",
    response_model=DiagnoseResponse
)
def diagnose(
    request: PreprocessRequest,
    db: Session = Depends(get_db)
):
    
    service = DiagnoseService(db)

    return service.diagnose(request)