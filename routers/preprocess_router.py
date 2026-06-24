from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database.db import get_db

from schemas.preprocess_schema import PreprocessRequest
from schemas.preprocess_schema import PreprocessResponse

from services.preprocess_service import PreprocessService

router = APIRouter(
    prefix="/ai",
    tags=["Preprocessing"]
)

@router.post(
    "/preprocess",
    response_model=PreprocessResponse
)

def preprocess(request: PreprocessRequest, db: Session = Depends(get_db)):
    
    service = PreprocessService(db)

    result = service.preprocess(request)

    return result