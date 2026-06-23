from pydantic import BaseModel
from typing import List

from schemas.generated_file_schema import ROIResponse

class PreprocessRequest(BaseModel):

    file_id: int
    file_url: str

class PreprocessResponse(BaseModel):

    message: str
    files: List[ROIResponse]