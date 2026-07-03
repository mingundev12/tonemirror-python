# schemas/diagnose_schema.py

from pydantic import BaseModel
from typing import List

from schemas.generated_file_schema import ROIResponse

class DiagnoseResponse(BaseModel):
    message: str

    original_image_id: int

    files: List[ROIResponse]

    personal_color: str

    detected_skin_hex: str