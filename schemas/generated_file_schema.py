from pydantic import BaseModel
from datetime import datetime

# Analysis 전달용
class ROIResponse(BaseModel):

    file_type: str
    file_url: str