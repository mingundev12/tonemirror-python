from pydantic import BaseModel

# 삭제 성공시 메시지 응답
class ROIDeleteResponse(BaseModel):
    message: str