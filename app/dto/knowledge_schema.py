from pydantic import BaseModel
from typing import Optional

class AskQuestion(BaseModel):
    question: str

class deleteKnowledgeBase(BaseModel):
    doc_uuid: Optional[str] = None