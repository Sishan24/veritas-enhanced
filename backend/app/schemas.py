from typing import Optional

from pydantic import BaseModel, Field


class EvidenceDoc(BaseModel):
    id: str = "doc"
    text: str


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=20000)
    evidence: Optional[list[EvidenceDoc]] = None  # None -> use the default evidence collection
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    weighted: Optional[bool] = False


class OcclusionRequest(BaseModel):
    claim: str
    evidence: str
