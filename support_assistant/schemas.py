from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str

class PolicyResponse(BaseModel):
    answer: str = Field(description="Direct response text")
    sources: List[str] = Field(default_factory=list, description="IDs of source docs used")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
