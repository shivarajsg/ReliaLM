import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

# ==========================================
# PHASE 1: Structured Output Schema
# ==========================================

class Phase1Output(BaseModel):
    issue_type: str = Field(
        ..., 
        description="The type of the issue, e.g., 'authentication', 'performance', 'ui_bug', 'data_loss', 'database', etc."
    )
    root_cause: str = Field(
        ..., 
        description="Short phrase identifying the root cause, e.g., 'expired_jwt', 'race_condition', 'null_pointer', 'query_timeout', etc."
    )
    priority: str = Field(
        ..., 
        description="One of: 'low', 'medium', 'high', 'critical'"
    )
    affected_component: str = Field(
        ..., 
        description="The affected component, e.g., 'login_service', 'payment_gateway', 'user_db', 'frontend_header', etc."
    )

    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        valid_priorities = {"low", "medium", "high", "critical"}
        if v.lower() not in valid_priorities:
            raise ValueError(f"Priority must be one of {valid_priorities}, got '{v}'")
        return v.lower()


# ==========================================
# PHASE 2: Function-Calling Schemas
# ==========================================

class ToolCall(BaseModel):
    tool: str = Field(..., description="The name of the tool to invoke")
    parameters: Dict[str, Any] = Field(..., description="Key-value arguments for the tool")
