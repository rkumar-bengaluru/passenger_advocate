from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class TopAction(BaseModel):
    tool: str = Field(..., description="Name of the tool")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation why this tool matches the query")

class FlightStatusArgs(BaseModel):
    airline: Optional[str] = Field(..., description="Airline Name or null if not provided")
    flight_number: Optional[str] = Field(..., description="Flight number string or null if not provided")
    flight_date: Optional[str] = Field(..., description="Date of the flight or null if not provided")

class PolicyDocumentsArgs(BaseModel):
    query: Optional[str] = Field(..., description="The semantic search query string or null if not provided")
    top_k: Optional[int] = Field(..., description="Number of document fragments to fetch or null if not provided")

class ExplicitParameters(BaseModel):
    get_flight_status_args: FlightStatusArgs = Field(
        ..., description="Extracted arguments specifically matching the get_flight_status tool configuration"
    )
    query_policy_documents_args: PolicyDocumentsArgs = Field(
        ..., description="Extracted arguments specifically matching the query_policy_documents tool configuration"
    )
    
    missing_required: List[str] = Field(..., description="Required parameter keys missing from the query")
    missing_optional: List[str] = Field(..., description="Optional parameter keys missing from the query")

class Analysis(BaseModel):
    top_actions: List[TopAction] = Field(..., min_length=1)
    selected_action: Literal["get_flight_status", "query_policy_documents"] = Field(..., description="Chosen tool name")
    parameters: ExplicitParameters = Field(..., description="Evaluated parameter structural breakdown")
    ready_to_execute: bool = Field(..., description="True if all required parameters are non-null for the selected tool")
    next_step: Literal["execute", "collect_params", "clarify"] = Field(..., description="Action routing state")
    clarification_needed: Optional[str] = Field(..., description="The context request sentence, or null if ready to execute")

class IntentClassificationResponse(BaseModel):
    analysis: Analysis = Field(..., description="Complete intent tracking analysis payload")
