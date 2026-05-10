from typing import Annotated, TypedDict, Optional, Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# ====================== STATE ======================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Structured extraction
    query_type: Optional[Literal["status", "policy", "combined", "general"]]
    airline_code: Optional[str]
    flight_number: Optional[str]
    flight_date: Optional[str]          # YYYY-MM-DD
    delay_hours: Optional[float]
    cancellation_reason: Optional[str]  # if known

    # Tool / RAG results
    flight_status: Optional[str]        # "delayed", "cancelled", "on_time", etc.
    status_details: Optional[dict]      # full info (gate, new time, etc.)
    policy_snippet: Optional[str]
    compensation_eligible: Optional[bool]
    
    # Final output
    final_answer: Optional[str]
    
    # Governance
    compliance_passed: Optional[bool]
    pii_safe: Optional[bool]
    evaluation_score: Optional[float]


# ====================== NODES ======================

def extract_intent_and_entities(state: AgentState) -> AgentState:
    """Use LLM with structured output (or tool) to classify + extract entities"""
    # In real code: use LLM with Pydantic output parser
    last_msg = state["messages"][-1].content.lower()
    
    # Simple heuristic + LLM fallback recommended
    if any(x in last_msg for x in ["status", "what time", "arrive", "depart"]):
        query_type = "status"
    elif any(x in last_msg for x in ["hotel", "voucher", "meal", "refund", "compensation", "entitled"]):
        query_type = "policy"
    else:
        query_type = "combined"   # safest default
    
    # TODO: Call LLM with structured extraction here (airline, flight num, date, etc.)
    
    return {
        "query_type": query_type,
        # Populate airline_code, flight_number, flight_date, etc.
    }


def fetch_flight_status(state: AgentState) -> AgentState:
    """Call your flight data source (Hive, API, Databricks table, etc.)"""
    # Example:
    # status = run_flight_query(state["airline_code"], state["flight_number"], state["flight_date"])
    
    state["flight_status"] = "delayed"          # placeholder
    state["status_details"] = {"delay_minutes": 180, "reason": "weather"}
    state["delay_hours"] = 3.0
    return state


def retrieve_policy(state: AgentState) -> AgentState:
    """RAG / vector search over airline policies + regulations (DOT, EU261, etc.)"""
    context = ""
    if state.get("flight_status") == "cancelled":
        context = "United Airlines cancellation policy + DOT rules on hotels..."
    elif state.get("delay_hours", 0) >= 3:
        context = "Meal voucher policy for delays over 3 hours..."
    
    state["policy_snippet"] = context
    return state


def decide_entitlements(state: AgentState) -> AgentState:
    """Rule + LLM hybrid decision on what passenger is entitled to"""
    status = state.get("flight_status")
    delay = state.get("delay_hours", 0)
    
    eligible = False
    reason = ""
    
    if status == "cancelled":
        eligible = True
        reason = "Flight cancellation - hotel + rebooking or refund"
    elif delay >= 3:
        eligible = True
        reason = "Significant delay - meal voucher"
    
    state["compensation_eligible"] = eligible
    # You can store detailed reasoning too
    return state


def synthesize_response(state: AgentState) -> AgentState:
    """Final LLM call to generate natural, empathetic, accurate answer"""
    # Pass flight info + policy + entitlement decision to LLM
    prompt = f"""You are a helpful Passenger Advocate...
    Flight: {state.get('airline_code')}{state.get('flight_number')} on {state.get('flight_date')}
    Status: {state.get('flight_status')}
    Policy: {state.get('policy_snippet')}
    Eligible: {state.get('compensation_eligible')}
    """
    
    # response = MODEL.invoke(...)
    state["final_answer"] = "Drafted response here..."
    return state


def compliance_check(state: AgentState) -> AgentState:
    # PII redaction, hallucination check, policy safety, etc.
    state["compliance_passed"] = True
    state["pii_safe"] = True
    return state


def evaluate_response(state: AgentState) -> AgentState:
    # DeepEval, Langfuse scoring, faithfulness, etc.
    state["evaluation_score"] = 0.92
    return state


def final_logging(state: AgentState) -> AgentState:
    # Langfuse trace, logging, analytics
    return state


# ====================== ROUTING ======================

def route_after_intent(state: AgentState) -> str:
    qt = state.get("query_type")
    if qt in ["status", "combined"]:
        return "fetch_status"
    elif qt == "policy":
        return "retrieve_policy"
    return "synthesize"


def route_after_status(state: AgentState) -> str:
    """After getting status, decide if policy lookup is needed"""
    if state.get("query_type") in ["policy", "combined"]:
        return "retrieve_policy"
    return "synthesize"


# ====================== BUILD GRAPH ======================

def build_passenger_advocate():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("extract_intent", extract_intent_and_entities)
    workflow.add_node("fetch_status", fetch_flight_status)
    workflow.add_node("retrieve_policy", retrieve_policy)
    workflow.add_node("decide_entitlements", decide_entitlements)
    workflow.add_node("synthesize", synthesize_response)
    workflow.add_node("compliance", compliance_check)
    workflow.add_node("evaluate", evaluate_response)
    workflow.add_node("log", final_logging)

    # Edges
    workflow.add_edge(START, "extract_intent")
    
    workflow.add_conditional_edges(
        "extract_intent",
        route_after_intent,
        {
            "fetch_status": "fetch_status",
            "retrieve_policy": "retrieve_policy",
            "synthesize": "synthesize"
        }
    )

    workflow.add_edge("fetch_status", "decide_entitlements")   # optional, can be direct
    workflow.add_conditional_edges(
        "fetch_status",
        route_after_status,
        {
            "retrieve_policy": "retrieve_policy",
            "synthesize": "synthesize"
        }
    )

    workflow.add_edge("retrieve_policy", "decide_entitlements")
    workflow.add_edge("decide_entitlements", "synthesize")
    workflow.add_edge("synthesize", "compliance")
    workflow.add_edge("compliance", "evaluate")
    workflow.add_edge("evaluate", "log")
    workflow.add_edge("log", END)

    return workflow.compile()