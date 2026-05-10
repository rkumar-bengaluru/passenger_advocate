from typing import Annotated, TypedDict, List, Optional
from databricks_langchain import ChatDatabricks
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.logger import PassengerAdvocateLogger

from langfuse import observe, get_client
import logging
import os 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
os.environ["LANGFUSE_DEBUG"] = "True"

# logger = PassengerAdvocateLogger(
# )

# Configuration
LLM_ENDPOINT = "databricks-gpt-5-nano"
print(LLM_ENDPOINT)
# MODEL = ChatDatabricks(endpoint=LLM_ENDPOINT)

def last_value(current: Optional[str], update: Optional[str]) -> Optional[str]:
    return update if update is not None else current


# State
class AgentState(TypedDict):
    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    query_type: Annotated[Optional[str], last_value]

    # Structured fields for workflow
    airline_code: Optional[str]
    flight_number: Optional[str]
    flight_date: Optional[str]

    # Results from tools
    status: Optional[str]           
    policy_snippet: Optional[str]   
    final_answer: Optional[str]     

    # Governance & monitoring
    compliance_passed: Optional[bool]
    pii_safe: Optional[bool]
    evaluation_score: Optional[float]

# Nodes
def call_model(state: AgentState):
    """
    Basic node that invokes the LLM.
    Currently, this has no access to tools.
    """
    # response = MODEL.invoke(state["messages"])
    response = "Model called..."
    logger.info("Node call_model executed | level=log")
    return {"messages": [response]}

def flight_status_query(state: AgentState) -> AgentState:
    # run Hive SQL query
    state["status"] = "Delayed by 45 minutes"
    logger.info("Node flight_status_query executed | level=log")
    return state

@observe(name="rag_retrieval", as_type="span")
def rag_retrieval(state: AgentState) -> AgentState:
    state["policy_snippet"] = "Airlines must provide timely updates for delays."
    logger.info("rag_retrieval executed")
    return state

@observe(name="llm_synthesis", as_type="span")
def llm_synthesis(state: AgentState) -> AgentState:
    state["final_answer"] = f"Flight {state.get('airline_code')}{state.get('flight_number')} " \
                           f"on {state.get('flight_date')} was {state.get('status')}. " \
                           f"{state.get('policy_snippet')}"
    logger.info("llm_synthesis executed")
    return state


def compliance_check(state: AgentState) -> AgentState:
    logger.info("Node compliance_check executed | level=log")
    # run policy compliance + PII filter
    return state

def evaluation(state: AgentState) -> AgentState:
    logger.info("Node evaluation executed | level=log")
    # integrate DeepEval metrics
    return state

def final_logging(state: AgentState) -> AgentState:
    logger.info("Node logging executed | level=log")
    # send traces to Langfuse
    return state

def intent_classification(state: AgentState) -> AgentState:
    logger.info("intent_classification | level=int | keys=%s", list(state.keys()))

    if not state.get("messages"):
        state["query_type"] = "general"
        return state

    last_message = state["messages"][-1].content.lower()

    if "status" in last_message or "flight" in last_message:
        state["query_type"] = "flight_status"
    elif any(k in last_message for k in ["policy", "refund", "delay"]):
        state["query_type"] = "rag"
    else:
        state["query_type"] = "general"

    return state


def route_from_intent(state: AgentState) -> str:
    if state["query_type"] == "flight_status":
        return "flight_status"
    elif state["query_type"] == "policy":
        return "rag"
    else:
        return "synthesis"  # fallback


# Build workflow
def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent", intent_classification)
    workflow.add_node("flight_status", flight_status_query)
    workflow.add_node("rag", rag_retrieval)
    workflow.add_node("synthesis", call_model)
    workflow.add_node("compliance", compliance_check)
    workflow.add_node("evaluation", evaluation)
    workflow.add_node("logging", final_logging)

    # Start → Intent
    workflow.add_edge(START, "intent")

    # Conditional routing from intent
    workflow.add_conditional_edges(
        "intent",
        route_from_intent,
        {
            "flight_status": "flight_status",
            "rag": "rag",
            "synthesis": "synthesis"
        }
    )

    # Downstream edges
    workflow.add_edge("flight_status", "synthesis")
    workflow.add_edge("rag", "synthesis")
    workflow.add_edge("synthesis", "compliance")
    workflow.add_edge("compliance", "evaluation")
    workflow.add_edge("evaluation", "logging")
    workflow.add_edge("logging", END)

    return workflow.compile()


agent_app = build_agent()