from typing import Annotated, TypedDict, List, Optional
from databricks_langchain import ChatDatabricks
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.template import IntentClassification, tools_to_prompt_string
from src.model import OpenAIModel, GeminiModel
from src.schema import IntentClassificationResponse
from src.tools import policy_search_tool, flight_status_tool, query_policy_documents
from src.flight_status import get_flight_status

from langfuse import observe, get_client
import logging
import os 
import json 

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

@observe(name="summarize_answer", as_type="generation")
def summarize_answer(state: AgentState):
    """
    Basic node that invokes the LLM.
    Currently, this has no access to tools.
    """
    # response = MODEL.invoke(state["messages"])
    response = "Model called..."
    logger.info("Node call_model executed | level=log")
    return {"messages": [response]}

@observe(name="flight_status_query", as_type="tool")
def flight_status_query(state: AgentState) -> AgentState:
    # run Hive SQL query
    state["status"] = "Delayed by 45 minutes"
    logger.info("Node flight_status_query executed | level=log")
    return state

@observe(name="get_policy_details", as_type="tool ")
def get_policy_details(state: AgentState) -> AgentState:
    state["policy_snippet"] = "Airlines must provide timely updates for delays."
    logger.info("get_policy_details executed")
    return state

@observe(name="policy_guardrails", as_type="guardrail")
def policy_guardrails(state: AgentState) -> AgentState:
    logger.info("Node compliance_check executed | level=log")
    # run policy compliance + PII filter
    return state

@observe(name="pii_check", as_type="guardrail")
def pii_check(state: AgentState) -> AgentState:
    logger.info("Node evaluation executed | level=log")
    # integrate DeepEval metrics
    return state

@observe(name="intent classification", as_type="span")
def intent_classification(state: AgentState) -> AgentState:
    logger.info("intent_classification | level=int | keys=%s", list(state.keys()))
    template = IntentClassification()
    tools = [flight_status_tool, policy_search_tool]
    toolsStr = tools_to_prompt_string(tools)
    prompt = template.generate_intent_classification_prompt_v1(
        agentName = "Passenger Advocate Agent",
        description = "Identify which tool is required based on the query",
        query = state["messages"][-1].content,
        context="",
        toolsStr = toolsStr,
    )

    model = GeminiModel()
    schema = IntentClassificationResponse
    response = model.generate(prompt,schema=schema)

    if isinstance(response, IntentClassificationResponse):
        logger.info("Valid response: %s", json.dumps(response, ensure_ascii=False))
    else:
        logger.error("Unexpected response type: %s", type(response))

    logger.info("final response %s", response)
    state["intent_analysis"] = (
    response.dict() if isinstance(response, IntentClassificationResponse) else response
    )
    if not state.get("messages"):
        state["query_type"] = "general"
        return state

    last_message = state["messages"][-1].content.lower()

    if "status" in last_message or "flight" in last_message:
        state["query_type"] = "get_flight_status"
    elif any(k in last_message for k in ["policy", "refund", "delay"]):
        state["query_type"] = "get_policy_details"
    else:
        state["query_type"] = "general"

    # Return response so Langfuse logs it automatically
   
    return state

def route_from_intent(state: AgentState) -> str:
    if state["query_type"] == "get_flight_status":
        return "get_flight_status"
    elif state["query_type"] == "get_policy_details":
        return "get_policy_details"
    else:
        return "summarize_answer"  # fallback
    
def route_from_intent_from_analysis(state: AgentState) -> str:
  
    # Otherwise fall back to intent_analysis
    analysis = None
    if "intent_analysis" in state and state["intent_analysis"]:
        analysis = state["intent_analysis"].get("analysis")

    if analysis:
        selected_action = analysis.get("selected_action")
        ready = analysis.get("ready_to_execute", False)
        # future use next steps, missing parameter cases.

        if selected_action == "get_flight_status":
            if ready:
                return "get_flight_status" 
            else:
                return "summarize_answer"       

        elif selected_action == "query_policy_documents":
            if ready:
                return "get_policy_details"
            else:
                return "summarize_answer"

        else:
            return "summarize_answer" 
    else:
        return "summarize_answer"   



# Build workflow
def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_classification", intent_classification)
    workflow.add_node("get_flight_status", flight_status_query)
    workflow.add_node("get_policy_details", get_policy_details)
    workflow.add_node("summarize_answer", summarize_answer)
    workflow.add_node("policy_guardrails", policy_guardrails)
    workflow.add_node("pii_check", pii_check)

    # Start → Intent
    workflow.add_edge(START, "intent_classification")

    # Conditional routing from intent
    workflow.add_conditional_edges(
        "intent_classification",
        route_from_intent,
        {
            "get_flight_status": "get_flight_status",
            "get_policy_details": "get_policy_details",
            "summarize_answer": "summarize_answer"
        }
    )
    # Downstream edges
    workflow.add_edge("get_flight_status", "summarize_answer")
    workflow.add_edge("get_policy_details", "summarize_answer")
    workflow.add_edge("summarize_answer", "policy_guardrails")
    workflow.add_edge("policy_guardrails", "pii_check")
    workflow.add_edge("pii_check", END)

    return workflow.compile()


agent_app = build_agent()