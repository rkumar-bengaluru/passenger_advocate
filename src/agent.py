from typing import Annotated, TypedDict, List, Optional, Dict, Any 
from databricks_langchain import ChatDatabricks
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from src.template import IntentClassification, tools_to_prompt_string
from src.model import OpenAIModel, GeminiModel
from src.schema import IntentClassificationResponse
from src.tools import policy_search_tool, flight_status_tool, query_policy_documents
from src.flight_status import get_flight_status
from src.template import build_summary_prompt,build_summary_prompt_for_flight_status
from src.utils import format_policy_snippets
from langchain_core.messages import AIMessage
from langfuse import observe, get_client
import logging
import os 
import json 

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)
# os.environ["LANGFUSE_DEBUG"] = "True"


# Configuration
LLM_ENDPOINT = "databricks-gpt-5-nano"
print(LLM_ENDPOINT)


MODEL = GeminiModel()
# MODEL = ChatDatabricks(endpoint=LLM_ENDPOINT)

def last_value(current: Optional[str], update: Optional[str]) -> Optional[str]:
    return update if update is not None else current


# State
class AgentState(TypedDict):
    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    intent_analysis: Optional[Dict[str, Any]]
    flight_status_history: List[Dict[str, Any]] 
    policy_snippets: List[Dict[str, Any]]   

    query_type: Annotated[Optional[str], last_value]

    # Results from tools
    status: Optional[str]     

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

    analysis = state.get("intent_analysis")
    if analysis:
        selected_action = analysis.get("selected_action")
        ready = analysis.get("ready_to_execute", False)
        logger.info("selected_action %s ready state %s", selected_action, ready)

        if selected_action == "get_flight_status":
            if ready:
                if not state.get("flight_status_history"):
                    raise TypeError(
                    f"Unexpected state: flight_status_history should be there"
                    )
                status = state["flight_status_history"][-1]
                prompt = build_summary_prompt_for_flight_status(state["messages"][-1].content, status)

                print(prompt)

                response, score = MODEL.generate(prompt=prompt, schema=None)
                # Add LLM answer to conversation history
                state["messages"].append(AIMessage(content=response))
                return state

        elif selected_action == "query_policy_documents":
            if ready:
                if not state.get("policy_snippets"):
                    raise TypeError("Unexpected state: policy_snippets should be there")
                context = format_policy_snippets(state["policy_snippets"][-1])
                prompt = build_summary_prompt(state["messages"][-1].content, context)

                print(prompt)

                response, score = MODEL.generate(prompt=prompt, schema=None)
                # Add LLM answer to conversation history
                state["messages"].append(AIMessage(content=response))
                return state
        else:
            raise TypeError(
            f"Unexpected state: selected_action can be only  get_flight_status or query_policy_documents for now"
            )
    
    else:
        raise TypeError(
        f"Unexpected state: analysis cannot be None at this stage "
        )
            
    response = "Model called..."
    logger.info("Node call_model executed | level=log")
    return {"messages": [response]}

@observe(name="flight_status_query", as_type="tool")
def flight_status_query(state: AgentState) -> AgentState:
    # run Hive SQL query

    # Initialize list if not present
    if "flight_status_history" not in state or state["flight_status_history"] is None:
        state["flight_status_history"] = []

    response = get_flight_status("OO", "4277", "2023-12-30")
    
    # Append new response to history
    state["flight_status_history"].append(response)

    logger.info("Node flight_status_query executed | level=log")
    return state

@observe(name="get_policy_details", as_type="tool ")
def get_policy_details(state: AgentState) -> AgentState:
    results: List[Dict[str, Any]] = query_policy_documents(state["messages"][-1].content)
    

    # Initialize list if not present
    if "policy_snippets" not in state or state["policy_snippets"] is None:
        state["policy_snippets"] = []

    state["policy_snippets"].append(results)  
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

    schema = IntentClassificationResponse
    response, score = MODEL.generate(prompt,schema=schema)

    if isinstance(response, IntentClassificationResponse):
        logger.info("Valid response: %s", json.dumps(response.model_dump(), ensure_ascii=False))
    else:
        raise TypeError(
        f"Unexpected response type: {type(response).__name__}. "
        "Expected IntentClassificationResponse."
        )

    logger.info("final response %s", response)
    
    state["intent_analysis"] = (
        response.model_dump().get("analysis") if isinstance(response, IntentClassificationResponse) else response
    )
    analysis = state.get("intent_analysis")
    logger.info("analysis in state: %s", json.dumps(analysis, ensure_ascii=False))
    return state

def route_from_intent_from_analysis(state: AgentState) -> str:
    analysis = state.get("intent_analysis")
    logger.info("analyzing intent %s", json.dumps(analysis))
    if analysis:
        selected_action = analysis.get("selected_action")
        ready = analysis.get("ready_to_execute", False)
        logger.info("selected_action %s ready state %s", selected_action, ready)
        if selected_action == "get_flight_status":
            return "get_flight_status" if ready else "summarize_answer"

        elif selected_action == "query_policy_documents":
            return "get_policy_details" if ready else "summarize_answer"

        else:
            logger.info("selected_action %s ready state %s", selected_action, ready)
            return "summarize_answer"
    else:
        logger.info("returning.... %s", "summarize_answer")
        return "summarize_answer"

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
        route_from_intent_from_analysis,
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