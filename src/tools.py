

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

from typing import List, Dict, Any
from src.retrieval import OpenAIEmbeddingModel
from src.flight_status import get_flight_status, GetFlightStatusInput
import logging

# Initialize the embedding model once (outside the function)
_embedding_model = None

def get_embedding_model():
    """Singleton pattern for the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAIEmbeddingModel()
    return _embedding_model


def query_policy_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search company policy documents using semantic search.
    
    Args:
        query: The user's question or search query
        top_k: Number of most relevant documents to return
    
    Returns:
        List of document chunks with metadata (usually containing 'content', 'score', etc.)
    """
    if not query or not query.strip():
        return [{"error": "Query cannot be empty"}]
    
    if top_k < 1 or top_k > 20:
        top_k = 5  # clamp to reasonable range
    
    try:
        model = get_embedding_model()
        results = model.search(query=query.strip(), top_k=top_k)
        
        # Optional: Add some post-processing
        for result in results:
            if isinstance(result, dict) and 'score' in result:
                result['relevance_score'] = round(float(result['score']), 4)
        
        return results
        
    except Exception as e:
        logging.error(f"Policy document search failed for query: {query[:100]}", exc_info=True)
        return [{
            "error": "Search failed. Please try again later.",
            "details": str(e) if __debug__ else None
        }]
    


policy_search_tool = StructuredTool.from_function(
    func=query_policy_documents,
    name="query_policy_documents",
    description="""Useful for answering questions about company policies, rules, guidelines, 
    and procedures. Use this tool when the user asks about any of the following topics:

    - Air fares, pricing, fare rules, refunds, and ticket changes
    - Flight schedules, ticketing, reservations, and modifications
    - Delayed/canceled flights, passenger rights, compensation, and rebooking
    - Overbooking and involuntary bumping policies
    - Baggage allowances, fees, lost/damaged baggage
    - Smoking rules and restrictions
    - Passengers with disabilities and special assistance
    - Frequent flyer / mileage programs and rewards
    - Contract of carriage and legal terms
    - Travel scams, fraud prevention, and consumer protection
    - Health and medical requirements
    - Safety and security procedures
    - How to file complaints and escalation process

    This tool performs semantic search over internal policy documents and returns 
    the most relevant excerpts with relevance scores.""",
)


flight_status_tool = StructuredTool.from_function(
    func=get_flight_status,
    name="get_flight_status",
    description="""Useful for retrieving the status of a specific commercial flight 
    (departure/arrival time, delay, cancellation, diversion, etc.) on a given date. 
    Use this tool when the user asks about a particular flight like UA123 on 2023-01-01.""",
    args_schema=GetFlightStatusInput
)

