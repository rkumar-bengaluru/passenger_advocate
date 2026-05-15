import threading
import time
from typing import List, Any, Dict
import logging
import json 

logger = logging.getLogger(__name__)

def get_input_with_timeout(prompt, timeout):
    """Get user input with a timeout using threading."""
    user_input = [None]
    input_received = threading.Event()
    
    def input_thread():
        try:
            user_input[0] = input(prompt)
            input_received.set()
        except:
            pass
    
    thread = threading.Thread(target=input_thread, daemon=True)
    thread.start()
    
    if input_received.wait(timeout):
        return user_input[0]
    else:
        return None

def run_agent_loop(agent_app, input_timeout=120):
    """Run interactive agent loop with timeout protection."""
    print("--- Rearc AI Quest: Support Agent ---")
    print("Type 'quit' to exit.")

    while True:
        user_input = get_input_with_timeout("\nUser: ", input_timeout)
        
        if user_input is None:
            print(f"\n⏱️ No input received for {input_timeout} seconds. Exiting...")
            break
        
        if user_input.lower() in ["quit", "exit"]:
            break
        
        # Yield user input for custom handling
        yield user_input
    
    print("\n👋 Session ended.")

def format_policy_snippets(results: List[Dict[str, Any]]) -> str:
    """
    Convert search results into a readable context string for the LLM prompt.
    """
    if not results:
        return "No relevant policy information found."

    formatted = []
    logger.info("results ---%s", json.dumps(results))
    for i, result in enumerate(results, start=1):
        text = result.get("text", "")
        score = result.get("score")
        if score is not None:
            formatted.append(f"[{i}] (score={score:.4f}): {text}")
        else:
            formatted.append(f"[{i}]: {text}")
        
    return "\n".join(formatted)
