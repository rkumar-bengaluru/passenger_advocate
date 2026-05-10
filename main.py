from langchain_core.messages import HumanMessage
from src.agent import agent_app
from src.utils import run_agent_loop
import traceback
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()

for user_input in run_agent_loop(agent_app):
    try:
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        config = {
            "callbacks": [langfuse_handler],
            "run_name": "PassengerAdvocateAgent",           # ← Clean trace name
            "metadata": {                                   # ← Extra useful info
                "user_query": user_input,
                "environment": "development"
            }
        }

        for event in agent_app.stream(initial_state, config=config):
            # Print agent response
            for value in event.values():
                if "messages" in value and value["messages"]:
                    last_msg = value["messages"][-1]
                    if hasattr(last_msg, "content"):
                        print(f"Agent: {last_msg.content}")

        get_client().flush()

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()