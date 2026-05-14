from langchain_core.messages import HumanMessage
from src.agent import agent_app
from src.utils import run_agent_loop
import traceback
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from src.retrieval import OpenAIEmbeddingModel

load_dotenv()

# from src.template import IntentClassification, tools_to_prompt_string
# from src.tools import policy_search_tool, flight_status_tool

# tools = [flight_status_tool, policy_search_tool]

# toolsStr = tools_to_prompt_string(tools)
# print(toolsStr)

# # create vector embedding.
# embedding = OpenAIEmbeddingModel()
# # embedding.create_embedding()
# results = embedding.search(query="Since it was cancelled, is United required to provide me with a hotel?")

# for result in results:
#     print(result['text'][:300] + "...\n")


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