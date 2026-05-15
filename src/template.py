import textwrap
from textwrap import dedent
from pydantic import BaseModel
import typing
import json 

def tools_to_prompt_string(tools) -> str:
    blocks = []
    for tool in tools:
        block = dedent(f"""
        #### {tool.name}
        **Description:** {tool.description.strip()}
        **Input Parameters:**
        {schema_to_string(tool.args_schema)}
        """)
        blocks.append(block)
    return "\n".join(blocks)


def schema_to_string(schema: type[BaseModel]) -> str:
    """Works with the model class itself, not an instance."""
    lines = []
    for name, field in schema.model_fields.items():
        # Required check
        required = "required" if field.is_required() else "optional"
        
        # Get clean type name
        annotation = field.annotation
        if hasattr(annotation, "__name__"):
            type_name = annotation.__name__
        elif typing.get_origin(annotation):
            type_name = str(annotation)
        else:
            type_name = str(annotation)
        
        desc = field.description or ""
        
        lines.append(f"- `{name}` ({type_name}, {required}) : {desc}")
    
    return "\n".join(lines)

def build_summary_prompt(user_query: str, retrieved_context: str) -> str:
    return dedent(f"""
    You are an intelligent assistant. Your task is to summarize information retrieved from the knowledge base
    in response to the user's query.

    ## User Query
    {user_query}

    ## Retrieved Context
    {retrieved_context}

    ## Instructions
    - Provide a concise, clear summary that directly answers the user's query.
    - Highlight the most relevant points from the retrieved context.
    - Do not include irrelevant details.
    - If the context does not contain enough information, state that clearly.

    ## Final Answer
    """)

def build_summary_prompt_for_flight_status(user_query: str, response: dict) -> str:
    return dedent(f"""
    You are a Passenger Advocate Agent. The user asked: "{user_query}"

    ## Tool Output Context
    {json.dumps(response, indent=2, ensure_ascii=False)}

    ## Task
    Summarize the above tool output into a clear, user-facing answer.
    - Mention the flight status, delays, cancellations, or policies as appropriate.
    - Use plain language, not JSON.
    - Keep it concise but informative.
    """)

class IntentClassification:

    def generate_intent_classification_prompt_v1(self,
            agentName: str,
            description: str,
            query: str, 
            context: str, 
            toolsStr: str):
        return textwrap.dedent(f"""You are an intelligent {agentName} with the following description: {description}.

            ## Context from Knowledge Base
            {context}

            ## Available Tools
            You can call the following tools to help the user:
            {toolsStr}

            ## Instructions

            ### Step 1: Intent Classification & Confidence Scoring
            Analyze the user query and determine the **top 3 most likely actions** with confidence scores (0.0 to 1.0).

            **Confidence Guidelines:**
            - 1.0: Query explicitly matches action with all required parameters present
            - 0.95–0.99: Very strong match...
            - ... (keep your guidelines)

            ### Step 2: Parameter Extraction & Gap Analysis
            (keep your instructions)

            ### Step 3: Response Format

            You **must** respond with **only** valid JSON matching the following schema. No explanations, no markdown, no extra text.

            ```json
            {{
            "analysis": {{
                "top_actions": [
                {{
                    "tool": "get_flight_status",
                    "confidence": 0.95,
                    "reasoning": "Query explicitly asks for flight status..."
                }},
                {{
                    "tool": "query_policy_documents",
                    "confidence": 0.1,
                    "reasoning": "Not relevant here"
                }}
                ],
                "selected_action": "get_flight_status",
                "parameters": {{
                "satisfied": {{
                    "airline": "Xyz",
                    "flight_number": "X01",
                    "flight_date": "12 Jan 2004"
                }},
                "missing_required": ["flight_date"],
                }},
                "ready_to_execute": false,
                "next_step": "collect_params",
                "clarification_needed": "Please provide the full date including year."
            }}
            }} 
            User query: "{query}"
                    JSON:
                    """
        )
