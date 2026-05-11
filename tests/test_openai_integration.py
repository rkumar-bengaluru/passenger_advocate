# tests/test_gemini_integration.py
import os
import pytest
from pydantic import BaseModel
from src.model import OpenAIModel
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)


load_dotenv()  # load .env file

# Example schema for structured output
class ExampleSchema(BaseModel):
    reason: str
    score: int

@pytest.fixture
def openai_model():
    # Ensure GOOGLE_API_KEY is set in environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment")
    print(api_key)
    return OpenAIModel(api_key=api_key)

def test_generate_text_response(openai_model):
    prompt = "Write a short haiku about clouds."
    output, score = openai_model.generate(prompt)
    logging.info("Text response: %s", output)
    assert isinstance(output, str)
    assert len(output) > 0
    print("Text response:", output)

def test_generate_schema_response(openai_model):
    prompt = "Give me a reason and a score between 1 and 10."
    schema = ExampleSchema
    output, score = openai_model.generate(prompt, schema=schema)

    # ✅ Type check
    assert isinstance(output, ExampleSchema)

    # ✅ Direct attribute access
    reason = output.reason
    score_value = output.score

   # Log values
    print("Reason: %s", reason)
    print("Score: %d", score_value)
                 
    assert isinstance(output.reason, str)
    assert isinstance(output.score, int)
