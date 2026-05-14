# src/model/openai_model.py
from src.model.base_model import BaseLLMModel
from typing import Optional, Tuple, Union, Any
from pydantic import SecretStr, BaseModel
import os
from openai import OpenAI
from langfuse import observe

default_openai_model = "gpt-4o"

class OpenAIModel(BaseLLMModel):

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, temperature: Optional[float] = None):
        if api_key:
            self.api_key = SecretStr(api_key)
        else:
            env_key = os.getenv("OPENAI_API_KEY")
            if not env_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.api_key = SecretStr(env_key)

        self.temperature = float(temperature) if temperature is not None else float(os.getenv("TEMPERATURE", 0.0))
        self.model = model or default_openai_model
        self.client: Optional[OpenAI] = None

        super().__init__(self.model)

    def load_model(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI(api_key=self.api_key.get_secret_value())
        return self.client

    @observe(name="openai llm call", as_type="generation")
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        
        client = self.load_model()

        if schema:
            # === Structured Output using json_schema (more stable) ===
            messages = [{"role": "user", "content": prompt}]
            completion = client.beta.chat.completions.parse(
                    model=self.name,
                    messages=messages,
                    response_format=schema,
                    temperature=self.temperature,
            )
            structured_output: BaseModel = completion.choices[
                    0
            ].message.parsed
            cost = self._calculate_cost(completion)
            return structured_output, cost
            
        else:
            completion = client.chat.completions.create(
                model=self.name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            output = completion.choices[0].message.content

            cost = self._calculate_cost(completion)
            return output, cost

    def _calculate_cost(self, response: Any) -> float:
        usage = getattr(response, "usage", None)
        if usage:
            # Approximate cost for gpt-4o
            return (usage.prompt_tokens * 0.0000025) + (usage.completion_tokens * 0.000010)
        return 0.0