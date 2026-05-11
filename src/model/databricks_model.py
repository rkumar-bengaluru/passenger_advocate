# src/model/databricks_model.py
from src.model.base_model import BaseLLMModel
from typing import Optional, Tuple, Union, Any
from pydantic import SecretStr, BaseModel
import os

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage

default_databricks_model = "databricks-gpt-5-nano"

class DataBricksModel(BaseLLMModel):

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        self.temperature = (
            float(temperature)
            if temperature is not None
            else float(os.getenv("TEMPERATURE", 0.0))
        )

        self.model = model or default_databricks_model
        self.endpoint = self.model

        self.llm: Optional[ChatDatabricks] = None

        super().__init__(self.model)

    def load_model(self) -> ChatDatabricks:
        """Lazy initialization"""
        if self.llm is None:
            self.llm = ChatDatabricks(
                endpoint=self.endpoint,
                temperature=self.temperature,
            )
        return self.llm

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        
        llm = self.load_model()

        if schema:
            # Structured Output (Recommended)
            try:
                structured_llm = llm.with_structured_output(
                    schema, 
                    method="json_schema"      # or "json_mode"
                )
                parsed_output = structured_llm.invoke(prompt)
                
                cost = self._calculate_cost(parsed_output)
                return parsed_output, cost

            except Exception:
                # Fallback
                structured_llm = llm.with_structured_output(schema, method="json_mode")
                parsed_output = structured_llm.invoke(prompt)
                cost = self._calculate_cost(parsed_output)
                return parsed_output, cost

        else:
            # Plain text generation
            response = llm.invoke([HumanMessage(content=prompt)])
            
            text = response.content if hasattr(response, "content") else str(response)
            cost = self._calculate_cost(response)
            
            return text, cost

    def _calculate_cost(self, response: Any) -> float:
        """Extract token usage if available"""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            
            # Adjust pricing based on your actual model
            return (prompt_tokens * 0.0000025) + (completion_tokens * 0.000010)
        
        return 0.0