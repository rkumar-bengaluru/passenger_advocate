
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any


from pydantic import SecretStr
from pydantic import BaseModel
from langfuse import observe

from openai import OpenAI
from google.genai import Client
from databricks_langchain import ChatDatabricks

from langchain_core.messages import HumanMessage

import os 


class BaseLLMModel(ABC):
    def __init__(self, model: Optional[str] = None, *args, **kwargs):
        self.name = model 
        self.model = self.load_model()

    @abstractmethod
    def load_model(self, *args, **kwargs) -> "BaseLLMModel":
        pass

    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        pass

    def get_model_name(self, *args, **kwargs) -> str:
        return self.name
    
    def generate_with_schema(self, *args, schema=None, **kwargs):
        if schema is not None:
            try:
                return self.generate(*args, schema=schema, **kwargs)
            except TypeError:
                pass 
        return self.generate(*args, **kwargs)



default_gemini_model = "gemini-2.5-pro"

class GeminiModel(BaseLLMModel):

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        # API key handling
        if api_key is not None:
            # keep it secret, keep it safe
            self.api_key: SecretStr = SecretStr(api_key)
        else:
            env_key = os.getenv("GOOGLE_API_KEY")
            if not env_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file")
            self.api_key = SecretStr(env_key)

        # Temperature handling
        if temperature is not None:
            self.temperature = float(temperature)
        elif os.getenv("TEMPERATURE") is not None:
            self.temperature = float(os.getenv("TEMPERATURE"))
        else:
            self.temperature = 0.0

        # Model name
        self.model = model or default_gemini_model

        # Client placeholder
        self.client: Optional[Client] = None

        super().__init__(self.model)


    def load_model(self) -> Client:
        """
        Initialize and return the Gemini client.
        """
        if not self.api_key:
            raise ValueError("Gemini API key is required to initialize the client.")

        # Initialize Google GenAI client
        self.client = Client(api_key=self.api_key.get_secret_value())

        return self.client
    
    @observe(name="gemini llm call", as_type="generation")
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        client = self.load_model()
        if schema is not None:
            response = client.models.generate_content(
                model=self.name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": self.temperature,
                },
            )
            return response.parsed, 0
        else:
            response = client.models.generate_content(
                model=self.name,
                contents=prompt,
               config={
                    "temperature": self.temperature,
                },
            )
            return response.text, 0
        

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