from src.model.base_model import BaseLLMModel
from typing import Optional, Union, Tuple
from google.genai import Client
from pydantic import SecretStr
from pydantic import BaseModel
import os 
from langfuse import observe

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