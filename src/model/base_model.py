
from abc import ABC, abstractmethod
from typing import Optional

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

        