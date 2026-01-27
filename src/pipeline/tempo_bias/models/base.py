from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelAdapter(ABC):
    def __init__(self, model_name: str, inference_config: Dict[str, Any] = None):
        self.model_name = model_name
        self.inference_config = inference_config or {}

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
