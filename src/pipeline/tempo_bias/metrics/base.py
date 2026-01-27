from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetric(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, predictions: Dict[str, Any]) -> Any:
        pass
