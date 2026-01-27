import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, schema_path: str = None):
        self.schema_path = schema_path or os.path.join(os.path.dirname(__file__), 'schema.yaml')

    def load(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self._validate(config)
        return config

    def _validate(self, config: Dict[str, Any]):
        """Basic validation against schema structure."""
        # TODO: Implement stricter validation (e.g., using pydantic or jsonschema)
        required_sections = ['experiment', 'dataset', 'prompt', 'model', 'metrics']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
