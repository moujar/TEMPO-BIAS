import os
from .base import ModelAdapter

class OpenAIAdapter(ModelAdapter):
    def __init__(self, model_name: str, inference_config: dict = None, api_key: str = None):
        super().__init__(model_name, inference_config)
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        
    def generate(self, prompt: str) -> str:
        # Simplistic implementation - can be expanded for chat vs completion models
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.inference_config.get("temperature", 0),
            max_tokens=self.inference_config.get("max_tokens", 100),
            **{k:v for k,v in self.inference_config.items() if k not in ["temperature", "max_tokens"]}
        )
        return response.choices[0].message.content.strip()
