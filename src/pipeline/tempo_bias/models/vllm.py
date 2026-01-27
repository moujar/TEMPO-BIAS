from .base import ModelAdapter
import logging

class VLLMAdapter(ModelAdapter):
    def __init__(self, model_name: str, inference_config: dict = None):
        super().__init__(model_name, inference_config)
        self.logger = logging.getLogger(f"VLLMAdapter-{model_name}")
        
        try:
            from vllm import LLM
            self.logger.info(f"Initializing vLLM with model {model_name}...")
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,  # Needed for some models like Falcon/Qwen
                gpu_memory_utilization=0.9
            )
        except ImportError:
            self.logger.error("vLLM is not installed. Please install it with: pip install vllm")
            raise ImportError("vLLM is required for this model but not installed")
        except Exception as e:
            self.logger.error(f"Failed to initialize vLLM with model {model_name}: {e}")
            raise e

    def generate(self, prompt: str) -> str:
        try:
            from vllm import SamplingParams
            
            # Merge inference config with defaults
            sampling_params = SamplingParams(
                max_tokens=self.inference_config.get("max_tokens", 100),
                temperature=self.inference_config.get("temperature", 0),
                top_p=self.inference_config.get("top_p", 1.0),
            )
            
            outputs = self.llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise e
