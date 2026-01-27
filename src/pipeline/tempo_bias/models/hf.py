import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .base import ModelAdapter
import logging

class HuggingFaceAdapter(ModelAdapter):
    def __init__(self, model_name: str, inference_config: dict = None):
        super().__init__(model_name, inference_config)
        self.logger = logging.getLogger(f"HFAdapter-{model_name}")
        
        # Load model and tokenizer
        # Note: In a real production environment, we should handle device_map more carefully
        # and possibly quantization (bitsandbytes) if requested.
        # For now, we default to whatever accelerator is available (MPS on Mac, CUDA on Linux)
        
        device_map = "auto"
        if torch.backends.mps.is_available():
            device_map = "mps" 
        elif not torch.cuda.is_available():
            device_map = "cpu"
            
        self.logger.info(f"Loading model {model_name} on {device_map}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map=device_map,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
                trust_remote_code=True # Needed for some models like Falcon/Qwen
            )
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise e

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Merge inference config with defaults
        gen_kwargs = {
            "max_new_tokens": self.inference_config.get("max_tokens", 100),
            "do_sample": False, # Deterministic by default as per spec
            "temperature": None, # Ignored when do_sample=False
            "top_p": None,
            "use_cache": False,  # Disable cache to avoid compatibility issues with some models
        }
        
        # If temperature > 0 is explicitly requested, enable sampling
        req_temp = self.inference_config.get("temperature", 0)
        if req_temp > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = req_temp
            gen_kwargs["top_p"] = self.inference_config.get("top_p", 0.95)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing: remove the prompt from the output if the model repeats it
        # (Standard HF behavior typically includes the prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
            
        return generated_text.strip()
