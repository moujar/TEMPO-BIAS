from typing import List, Optional
from dataclasses import dataclass

@dataclass
class PromptSpec:
    instruction: str
    few_shot_examples: List[dict]  # List of {input: str, output: str} or similar
    label_space: List[str]
    language: Optional[str] = "en"
    system_prompt: Optional[str] = None  # System prompt for models that support it

class PromptBuilder:
    def __init__(self, spec: PromptSpec):
        self.spec = spec

    def build(self, template: str, entity: str) -> str:
        """
        Construct the final prompt.
        1. System prompt (if provided, will be handled separately for chat models)
        2. Instruction
        3. Few-shot examples
        4. Query
        """
        # Substitute entity into template
        query = template.replace("{entity}", entity)
        
        parts = []
        
        # 1. System prompt (included in text for non-chat models)
        if self.spec.system_prompt:
            parts.append(self.spec.system_prompt)
            parts.append("\n")
        
        # 2. Instruction
        parts.append(self.spec.instruction)
        
        # 3. Few-shot examples
        if self.spec.few_shot_examples:
            parts.append("\nExamples:")
            for ex in self.spec.few_shot_examples:
                # Assuming simple format for now, can be sophisticated
                parts.append(f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}")
                
        # 4. Query
        parts.append(f"\nInput: {query}\nOutput:")
        
        return "\n".join(parts)
    
    def get_system_prompt(self) -> Optional[str]:
        """Get system prompt separately for models that support it (e.g., OpenAI chat models)"""
        return self.spec.system_prompt
