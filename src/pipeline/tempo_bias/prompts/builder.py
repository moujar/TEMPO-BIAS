from typing import List, Optional
from dataclasses import dataclass

@dataclass
class PromptSpec:
    instruction: str
    few_shot_examples: List[dict]  # List of {input: str, output: str} or similar
    label_space: List[str]
    language: Optional[str] = "en"

class PromptBuilder:
    def __init__(self, spec: PromptSpec):
        self.spec = spec

    def build(self, template: str, entity: str) -> str:
        """
        Construct the final prompt.
        1. Instruction
        2. Few-shot examples
        3. Query
        """
        # Substitute entity into template
        query = template.replace("{entity}", entity)
        
        parts = []
        
        # 1. Instruction
        parts.append(self.spec.instruction)
        
        # 2. Few-shot examples
        if self.spec.few_shot_examples:
            parts.append("\nExamples:")
            for ex in self.spec.few_shot_examples:
                # Assuming simple format for now, can be sophisticated
                parts.append(f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}")
                
        # 3. Query
        parts.append(f"\nInput: {query}\nOutput:")
        
        return "\n".join(parts)
