from typing import List, Dict
import hashlib
from datetime import datetime
from tqdm import tqdm
from ..models.base import ModelAdapter
from ..metrics.base import BaseMetric
from ..outputs.writers import OutputWriter

class PipelineRunner:
    def __init__(
        self,
        dataset,
        prompt_builder,
        model: ModelAdapter,
        metrics: List[BaseMetric],
        output_writer: OutputWriter,
        config: Dict
    ):
        self.dataset = dataset
        self.prompt_builder = prompt_builder
        self.model = model
        self.metrics = metrics
        self.output_writer = output_writer
        self.config = config

    def run(self):
        all_responses = []
        predictions_by_sentence = {} # {sentence_id: [labels]}

        for row in tqdm(self.dataset, desc="Processing Dataset"):
            # 1. Build Prompt
            prompt = self.prompt_builder.build(row['template'], row['entity'])
            prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
            
            # 2. Query Model
            raw_response = self.model.generate(prompt)
            
            # 3. Normalize (Basic exact match or substring check for now)
            # This logic should likely be abstracted strategy
            normalized_label = self._normalize(raw_response, self.prompt_builder.spec.label_space)
            
            # 4. Collect Data
            record = {
                "model": self.model.model_name,
                "sentence_id": row['sentence_id'],
                "entity": row['entity'],
                "prompt_hash": prompt_hash,
                "raw_response": raw_response,
                "normalized_label": normalized_label,
                "timestamp": datetime.now().isoformat()
            }
            all_responses.append(record)
            
            if row['sentence_id'] not in predictions_by_sentence:
                predictions_by_sentence[row['sentence_id']] = []
            predictions_by_sentence[row['sentence_id']].append(normalized_label)
            
        # 5. Persist Raw
        if self.config['output'].get('save_raw', True):
            self.output_writer.write_responses(all_responses)
            
        # 6. Compute Metrics
        metric_results = []
        for metric in self.metrics:
            val = metric.compute(predictions_by_sentence)
            metric_results.append({
                "model": self.model.model_name,
                "metric_name": metric.name,
                "metric_value": val,
                "language": self.config['prompt'].get('language', 'en'),
                "timestamp": datetime.now().isoformat()
            })
            
        # 7. Persist Metrics
        if self.config['output'].get('save_metrics', True):
            self.output_writer.write_metrics(metric_results)

    def _normalize(self, response: str, label_space: List[str]) -> str:
        """
        Naive normalization: check if any label exists in the response.
        Prioritize longer labels to avoid substring matches (e.g., 'no' in 'not sure').
        """
        response_lower = response.lower()
        # Sort labels by length descending
        sorted_labels = sorted(label_space, key=len, reverse=True)
        
        for label in sorted_labels:
            if label.lower() in response_lower:
                return label 
        return "UNKNOWN"
