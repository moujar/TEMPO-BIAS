import csv
import os
from datetime import datetime

class OutputWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def write_responses(self, responses: list):
        path = os.path.join(self.output_dir, "responses.csv")
        fieldnames = ["model", "sentence_id", "entity", "prompt_hash", "raw_response", "normalized_label", "timestamp"]
        
        file_exists = os.path.exists(path)
        
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(responses)

    def write_metrics(self, metrics: list):
        path = os.path.join(self.output_dir, "metrics.csv")
        fieldnames = ["model", "metric_name", "metric_value", "language", "timestamp"]
        
        file_exists = os.path.exists(path)
        
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(metrics)
