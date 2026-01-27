import numpy as np
from scipy.stats import entropy
from collections import Counter
from .base import BaseMetric

class PredictionInconsistency(BaseMetric):
    def __init__(self):
        super().__init__("IC")

    def compute(self, predictions: dict) -> float:
        """
        predictions: {sentence_id: [normalized_label_1, normalized_label_2, ...]}
        """
        sentence_entropies = []
        
        for sent_id, labels in predictions.items():
            if not labels:
                continue
            
            # Count frequencies
            counts = Counter(labels)
            total = len(labels)
            probs = [c / total for c in counts.values()]
            
            # Compute entropy (base e by default, can use base 2)
            ent = entropy(probs)
            sentence_entropies.append(ent)
            
        return float(np.mean(sentence_entropies)) if sentence_entropies else 0.0
