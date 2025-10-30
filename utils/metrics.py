"""
Evaluation metrics for summarization quality.
"""

from typing import List, Dict
from rouge_score import rouge_scorer
import time

class EvaluationMetrics:
    def __init__(self):
        """Initialize the evaluation metrics calculator."""
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def calculate_rouge_scores(self, reference: str, summary: str) -> Dict[str, float]:
        """Calculate ROUGE scores between the reference and generated summary."""
        scores = self.scorer.score(reference, summary)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def calculate_compression_ratio(self, original_text: str, summary: str) -> float:
        """Calculate compression ratio between original text and summary."""
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        return (summary_words / original_words) * 100 if original_words > 0 else 0
    
    def word_counts(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
class Timer:
    def __enter__(self):
        """Start timing."""
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        """Stop timing."""
        self.end = time.time()
        self.duration = self.end - self.start