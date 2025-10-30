"""
Sentiment analysis using Hugging Face transformers.
"""

from transformers import pipeline
from typing import Dict, Optional

class SentimentAnalyzer:
    def __init__(self, model_name: str = 'distilbert-base-uncased-finetuned-sst-2-english', device: str = None):
        """Initialize the sentiment analyzer."""
        self.model_name = model_name
        self.device = device if device else -1 
        
    
        self.analyzer = pipeline(
            'sentiment-analysis',
            model=model_name,
            device=self.device
        )

    def analyze(self, text: str) -> Dict[str, any]:
        """Analyze sentiment of text."""
        try:
           
            result = self.analyzer(text[:1000])[0]  
            
           
            sentiment_map = {
                'POSITIVE': '😊 Positive',
                'NEGATIVE': '😞 Negative',
                'NEUTRAL': '😐 Neutral'
            }
            
            sentiment = sentiment_map.get(result['label'], result['label'])
            confidence = result['score']
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'sentiment': '😐 Neutral',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }

    def get_color(self, sentiment: str) -> str:
        """Get color code for sentiment visualization."""
        color_map = {
            '😊 Positive': '#28a745',  
            '😞 Negative': '#dc3545',  
            '😐 Neutral': '#6c757d'    
        }
        return color_map.get(sentiment, '#6c757d')
