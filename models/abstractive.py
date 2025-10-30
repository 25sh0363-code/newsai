"""
T5-based abstractive summarizer.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Optional

class AbstractiveSummarizer:
    def __init__(self, model_name: str = 't5-small', device: str = None):
        """Initialize the abstractive summarizer."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        try:
            
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        except ImportError as e:
     
            raise ImportError(
                "T5Tokenizer requires the 'sentencepiece' package. "
                "Add 'sentencepiece' to your requirements.txt and redeploy (pip package name: sentencepiece)."
            ) from e
        
    
        self.model.eval()

    def summarize(self, 
                 text: str, 
                 max_length: int = 150,
                 min_length: int = 40,
                 temperature: float = 0.7) -> Dict[str, any]:
        """Generate abstractive summary."""
        try:
          
            input_text = f"summarize: {text}"
            
         
            inputs = self.tokenizer(input_text, 
                                  return_tensors='pt',
                                  max_length=1024,
                                  truncation=True).to(self.device)
     
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=max_length,
                    min_length=min_length,
                    temperature=temperature,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
      
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'summary': summary,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'summary': '',
                'success': False,
                'error': str(e)
            }

    def chunk_and_summarize(self, 
                          text: str,
                          chunk_size: int = 1000,
                          max_length: int = 150,
                          min_length: int = 40,
                          temperature: float = 0.7) -> Dict[str, any]:
        """Handle long texts by chunking and summarizing each chunk."""
        try:
         
            words = text.split()
            chunks = [' '.join(words[i:i + chunk_size]) 
                     for i in range(0, len(words), chunk_size)]
            
            
            chunk_summaries = []
            for chunk in chunks:
                result = self.summarize(chunk, max_length, min_length, temperature)
                if result['success']:
                    chunk_summaries.append(result['summary'])
            
            if not chunk_summaries:
                return {
                    'summary': '',
                    'success': False,
                    'error': 'Failed to generate summaries for chunks'
                }
            
          
            if len(chunk_summaries) > 1:
                combined = ' '.join(chunk_summaries)
                return self.summarize(combined, max_length, min_length, temperature)
            
            return {
                'summary': chunk_summaries[0],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'summary': '',
                'success': False,
                'error': str(e)
            }
