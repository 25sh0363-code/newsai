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
        
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()

    def summarize(self, 
                 text: str, 
                 max_length: int = 150,
                 min_length: int = 40,
                 temperature: float = 0.7) -> Dict[str, any]:
        """Generate abstractive summary."""
        try:
            # Prepare input text
            input_text = f"summarize: {text}"
            
            # Tokenize input
            inputs = self.tokenizer(input_text, 
                                  return_tensors='pt',
                                  max_length=1024,
                                  truncation=True).to(self.device)
            
            # Generate summary
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
            
            # Decode summary
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
            # Split text into chunks
            words = text.split()
            chunks = [' '.join(words[i:i + chunk_size]) 
                     for i in range(0, len(words), chunk_size)]
            
            # Summarize each chunk
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
            
            # If multiple chunks, summarize the combined summaries
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