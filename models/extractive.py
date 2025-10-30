"""
BERT-based extractive summarizer.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ExtractiveSummarizer:
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = None):
        """Initialize the extractive summarizer."""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
       
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
        self.model.eval()

    def _get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Get BERT embeddings for sentences."""
        embeddings = []
        
        with torch.no_grad():
            for sentence in sentences:
               
                inputs = self.tokenizer(sentence, return_tensors='pt', 
                                      padding=True, truncation=True, 
                                      max_length=512).to(self.device)
                
             
                outputs = self.model(**inputs)
                
             
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)

    def _rank_sentences(self, embeddings: np.ndarray) -> np.ndarray:
        """Rank sentences based on cosine similarity."""
   
        similarity_matrix = cosine_similarity(embeddings)
        
       
        scores = np.sum(similarity_matrix, axis=1)
        
        
        ranked_sentences = np.argsort(scores)[::-1]
        
        return ranked_sentences

    def summarize(self, sentences: List[str], num_sentences: int = 3) -> Dict[str, any]:
        """Generate extractive summary."""
        if not sentences:
            return {
                'summary': '',
                'selected_indices': [],
                'success': False,
                'error': 'No sentences provided'
            }
            
        try:
           
            num_sentences = min(num_sentences, len(sentences))
            
           
            embeddings = self._get_embeddings(sentences)
            
           
            ranked_sentences = self._rank_sentences(embeddings)
            
          
            selected_indices = sorted([int(i) for i in ranked_sentences[:num_sentences]])
            
            
            summary = ' '.join([sentences[i] for i in selected_indices])
            
            return {
                'summary': summary,
                'selected_indices': selected_indices,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'summary': '',
                'selected_indices': [],
                'success': False,
                'error': str(e)
            }
