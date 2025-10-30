"""
Text preprocessing utilities for news article processing.
"""

import os
import re
import nltk
from typing import List, Optional
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_nltk_dir = os.path.join(project_root, 'nltk_data')
    if not os.path.isdir(local_nltk_dir):
        try:
            os.makedirs(local_nltk_dir, exist_ok=True)
        except Exception:
            
            local_nltk_dir = None

    if local_nltk_dir and local_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_dir)

  
    try:
        sent_tokenize("Test sentence.")
    except LookupError:
     
        try:
            nltk.download('punkt_tab', download_dir=local_nltk_dir)
        except Exception:
           
            nltk.download('punkt', download_dir=local_nltk_dir)

    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', download_dir=local_nltk_dir)

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        """Initialize the text preprocessor."""
     
        ensure_nltk_data()
        
        self.language = language
        self.stop_words = set(stopwords.words(language))

    def clean_text(self, text: str) -> str:
        """Remove special characters and excessive whitespace."""
       
        text = re.sub(r'[^\w\s.,!?]', '', text)
      
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)

    def tokenize_words(self, text: str) -> List[str]:
        """Split text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, text: str) -> str:
        """Remove stop words from text."""
        words = self.tokenize_words(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def preprocess(self, text: str, remove_stopwords: bool = False) -> str:
        """Apply full preprocessing pipeline."""
       
        text = self.clean_text(text)
        
        
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text

    def split_into_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split long text into smaller chunks for model processing."""
        sentences = self.tokenize_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
