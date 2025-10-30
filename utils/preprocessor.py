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
    # Prefer a local nltk_data directory inside the project so Streamlit/venv
    # environments can write there and we have deterministic behavior.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_nltk_dir = os.path.join(project_root, 'nltk_data')
    if not os.path.isdir(local_nltk_dir):
        try:
            os.makedirs(local_nltk_dir, exist_ok=True)
        except Exception:
            # If we cannot create the directory, fall back to default downloader behavior
            local_nltk_dir = None

    # Ensure NLTK looks in our local directory first
    if local_nltk_dir and local_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_dir)

    # Download required corpora if missing, prefer local download_dir
    try:
        sent_tokenize("Test sentence.")
    except LookupError:
        # Newer NLTK versions may expect 'punkt_tab' data. Try that first,
        # then fall back to the classic 'punkt' package.
        try:
            nltk.download('punkt_tab', download_dir=local_nltk_dir)
        except Exception:
            # Fallback to classic punkt
            nltk.download('punkt', download_dir=local_nltk_dir)

    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', download_dir=local_nltk_dir)

class TextPreprocessor:
    def __init__(self, language: str = 'english'):
        """Initialize the text preprocessor."""
        # Ensure NLTK data is available
        ensure_nltk_data()
        
        self.language = language
        self.stop_words = set(stopwords.words(language))

    def clean_text(self, text: str) -> str:
        """Remove special characters and excessive whitespace."""
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove extra whitespace
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
        # Clean text
        text = self.clean_text(text)
        
        # Optionally remove stopwords
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