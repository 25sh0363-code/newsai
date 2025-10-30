# ML-Powered News Summarizer

An advanced news article summarization system using state-of-the-art machine learning models. This application supports both extractive and abstractive summarization techniques, along with sentiment analysis.

## Features

- **Dual Summarization Methods**
  - Extractive summarization using BERT
  - Abstractive summarization using T5
- **Smart Article Extraction**
  - Direct text input
  - URL-based article fetching
- **Sentiment Analysis**
  - Emotion classification
  - Confidence scoring
  - Visual indicators
- **Evaluation Metrics**
  - Word count comparison
  - Compression ratio
  - Processing time

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/news-summarizer.git
cd news-summarizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Choose your input method:
   - Paste article text directly
   - Enter article URL for automatic extraction

4. Select summarization settings:
   - Choose between extractive and abstractive summarization
   - Adjust summary length and parameters
   - Enable/disable sentiment analysis

5. Click "Summarize" and view the results

## Tech Stack

- **Python 3.8+**
- **Core Libraries:**
  - Hugging Face Transformers
  - NLTK
  - Streamlit
  - BeautifulSoup4/newspaper3k
  - PyTorch
  - scikit-learn

## Project Structure

```
news-summarizer/
├── app.py                 # Main Streamlit application
├── models/
│   ├── extractive.py     # BERT-based extractive summarizer
│   ├── abstractive.py    # T5-based abstractive summarizer
│   └── sentiment.py      # Sentiment analysis module
├── utils/
│   ├── scraper.py        # Web scraping utilities
│   ├── preprocessor.py   # Text preprocessing functions
│   └── metrics.py        # Evaluation metrics
├── requirements.txt       # Dependencies
├── config.py             # Configuration settings
└── README.md             # Documentation
```

## Model Details

### Extractive Summarizer
- Based on BERT (bert-base-uncased)
- Uses sentence embeddings and cosine similarity
- Ranks sentences by importance

### Abstractive Summarizer
- Based on T5 (t5-small)
- Generates paraphrased summaries
- Supports long text chunking

### Sentiment Analyzer
- Based on DistilBERT
- Pre-trained on SST-2 dataset
- Provides confidence scores

## Performance Considerations

- Models are cached to improve loading time
- GPU acceleration if available
- Automatic text chunking for long articles
- Configurable summary lengths

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing pre-trained models
- Streamlit for the web interface framework
- NLTK for text processing utilities