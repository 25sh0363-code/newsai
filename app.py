"""
Main Streamlit application for the ML-powered news summarizer.
"""

import streamlit as st
import torch
from models.extractive import ExtractiveSummarizer
from models.abstractive import AbstractiveSummarizer
from models.sentiment import SentimentAnalyzer
from utils.scraper import ArticleScraper
from utils.preprocessor import TextPreprocessor
from utils.metrics import EvaluationMetrics, Timer
from config import MODELS, UI_CONFIG

# Initialize session state
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()
if 'metrics' not in st.session_state:
    st.session_state.metrics = EvaluationMetrics()
if 'scraper' not in st.session_state:
    st.session_state.scraper = ArticleScraper()

@st.cache_resource
def load_extractive_model():
    """Load extractive summarization model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return ExtractiveSummarizer(
        model_name=MODELS['extractive']['name'],
        device=device
    )

@st.cache_resource
def load_abstractive_model():
    """Load abstractive summarization model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return AbstractiveSummarizer(
        model_name=MODELS['abstractive']['name'],
        device=device
    )

@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentimentAnalyzer(
        model_name=MODELS['sentiment']['name'],
        device=device
    )

def main():
    # Page config
    st.set_page_config(
        page_title="📰 MLehheheh-Powered News Summarizer",
        page_icon="📰",
        layout="wide"
    )
    
    # Title
    st.title("📰 ML-Powered News Summarizer")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Input Method",
        ["Text", "URL"]
    )
    
    # Summarization mode
    summary_mode = st.sidebar.radio(
        "Summarization Mode",
        ["Abstractive", "Extractive"]
    )
    
    # Summary length
    if summary_mode == "Extractive":
        num_sentences = st.sidebar.slider(
            "Number of Sentences",
            min_value=1,
            max_value=UI_CONFIG['max_summary_sentences'],
            value=UI_CONFIG['default_summary_sentences']
        )
    else:
        max_length = st.sidebar.slider(
            "Maximum Summary Length (tokens)",
            min_value=50,
            max_value=200,
            value=MODELS['abstractive']['max_length']
        )
        min_length = st.sidebar.slider(
            "Minimum Summary Length (tokens)",
            min_value=30,
            max_value=100,
            value=MODELS['abstractive']['min_length']
        )
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=MODELS['abstractive']['temperature']
        )
    
    # Enable sentiment analysis
    enable_sentiment = st.sidebar.checkbox("Enable Sentiment Analysis", value=True)
    
    # Main content area
    if input_method == "Text":
        text_input = st.text_area(
            "Paste your article text here",
            height=200,
            max_chars=UI_CONFIG['max_input_length']
        )
        
    else:  # URL input
        url_input = st.text_input("Enter article URL")
        
        if url_input:
            if not st.session_state.scraper.validate_url(url_input):
                st.error("Please enter a valid URL")
                return
                
            with st.spinner("Fetching article..."):
                article = st.session_state.scraper.extract_from_url(url_input)
                
                if not article['success']:
                    st.error(f"Failed to fetch article: {article['error']}")
                    return
                    
                text_input = article['text']
                if article['title']:
                    st.subheader(f"Article Title: {article['title']}")
        else:
            text_input = ""
    
    # Process text when available
    if text_input:
        with st.spinner("Processing..."):
            # Preprocess text
            clean_text = st.session_state.preprocessor.preprocess(text_input)
            sentences = st.session_state.preprocessor.tokenize_sentences(clean_text)
            
            # Generate summary
            with Timer() as timer:
                if summary_mode == "Extractive":
                    with st.spinner("Loading extractive model..."):
                        model = load_extractive_model()
                    result = model.summarize(
                        sentences,
                        num_sentences=num_sentences
                    )
                else:
                    with st.spinner("Loading abstractive model..."):
                        model = load_abstractive_model()
                    result = model.chunk_and_summarize(
                        clean_text,
                        max_length=max_length,
                        min_length=min_length,
                        temperature=temperature
                    )
            
            if not result['success']:
                st.error(f"Failed to generate summary: {result['error']}")
                return
            
            # Display results
            st.header("Summary Results")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Original Length",
                    f"{st.session_state.metrics.word_counts(text_input)} words"
                )
            
            with col2:
                st.metric(
                    "Summary Length",
                    f"{st.session_state.metrics.word_counts(result['summary'])} words"
                )
            
            with col3:
                compression = st.session_state.metrics.calculate_compression_ratio(
                    text_input,
                    result['summary']
                )
                st.metric(
                    "Compression Ratio",
                    f"{compression:.1f}%"
                )
            
            # Display summary
            st.subheader("Generated Summary")
            st.write(result['summary'])
            
            # Display processing time
            st.info(f"Processing time: {timer.duration:.2f} seconds")
            
            # Sentiment analysis
            if enable_sentiment:
                st.subheader("Sentiment Analysis")
                
                with st.spinner("Loading sentiment model..."):
                    model = load_sentiment_model()
                sentiment_result = model.analyze(result['summary'])
                
                if sentiment_result['success']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(
                            f"<h3 style='color: {model.get_color(sentiment_result['sentiment'])};'>"
                            f"{sentiment_result['sentiment']}</h3>",
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.progress(sentiment_result['confidence'])
                        st.text(f"Confidence: {sentiment_result['confidence']:.2%}")
                else:
                    st.error(f"Failed to analyze sentiment: {sentiment_result['error']}")

if __name__ == "__main__":
    main()
