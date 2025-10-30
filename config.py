# Configuration settings for the news summarizer

# Model configurations
MODELS = {
    'extractive': {
        'name': 'bert-base-uncased',
        'max_length': 512,
        'num_sentences': 5
    },
    'abstractive': {
        'name': 't5-small',
        'max_length': 150,
        'min_length': 40,
        'temperature': 0.7
    },
    'sentiment': {
        'name': 'distilbert-base-uncased-finetuned-sst-2-english'
    }
}

# Text preprocessing settings
TEXT_CLEANING = {
    'remove_special_chars': True,
    'remove_stopwords': True,
    'language': 'english'
}

# UI configurations
UI_CONFIG = {
    'max_input_length': 10000,  # Maximum characters allowed in text input
    'default_summary_sentences': 3,
    'max_summary_sentences': 10
}

# Cache settings
CACHE_CONFIG = {
    'model_cache_dir': '.model_cache',
    'enable_caching': True
}